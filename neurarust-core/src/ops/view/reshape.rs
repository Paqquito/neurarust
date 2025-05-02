use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

/// Performs the reshape operation. Currently only supports creating a view
/// for contiguous tensors. For non-contiguous tensors, call `.contiguous()` first.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `new_shape_vec`: The desired new shape.
///
/// # Returns
/// A new Tensor representing the reshaped view, or an error.
pub(crate) fn reshape_op<T>(
    tensor: &Tensor<T>,
    new_shape_vec: Vec<usize>,
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Default
        + Send
        + Sync
        + 'static
        + Debug
        + Copy
        + Zero
        + AddAssign
        + PartialEq
        + PartialOrd
        + Sum
        + One,
{
    // --- Autograd Setup ---
    let requires_grad = tensor.requires_grad();
    let mut input_id_maybe: Option<NodeId<T>> = None;
    let mut original_shape_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
        original_shape_maybe = Some(tensor.shape());
    }
    // --- End Autograd Setup ---

    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for reshape".to_string(),
        )
    })?;

    // 2. Validate number of elements
    let original_numel: usize = guard.shape.iter().product();
    let new_numel: usize = new_shape_vec.iter().product();

    if original_numel != new_numel {
        return Err(NeuraRustError::ShapeMismatch {
            expected: guard.shape.clone(),
            actual: new_shape_vec,
            operation: "reshape".to_string(),
        });
    }

    // 3. Check for contiguity (current limitation)
    if !guard.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(
            "Reshape currently only supports contiguous tensors. Call .contiguous() first."
                .to_string(),
        ));
    }

    // 4. If contiguous and numel matches, calculate new strides and create view
    let new_strides = TensorData::<T>::calculate_contiguous_strides(&new_shape_vec);

    // Get necessary info from locked data
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    drop(guard);

    // Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape_vec.clone(),
        new_strides,
    );

    // Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        let original_shape = original_shape_maybe.ok_or_else(|| NeuraRustError::InternalError("Missing original shape for reshape backward pass".to_string()))?;

        let backward_context = ReshapeBackward {
            input_id: input_id_maybe.unwrap(),
            original_shape,
            _phantom: std::marker::PhantomData,
        };

        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        new_tensor.set_requires_grad(true)?;
        new_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(new_tensor)
}

// --- Reshape Backward Operation ---

#[derive(Debug)]
struct ReshapeBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    original_shape: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for ReshapeBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for ReshapeBackward<T> {}

impl<T> BackwardOp<T> for ReshapeBackward<T>
where
    T: Default
        + Send
        + Sync
        + 'static
        + Debug
        + Copy
        + Zero
        + AddAssign
        + PartialEq
        + PartialOrd
        + Sum
        + One,
{
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input_id]
    }

    /// Computes the gradient for the input tensor of the reshape operation.
    /// Since reshape (as implemented here as a view) only changes metadata,
    /// the backward pass simply reshapes the incoming gradient back to the
    /// original input tensor's shape.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let _input_id = self.input_id; // Keep field access if needed elsewhere potentially
        let input_grad = reshape_op(grad_output, self.original_shape.clone())?;
        Ok(vec![input_grad])
    }
}

// --- Tests for Reshape Op ---
#[cfg(test)]
mod tests {
     // Importe reshape_op, etc.
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::utils::testing::create_test_tensor_with_grad;
    
    

    #[test]
    fn test_reshape_backward() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].reshape(vec![3, 2])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=6).map(|x| x as f64).collect(),
            vec![2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![3, 2]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for reshape: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward_flatten() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            let numel = inputs[0].numel();
            inputs[0].reshape(vec![numel])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=12).map(|x| x as f64).collect(),
            vec![2, 2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![12]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for flatten reshape: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward_add_dim() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].reshape(vec![2, 2, 1, 3])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=12).map(|x| x as f64).collect(),
            vec![2, 2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![2, 2, 1, 3]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for reshape adding dim: {:?}", result.err());
    }

} 