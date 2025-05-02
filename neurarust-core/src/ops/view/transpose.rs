use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

/// Performs the transpose operation between two dimensions, creating a view.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `dim1`: The first dimension to transpose.
/// * `dim2`: The second dimension to transpose.
///
/// # Returns
/// A new Tensor representing the transposed view, or an error.
pub(crate) fn transpose_op<T>(
    tensor: &Tensor<T>,
    dim1: usize,
    dim2: usize,
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

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
    }

    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for transpose".to_string(),
        )
    })?;

    let rank = guard.shape.len();

    // 2. Validate dimensions
    if dim1 >= rank || dim2 >= rank {
        return Err(NeuraRustError::DimensionMismatch {
            expected: rank,
            actual: std::cmp::max(dim1, dim2),
        });
    }

    // 3. Calculate new shape and strides
    let mut new_shape = guard.shape.clone();
    let mut new_strides = guard.strides.clone();

    // Swap shape and strides at dim1 and dim2
    new_shape.swap(dim1, dim2);
    new_strides.swap(dim1, dim2);

    // 4. Get other necessary info
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    // Drop the read guard before creating the new RwLock
    drop(guard);

    // 5. Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape,
        new_strides,
    );

    // 6. Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        let backward_context = TransposeBackward {
            input_id: input_id_maybe.unwrap(),
            dim1,
            dim2,
            _phantom: std::marker::PhantomData,
        };

        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        new_tensor.set_requires_grad(true)?;
        new_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(new_tensor)
}

// --- Transpose Backward Operation ---

#[derive(Debug)]
struct TransposeBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    dim1: usize,
    dim2: usize,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for TransposeBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for TransposeBackward<T> {}

impl<T> BackwardOp<T> for TransposeBackward<T>
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

    /// Computes the gradient for the input tensor of the transpose operation.
    /// Since transpose is just rearranging data (a view), the backward pass
    /// simply applies the *same* transpose operation to the incoming gradient.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let input_grad = transpose_op(grad_output, self.dim1, self.dim2)?;
        Ok(vec![input_grad])
    }
}

// --- Tests for Transpose Op ---
#[cfg(test)]
mod tests {
    use super::*; // Importe transpose_op, etc.
     // Pour test_transpose_backward
    use crate::tensor::Tensor;
    use crate::utils::testing::{create_test_tensor, create_test_tensor_with_grad};
    
    use std::sync::Arc;

    #[test]
    fn test_transpose_basic() {
        let t = create_test_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let transposed = t.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), vec![3, 2]);

        let t_guard = t.read_data();
        let transposed_guard = transposed.read_data();
        assert_eq!(Arc::as_ptr(&t_guard.data), Arc::as_ptr(&transposed_guard.data), "Transpose should share buffer");
        assert_eq!(transposed_guard.offset, t_guard.offset);
        assert_eq!(transposed_guard.strides, vec![1, 3]);
        assert!(!transposed.is_contiguous());

        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[0, 0])], 1.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[0, 1])], 4.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[1, 0])], 2.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[1, 1])], 5.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[2, 0])], 3.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[2, 1])], 6.0);
    }

    #[test]
    fn test_transpose_backward() {
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![3, 2]).unwrap();

        // Calculate analytical gradient
        let output = input_data.transpose(0, 1).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is transpose(output_grad_val)
        let expected_grad = transpose_op(&output_grad_val, 0, 1).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

    #[test]
    fn test_transpose_backward_higher_dim() {
        let input_data = create_test_tensor_with_grad(
            (1..=24).map(|x| x as f64).collect(),
            vec![2, 3, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![2, 4, 3]).unwrap();

        // Calculate analytical gradient
        let output = input_data.transpose(1, 2).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is transpose(output_grad_val)
        let expected_grad = transpose_op(&output_grad_val, 1, 2).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

} 