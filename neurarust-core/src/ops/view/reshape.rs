use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
// Restore necessary imports

/// Reshapes the tensor into a new shape, creating a view if possible.
///
/// This is a crate-internal function, typically called via the `Tensor::reshape` method.
/// It attempts to create a view of the input tensor with the `new_shape_vec`.
///
/// **Constraint:** This operation can only create a view (without copying data) if the
/// input tensor is **contiguous**. If the input tensor is not contiguous, this function
/// will return an error. In such cases, call `.contiguous()` on the tensor before reshaping
/// to obtain a reshaped tensor (which will involve a data copy).
///
/// The total number of elements must remain the same between the original shape and the new shape.
///
/// # Arguments
///
/// * `input` - The input tensor (must be contiguous).
/// * `new_shape_vec` - The desired new shape as a `Vec<usize>`.
///
/// # Returns
///
/// A `Result` containing the reshaped `Tensor` view. Returns an error if:
/// *   The total number of elements differs between the input and target shapes.
/// *   The input tensor is not contiguous.
/// *   Device or autograd operations fail.
///
/// # Example (Conceptual - Use `Tensor::reshape` instead)
///
/// ```rust,ignore
/// // Assuming t is a contiguous Tensor of shape [2, 6]
/// // use crate::ops::view::reshape::reshape_op; // Assuming direct access
///
/// let reshaped_view = reshape_op(&t, vec![4, 3])?;
/// // reshaped_view will have shape [4, 3]
///
/// // Assuming non_contig is a non-contiguous Tensor of shape [2, 6]
/// // let error = reshape_op(&non_contig, vec![4, 3]); // This will return an error
/// // assert!(error.is_err());
///
/// // Correct approach for non-contiguous:
/// // let reshaped_copied = non_contig.contiguous()?.reshape(vec![4, 3])?;
/// ```
pub(crate) fn reshape_op(input: &Tensor, new_shape_vec: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    let input_data_guard = input.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock input TensorData for read in reshape_op".to_string(),
    })?;

    let input_shape = input_data_guard.shape.clone();
    let numel: usize = input_shape.iter().product();
    let new_numel: usize = new_shape_vec.iter().product();

    if numel != new_numel {
        return Err(NeuraRustError::ShapeMismatch {
            expected: format!("numel={}", numel),
            actual: format!("numel={}", new_numel),
            operation: "reshape".to_string(),
        });
    }

    // Reshape is only possible on contiguous tensors without copying
    if !input_data_guard.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(
            "Reshape requires a contiguous tensor. Call .contiguous() first.".to_string(),
        ));
    }

    // Create view: reuse buffer, offset, device; update shape, calculate new strides
    let new_strides = TensorData::calculate_contiguous_strides(&new_shape_vec);
    let buffer_arc = Arc::clone(&input_data_guard.buffer);
    let device = input_data_guard.device;
    let offset = input_data_guard.offset;
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };

    drop(input_data_guard); // Drop lock before creating new TensorData

    let view_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape_vec.clone(), // Use the input shape vec
        new_strides,
    )?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    // Autograd setup
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                 lock_type: "write".to_string(),
                 reason: "Failed to lock output TensorData for write (autograd setup in reshape_op)".to_string(),
             })?;
             output_data_write_guard.requires_grad = true;
             let backward_op = ReshapeBackward {
                 input_node: node_arc,
                 original_shape: input_shape, // Store original shape for backward
             };
             output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in reshape_op".to_string()));
        }
    }

    Ok(output_tensor)
}

// --- Reshape Backward Operation ---

/// Backward operation context for the `reshape` operation.
///
/// Stores the original input tensor node and the original shape (`original_shape`)
/// before the reshape operation was applied. This shape is needed to reshape
/// the incoming gradient back during the backward pass.
#[derive(Debug)]
struct ReshapeBackward {
    input_node: Arc<RwLock<TensorData>>,
    original_shape: Vec<usize>,
}

impl BackwardOp for ReshapeBackward {
    /// Computes the gradient for the reshape operation.
    ///
    /// The gradient of a reshape operation is simply the incoming gradient (`grad_output`)
    /// reshaped back to the original input tensor's shape.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original reshape operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with a single element: the gradient
    /// with respect to the original input tensor (which is the reshaped `grad_output`).
    /// Returns an error if the reshape operation on the gradient fails.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // Reshape the gradient back to the original input shape
        reshape_op(grad_output, self.original_shape.clone())
             .map(|grad_input| vec![grad_input]) // Wrap the result in a Vec
             .map_err(|e| NeuraRustError::BackwardError(format!("Error in ReshapeBackward: {}", e)))
     }
 
     fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
         vec![Arc::as_ptr(&self.input_node)]
     }
}

// Link the external tests file
#[cfg(test)]
#[path = "reshape_test.rs"] mod tests; 