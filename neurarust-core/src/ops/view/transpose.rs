use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::sync::{Arc, RwLock};
use std::fmt::Debug;
use super::utils;

/// Performs the transpose operation between two dimensions, creating a view.
///
/// This is a crate-internal function, typically called via the `Tensor::transpose` method.
/// It swaps the specified dimensions `dim1` and `dim2` of the input tensor,
/// returning a new tensor view without copying data. The underlying data buffer
/// and offset remain the same, only the shape and strides are modified.
///
/// # Arguments
///
/// * `input` - The input tensor.
/// * `dim1` - The first dimension to transpose.
/// * `dim2` - The second dimension to transpose.
///
/// # Returns
///
/// A `Result` containing the transposed `Tensor` view. Returns an error if:
/// *   `dim1` or `dim2` are out of bounds for the tensor's rank.
/// *   Device or autograd operations fail.
///
/// # Example (Conceptual - Use `Tensor::transpose` instead)
///
/// ```rust,ignore
/// // Assuming t is a Tensor of shape [2, 3, 4]
/// // use crate::ops::view::transpose::transpose_op; // Assuming direct access
///
/// // Transpose dimensions 0 and 2
/// let transposed_view = transpose_op(&t, 0, 2)?;
/// // transposed_view will have shape [4, 3, 2]
/// ```
pub fn transpose_op(input: &Tensor, dim1: usize, dim2: usize) -> Result<Tensor, NeuraRustError> {
    let input_data_guard = input.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock input TensorData for read in transpose_op".to_string(),
    })?;

    // Validate dimensions
    utils::validate_transpose_dims(input_data_guard.shape.len(), dim1, dim2)?;

    // Calculate new shape and strides
    let new_shape = {
        let mut current_shape = input_data_guard.shape.clone();
        current_shape.swap(dim1, dim2); // Swap dimensions in shape
        current_shape
    };
    let new_strides = {
        let mut current_strides = input_data_guard.strides.clone();
        current_strides.swap(dim1, dim2);
        current_strides
    };
    let offset = input_data_guard.offset;
    let device = input_data_guard.device;
    let buffer_arc = Arc::clone(&input_data_guard.buffer); // Clone the Arc<Buffer>
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };
    let original_shape_clone = input_data_guard.shape.clone(); // For backward

    // Drop guard before creating new TensorData
    drop(input_data_guard);

    // Create the view TensorData
    let view_td = TensorData::new_view(buffer_arc, device, offset, new_shape, new_strides)?;

    // Create the output tensor
    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    // --- Autograd Setup --- (If input requires grad)
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
             let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                 lock_type: "write".to_string(),
                 reason: "Failed to lock output TensorData for write (autograd setup in transpose_op)".to_string(),
             })?;
             output_data_write_guard.requires_grad = true;
             let backward_op = TransposeBackward {
                 input_node: node_arc,
                 dim1, // Store transposed dims for backward
                 dim2,
                 _original_shape: original_shape_clone,
             };
             output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
            return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in transpose_op".to_string()));
        }
    }

    Ok(output_tensor)
}

/// Backward operation context for the `transpose` operation.
///
/// Stores the original input tensor node and the dimensions that were swapped
/// (`dim1`, `dim2`) during the forward pass. This information is needed to
/// transpose the incoming gradient correctly.
#[derive(Debug)]
struct TransposeBackward {
    input_node: Arc<RwLock<TensorData>>,
    dim1: usize,
    dim2: usize,
    _original_shape: Vec<usize>,
}

impl BackwardOp for TransposeBackward {
    /// Computes the gradient for the transpose operation.
    ///
    /// The gradient of a transpose operation is simply the transpose of the incoming
    /// gradient (`grad_output`) with the same dimensions swapped back.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original transpose operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with a single element: the gradient
    /// with respect to the original input tensor (which is the transposed `grad_output`).
    /// Returns an error if the transpose operation on the gradient fails.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // Transposing the gradient is the same as transposing the input
        let grad_input = transpose_op(grad_output, self.dim1, self.dim2)?;

        // The autograd engine will handle accumulation. Just return the calculated grad.
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// Link the external tests file
#[cfg(test)]
#[path = "transpose_test.rs"] mod tests; 