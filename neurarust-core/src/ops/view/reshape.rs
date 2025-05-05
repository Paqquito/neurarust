use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
// Restore necessary imports

/// Performs the reshape operation. Currently only supports creating a view
/// for contiguous tensors. For non-contiguous tensors, call `.contiguous()` first.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `new_shape_vec`: The desired new shape.
///
/// # Returns
/// A new Tensor representing the reshaped view, or an error.
pub fn reshape_op(input: &Tensor, new_shape_vec: Vec<usize>) -> Result<Tensor, NeuraRustError> {
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

#[derive(Debug)]
struct ReshapeBackward {
    input_node: Arc<RwLock<TensorData>>,
    original_shape: Vec<usize>,
}

impl BackwardOp for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        reshape_op(grad_output, self.original_shape.clone())
             .map(|grad_input| vec![grad_input])
             .map_err(|e| NeuraRustError::BackwardError(format!("Error in ReshapeBackward: {}", e)))
     }
 
     fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
         vec![Arc::as_ptr(&self.input_node)]
     }
}

// Link the external tests file
#[cfg(test)]
#[path = "reshape_test.rs"] mod tests; 