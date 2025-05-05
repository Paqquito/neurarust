use crate::autograd::BackwardOp;
 // Non-generic
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use super::utils; // Use super::utils

use std::sync::{Arc, RwLock};
use std::fmt::Debug;
 // For test signatures

// --- Backward Operation Structure ---
#[derive(Debug)]
struct PermuteBackward {
    input_node: Arc<RwLock<TensorData>>,
    original_axes: Vec<usize>,
}

impl PermuteBackward {
    // Helper to find the inverse permutation
    fn inverse_axes(&self) -> Vec<usize> {
        let mut inverse = vec![0; self.original_axes.len()];
        for (i, &axis) in self.original_axes.iter().enumerate() {
            inverse[axis] = i;
        }
        inverse
    }
}

// --- Backward Operation Implementation ---
impl BackwardOp for PermuteBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let inverse_axes = self.inverse_axes();
        
        // Appeler permute_op avec les axes inverses sur le gradient entrant original
        let grad_input = permute_op(grad_output, &inverse_axes)?;

        // Retourner le gradient calculé dans un vecteur
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- Forward Operation ---
pub fn permute_op(input: &Tensor, dims: &[usize]) -> Result<Tensor, NeuraRustError> {
    let input_data_guard = input.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock input TensorData for read in permute_op".to_string(),
    })?;

    let rank = input_data_guard.shape.len();
    // Validate permutation dimensions
    utils::validate_permutation(rank, dims)?;

    // Calculate new shape and strides
    let new_shape = utils::permute_shape(&input_data_guard.shape, dims);
    let new_strides = utils::permute_strides(&input_data_guard.strides, dims);
    let offset = input_data_guard.offset;
    let device = input_data_guard.device;
    let buffer_arc = Arc::clone(&input_data_guard.buffer);
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };
    let original_dims_clone = dims.to_vec(); // For backward

    drop(input_data_guard);

    let view_td = TensorData::new_view(buffer_arc, device, offset, new_shape, new_strides)?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    // Autograd setup
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                 lock_type: "write".to_string(),
                 reason: "Failed to lock output TensorData for write (autograd setup in permute_op)".to_string(),
             })?;
             output_data_write_guard.requires_grad = true;
             let backward_op = PermuteBackward {
                 input_node: node_arc,
                 original_axes: original_dims_clone, // Store original permutation
             };
             output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in permute_op".to_string()));
        }
    }

    Ok(output_tensor)
}

// Link the external tests file
#[cfg(test)]
#[path = "permute_test.rs"] mod tests; 