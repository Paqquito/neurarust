use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;

use std::sync::{Arc, RwLock};
use std::fmt::Debug;

// --- Backward Operation Structure ---
#[derive(Debug)]
struct ExpandBackward {
    input_node: Arc<RwLock<TensorData>>,
    original_shape: Vec<usize>,
}

// --- Backward Operation Implementation ---
impl BackwardOp for ExpandBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let grad_shape = grad_output.shape();
        let grad_rank = grad_shape.len();
        let original_rank = self.original_shape.len();
        let rank_diff = grad_rank - original_rank;

        let mut axes_to_sum = Vec::new();

        // 1. Sum across the newly added dimensions (at the front)
        for i in 0..rank_diff {
            axes_to_sum.push(i);
        }

        // 2. Sum across dimensions that were expanded from 1
        for i in 0..original_rank {
            let original_dim = self.original_shape[i];
            let grad_dim_idx = i + rank_diff;
            let grad_dim = grad_shape[grad_dim_idx];

            if original_dim == 1 && grad_dim > 1 {
                axes_to_sum.push(grad_dim_idx);
            }
        }

        // 3. Perform summation
        let grad_input = if !axes_to_sum.is_empty() {
            // Use keep_dims=false to remove summed dimensions
            crate::ops::reduction::sum_op(grad_output, Some(&axes_to_sum), false)? 
        } else {
            // If no axes were summed, the gradient shape already matches original shape
            grad_output.clone()
        };
        
        // Verify the shape just in case (optional debug assert)
        // assert_eq!(grad_input.shape(), self.original_shape, "Gradient shape mismatch after sum");

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Validates shapes for the expand operation.
fn validate_expand_shapes(
    tensor: &Tensor,
    target_shape: &[usize],
) -> Result<(), NeuraRustError> {
    let tensor_shape = tensor.shape();
    let tensor_rank = tensor_shape.len();
    let target_rank = target_shape.len();

    if tensor_rank > target_rank {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "expand".to_string(),
            expected: format!("Input rank ({}) <= Target rank ({})", tensor_rank, target_rank),
            actual: "Input rank > Target rank".to_string(),
        });
    }

    let mut tensor_idx = (tensor_rank as isize) - 1;
    for target_idx in (0..target_rank).rev() {
        let target_dim = target_shape[target_idx];

        if tensor_idx >= 0 {
            // Dimension exists in both input and target
            let shape_dim = tensor_shape[tensor_idx as usize];
            // Expansion is only valid if target dim matches input dim, OR input dim is 1.
            if !(shape_dim == target_dim || shape_dim == 1) {
                return Err(NeuraRustError::ShapeMismatch {
                    operation: "expand".to_string(),
                    expected: format!(
                        "Target dim {} or 1 at index {} (from right)",
                        shape_dim,
                        target_rank - 1 - target_idx
                    ),
                    actual: format!(
                        "Target dim {} at index {} (from right)",
                        target_dim,
                        target_rank - 1 - target_idx
                    ),
                });
            }
            tensor_idx -= 1;
        } else {
            // New dimension added by expand
            // New dimensions cannot be size 0
            if target_dim == 0 {
                 return Err(NeuraRustError::ShapeMismatch {
                    operation: "expand".to_string(),
                    expected: "Target dim > 0 for new dimensions".to_string(),
                    actual: format!(
                        "Target dim 0 at index {} (from right)",
                         target_rank - 1 - target_idx
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Performs the expand operation, creating a view with potentially larger dimensions.
pub fn expand_op(tensor: &Tensor, target_shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    validate_expand_shapes(tensor, &target_shape)?;

    let tensor_data_guard = tensor.read_data();

    let original_shape = &tensor_data_guard.shape;
    let original_strides = &tensor_data_guard.strides;
    let original_rank = original_shape.len();
    let target_rank = target_shape.len();

    let mut new_shape = target_shape.clone();
    let mut new_strides = vec![0; target_rank];

    let mut shape_idx = (original_rank as isize) - 1;
    for i in (0..target_rank).rev() {
        let target_dim = target_shape[i];

        if shape_idx >= 0 {
            let original_dim = original_shape[shape_idx as usize];
            let original_stride = original_strides[shape_idx as usize];

            if target_dim == usize::MAX {
                new_shape[i] = original_dim;
                new_strides[i] = original_stride;
            } else if original_dim == target_dim {
                new_strides[i] = original_stride;
            } else if original_dim == 1 {
                new_strides[i] = 0;
            } else {
                unreachable!("Invalid expand case detected after validation.");
            }
            shape_idx -= 1;
        } else {
            if target_dim == usize::MAX {
                return Err(NeuraRustError::UnsupportedOperation(
                    "Target shape cannot use usize::MAX (sentinel for -1) for new dimensions.".to_string()
                ));
            }
            new_strides[i] = 0;
        }
    }

    let buffer_arc = tensor_data_guard.buffer.clone();
    let device = tensor_data_guard.device;
    let offset = tensor_data_guard.offset;
    let requires_grad = tensor_data_guard.requires_grad;
    let input_node_arc = if requires_grad { Some(Arc::clone(&tensor.data)) } else { None };

    drop(tensor_data_guard);

    let view_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape,
        new_strides
    )?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    if let Some(input_node) = input_node_arc {
        if requires_grad {
            let original_shape_for_backward = tensor.shape();
            let backward_op = ExpandBackward {
                input_node: input_node,
                original_shape: original_shape_for_backward,
            };
            let backward_op_arc: Arc<dyn BackwardOp + Send + Sync> = Arc::new(backward_op);
            output_tensor.write_data().grad_fn = Some(backward_op_arc);
            output_tensor.write_data().requires_grad = true;
        }
    }

    Ok(output_tensor)
}

// --- Tests ---
// Link the external test file
#[cfg(test)]
#[path = "expand_test.rs"]
mod tests;

/* --- REMOVED internal tests module --- 
#[cfg(test)]
mod tests {
    // ... (contenu de l'ancien module de tests) ...
}
*/ 