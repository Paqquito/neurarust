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

/// Validates shapes for the expand operation, assuming -1 has been processed.
fn validate_expand_shapes_processed(
    tensor_shape: &[usize],
    processed_target_shape: &[usize],
) -> Result<(), NeuraRustError> {
    let tensor_rank = tensor_shape.len();
    let target_rank = processed_target_shape.len();

    if tensor_rank > target_rank {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "expand".to_string(),
            expected: format!("Input rank ({}) <= Target rank ({})", tensor_rank, target_rank),
            actual: "Input rank > Target rank".to_string(),
        });
    }

    let mut tensor_idx = (tensor_rank as isize) - 1;
    for target_idx in (0..target_rank).rev() {
        let target_dim = processed_target_shape[target_idx];

        if tensor_idx >= 0 {
            // Dimension exists in both input and target
            let shape_dim = tensor_shape[tensor_idx as usize];
            // Expansion is only valid if target dim matches input dim, OR input dim is 1.
            // Since -1 is processed, target_dim should match shape_dim if shape_dim != 1.
            if !(shape_dim == target_dim || shape_dim == 1) {
                return Err(NeuraRustError::ShapeMismatch {
                    operation: "expand".to_string(),
                    expected: format!(
                        "Target dim {} or 1 at index {} (from right, original tensor dim {})",
                        shape_dim,
                        target_rank - 1 - target_idx,
                        shape_dim
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
            // New dimension added by expand (e.g. tensor [2], target [3,2])
            // target_dim here is a new dimension being created.
            // It cannot be 0 if it's a new dimension, already handled by target_shape processing.
            // No specific check needed here as -1 should have been resolved or errored.
        }
    }

    Ok(())
}

/// Performs the expand operation, creating a view with potentially larger dimensions.
pub fn expand_op(tensor: &Tensor, target_shape: &[isize]) -> Result<Tensor, NeuraRustError> {
    let t_guard = tensor.read_data();
    let source_shape = &t_guard.shape;
    let source_rank = source_shape.len();
    let target_rank = target_shape.len();

    if target_rank < source_rank {
        return Err(NeuraRustError::UnsupportedOperation(
            "Target rank cannot be less than source rank for expand.".to_string(),
        ));
    }

    // --- 1. Validate and Finalize Target Shape ---
    let mut final_target_shape = Vec::with_capacity(target_rank);
    let rank_diff = target_rank - source_rank;

    for j in 0..target_rank {
        let val = target_shape[j];
        let is_new_dim = j < rank_diff;
        let source_dim_idx_opt = if is_new_dim { None } else { Some(j - rank_diff) };

        if val == -1 {
            if is_new_dim {
                return Err(NeuraRustError::UnsupportedOperation(
                    "Dimension size -1 not allowed for new dimensions in expand.".to_string(),
                ));
            }
            // Existing dimension, infer from source
            final_target_shape.push(source_shape[source_dim_idx_opt.unwrap()]);
        } else if val == 0 {
            if is_new_dim {
                // New dimension of size 0 is allowed
                final_target_shape.push(0);
            } else {
                // Existing dimension
                let src_dim_size = source_shape[source_dim_idx_opt.unwrap()];
                if src_dim_size == 0 {
                    final_target_shape.push(0); // Copying a 0-sized dimension
                } else {
                    return Err(NeuraRustError::UnsupportedOperation(format!(
                        "Target dimension {} is 0, but corresponding source dimension {} is not 0 (it's {}). Only a 0-sized source dim can be targeted with 0.",
                        j, source_dim_idx_opt.unwrap(), src_dim_size
                    )));
                }
            }
        } else if val > 0 {
            let current_target_dim_size = val as usize;
            if is_new_dim {
                // New dimension with a specified positive size
                final_target_shape.push(current_target_dim_size);
            } else {
                // Existing dimension
                let src_dim_size = source_shape[source_dim_idx_opt.unwrap()];
                if current_target_dim_size < src_dim_size {
                    return Err(NeuraRustError::UnsupportedOperation(format!(
                        "Target dimension {} size {} is smaller than source dimension {} size {}.",
                        j, current_target_dim_size, source_dim_idx_opt.unwrap(), src_dim_size
                    )));
                }
                if current_target_dim_size > src_dim_size && src_dim_size != 1 {
                    return Err(NeuraRustError::UnsupportedOperation(format!(
                        "Cannot expand source dimension {} (size {}) to target size {} because source is not 1.",
                        source_dim_idx_opt.unwrap(), src_dim_size, current_target_dim_size
                    )));
                }
                final_target_shape.push(current_target_dim_size);
            }
        } else { // val < -1
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "Invalid target dimension size: {}. Must be -1 (infer), 0 (copy if original is 0 or new 0-dim), or > 0.",
                val
            )));
        }
    }

    // If shapes are identical after finalization, just return a view (clone of the Arc)
    if *source_shape == final_target_shape {
        // No need to drop t_guard yet, tensor.clone() is cheap.
        return Ok(tensor.clone());
    }

    validate_expand_shapes_processed(source_shape, &final_target_shape)?;

    let original_strides = &t_guard.strides;
    let current_original_rank = source_shape.len(); 
    let current_target_rank = final_target_shape.len();

    let mut new_strides = vec![0; current_target_rank];

    let mut shape_idx = (current_original_rank as isize) - 1;
    for i in (0..current_target_rank).rev() { 
        let target_dim_val = final_target_shape[i];

        if shape_idx >= 0 { 
            let original_dim_val = source_shape[shape_idx as usize];
            let original_stride_val = original_strides[shape_idx as usize];

            if original_dim_val == target_dim_val {
                new_strides[i] = original_stride_val;
            } else if original_dim_val == 1 && target_dim_val > 1 {
                new_strides[i] = 0;
            } else if original_dim_val == 1 && target_dim_val == 1 {
                new_strides[i] = original_stride_val;
            } else {
                unreachable!(
                    "Invalid expand case detected after validation. Original: {}, Target: {}",
                    original_dim_val, target_dim_val
                );
            }
            shape_idx -= 1;
        } else {
            new_strides[i] = 0;
        }
    }

    let buffer_arc = t_guard.buffer.clone();
    let device = t_guard.device;
    let offset = t_guard.offset;
    let requires_grad = t_guard.requires_grad;
    let input_node_arc = if requires_grad { Some(Arc::clone(&tensor.data)) } else { None };

    let view_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        final_target_shape,
        new_strides
    )?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    if let Some(input_node) = input_node_arc {
        if requires_grad {
            let backward_op = ExpandBackward {
                input_node: input_node,
                original_shape: source_shape.to_vec(), 
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