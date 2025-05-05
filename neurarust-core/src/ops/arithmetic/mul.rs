use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use std::sync::Arc;
use crate::autograd::graph::NodeId;

use std::fmt::Debug;
use std::sync::RwLock;
// Import the iterators from their new location
use crate::tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64};

// --- Iterator code removed --- 

// +++ Copied from add.rs - TODO: Move to a shared utils module +++
/// Reduces a gradient tensor to match a target shape, summing along broadcasted dimensions.
fn reduce_gradient_to_shape(
    grad: &Tensor,
    target_shape: &[usize],
) -> Result<Tensor, NeuraRustError> {
    let grad_shape = grad.shape();

    // No reduction needed if shapes already match
    if grad_shape == target_shape {
        return Ok(grad.clone()); // No reduction needed
    }
    
    // Get DType for later tensor creation
    let grad_dtype = grad.dtype();

    // Handle scalar target shape (sum all elements)
    if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
         // sum_op handles DType
         return crate::ops::reduction::sum::sum_op(grad, None, false); // Sum all
    }

    let grad_rank = grad_shape.len();
    let target_rank = target_shape.len();

    if target_rank > grad_rank {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "reduce_gradient_to_shape".to_string(),
            expected: format!("rank <= {}", grad_rank),
            actual: format!("rank {}", target_rank),
        });
    }

    // Identify axes to sum over
    let mut axes_to_sum = Vec::new();
    let rank_diff = grad_rank - target_rank;

    // Sum over dimensions that were added during broadcasting
    for i in 0..rank_diff {
        axes_to_sum.push(i);
    }

    // Sum over dimensions that were broadcasted from 1
    for i in 0..target_rank {
        if target_shape[i] == 1 && grad_shape[i + rank_diff] > 1 {
            axes_to_sum.push(i + rank_diff);
        }
        // Sanity check
        if target_shape[i] > grad_shape[i + rank_diff] {
             return Err(NeuraRustError::ShapeMismatch {
                 operation: "reduce_gradient_to_shape (dimension check)".to_string(),
                 expected: format!("dim {} size <= {}", i, grad_shape[i + rank_diff]),
                 actual: format!("dim {} size {}", i, target_shape[i]),
             });
        }
    }

    if axes_to_sum.is_empty() {
        // Check if reshape is needed due to rank difference (e.g., [1, 2] -> [2])
        if grad_rank != target_rank {
            // reshape_op should be dtype agnostic
            return crate::ops::view::reshape_op(grad, target_shape.to_vec());
        } else {
            // Shapes must be compatible if no axes identified and ranks match
            return Ok(grad.clone());
        }
    }

    // Perform summation using the adapted sum_op (handles DType)
    let summed_grad = crate::ops::reduction::sum::sum_op(grad, Some(&axes_to_sum), false)?;

    // Reshape if necessary to match target shape
    let final_grad = if summed_grad.shape() != target_shape {
        // reshape_op should be dtype agnostic
        crate::ops::view::reshape_op(&summed_grad, target_shape.to_vec())?
    } else {
        summed_grad
    };

    // Final check: ensure the output dtype matches the input gradient dtype
    if final_grad.dtype() != grad_dtype {
        return Err(NeuraRustError::InternalError(format!(
            "reduce_gradient_to_shape: DType mismatch after reduction/reshape. Expected {:?}, got {:?}",
            grad_dtype, final_grad.dtype()
        )));
    }

    Ok(final_grad)
}
// +++ End of copied code +++

// --- Backward Operation Structure ---
#[derive(Debug)]
struct MulBackward {
    a: Tensor,
    b: Tensor,
    // Store Option<Arc> for graph linkage
    a_node: Option<Arc<RwLock<TensorData>>>,
    b_node: Option<Arc<RwLock<TensorData>>>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    a_requires_grad: bool,
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut result_grads = Vec::new();

        if self.a_requires_grad {
            let unreduced_grad_a = mul_op(grad_output, &self.b)?;
            let grad_a = reduce_gradient_to_shape(&unreduced_grad_a, &self.a_shape)?;
            result_grads.push(grad_a);
        }

        if self.b_requires_grad {
            let unreduced_grad_b = mul_op(grad_output, &self.a)?;
            let grad_b = reduce_gradient_to_shape(&unreduced_grad_b, &self.b_shape)?;
            result_grads.push(grad_b);
        }

        Ok(result_grads)
    }

    fn inputs(&self) -> Vec<NodeId> {
        let mut ids = Vec::new();
        if let Some(node) = &self.a_node {
            ids.push(Arc::as_ptr(node));
        }
        if let Some(node) = &self.b_node {
            ids.push(Arc::as_ptr(node));
        }
        ids
    }
}

// --- Forward Operation ---
pub fn mul_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    // Lock data for reading
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device and DType Checks ---
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "mul_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }
    if a_guard.dtype != b_guard.dtype {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "mul_op".to_string(),
            expected: a_guard.dtype,
            actual: b_guard.dtype,
        });
    }

    // --- Broadcasting --- 
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    // --- Prepare for Autograd --- 
    let requires_grad = a_guard.requires_grad || b_guard.requires_grad;
    let a_node_arc = if a_guard.requires_grad { Some(Arc::clone(&a.data)) } else { None };
    let b_node_arc = if b_guard.requires_grad { Some(Arc::clone(&b.data)) } else { None };
    let a_shape_clone = a_guard.shape.clone();
    let b_shape_clone = b_guard.shape.clone();
    let a_req_grad_clone = a_guard.requires_grad;
    let b_req_grad_clone = b_guard.requires_grad;

    // --- DType Dispatch for Computation and Output Tensor Creation ---
    let output_tensor = match a_guard.dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            
            let iter_a = NdArrayBroadcastingIter::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIter::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            
            let output_data_vec: Vec<f32> = iter_a.zip(iter_b).map(|(val_a, val_b)| val_a * val_b).collect();
            
            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!("mul_op F32: Output vec len {} mismatch with expected numel {}", output_data_vec.len(), numel)));
            }
            
            drop(a_guard);
            drop(b_guard);
            Tensor::new(output_data_vec, output_shape)?
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;

            let iter_a = NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;

            let output_data_vec: Vec<f64> = iter_a.zip(iter_b).map(|(val_a, val_b)| val_a * val_b).collect();

            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!("mul_op F64: Output vec len {} mismatch with expected numel {}", output_data_vec.len(), numel)));
            }

            drop(a_guard);
            drop(b_guard);
            Tensor::new_f64(output_data_vec, output_shape)?
        }
    };

    // --- Autograd Setup --- 
    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
            lock_type: "write".to_string(), // Add missing fields
            reason: "Failed to lock output TensorData for write (autograd setup in mul_op)".to_string(),
        })?;
        output_data_write_guard.requires_grad = true;
        let backward_op = MulBackward {
            a: a_clone, 
            b: b_clone,
            a_node: a_node_arc,
            b_node: b_node_arc,
            a_shape: a_shape_clone, 
            b_shape: b_shape_clone,
            a_requires_grad: a_req_grad_clone,
            b_requires_grad: b_req_grad_clone,
        };
        output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
    }

    Ok(output_tensor)
}

// --- Tests --- 
// Link the external test file
#[cfg(test)]
#[path = "mul_test.rs"]
mod tests;

/* --- REMOVED previous comments about removed internal module --- 
// --- REMOVED internal tests module --- 
// Tests for this module can be found in src/ops/arithmetic/mul_test.rs

#[cfg(test)]
mod tests {
    // ... (contenu de l'ancien module de tests) ...
}
*/
