// neurarust-core/src/ops/math_elem/ln.rs

use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
// Add imports for backward
use crate::ops::arithmetic::{div_op, mul_op};

// --- LnBackward Definition ---

/// Backward operation context for `ln_op`.
#[derive(Debug)]
struct LnBackward {
    input_node: Arc<RwLock<TensorData>>, // Store original tensor info for gradient calculation (1/x)
}

// --- BackwardOp Implementation for LnBackward ---

impl BackwardOp for LnBackward {
    /// Computes gradient for the natural logarithm operation: grad(input) = grad_output * (1 / input)
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // 1. Get input tensor from self.input_node
        //    We need the actual tensor values, not just the node ID.
        //    Let's create a Tensor instance from the Arc.
        let input_tensor = Tensor { data: self.input_node.clone() };
        
        // 2. Calculate 1 / input using div_op 
        //    Create a tensor of 1s with the same shape/device/dtype as input.
        //    Use ones_like (if available and working) or manually create.
        //    Assuming F32 CPU for now.
        let ones = crate::tensor::ones_like(&input_tensor)?;
        let one_over_input = div_op(&ones, &input_tensor)?;

        // 3. Multiply grad_output by (1 / input) using mul_op
        let grad_input = mul_op(grad_output, &one_over_input)?;
        
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- ln_op Implementation (Public API + Autograd Setup) ---

/// Computes the element-wise natural logarithm (ln) of a tensor.
/// Input values must be positive.
/// Currently supports F32 CPU tensors only.
pub fn ln_op(tensor: &Tensor) -> Result<Tensor, NeuraRustError> {
    let requires_grad = tensor.requires_grad();
    let input_node_arc = if requires_grad { Some(tensor.data.clone()) } else { None };

    let input_guard = tensor.read_data();

    // --- Device and DType Check ---
    if input_guard.device != StorageDevice::CPU || input_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "ln_op is currently only supported on F32 CPU tensors, not {:?}/{:?}",
            input_guard.device, input_guard.dtype
        )));
    }

    // --- Get CPU Data Buffer & Kernel Call --- 
    let input_buffer_arc = input_guard.buffer().try_get_cpu_f32()?.clone();
    let input_data_slice = input_buffer_arc.as_slice();
    let shape = input_guard.shape.clone(); // Clone shape before dropping guard
    let _device = input_guard.device; // Keep device info if needed later
    
    // Kernel logic inline for now
    let mut result_data = Vec::with_capacity(input_data_slice.len());
    for &val in input_data_slice {
        if val <= 0.0 {
            // ln is undefined for non-positive values. Return NaN or Error?
            // Returning NaN for now, consistent with some frameworks.
             // Consider returning error: NeuraRustError::DomainError? 
            result_data.push(f32::NAN);
            // Alternatively:
            // return Err(NeuraRustError::DomainError(format!(
            //     "Cannot compute ln of non-positive value: {}", val
            // )));
        } else {
            result_data.push(val.ln());
        }
    }
    
    // Drop the read guard
    drop(input_guard);

    // --- Create Result Tensor --- 
    let result_tensor = Tensor::new(result_data, shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let Some(node_arc) = input_node_arc {
            let grad_fn = LnBackward {
                input_node: node_arc,
            };
            let mut result_guard = result_tensor.write_data();
            result_guard.requires_grad = true;
            result_guard.grad_fn = Some(Arc::new(grad_fn));
        } else {
            // This case should technically not happen if requires_grad is true
            return Err(NeuraRustError::InternalError(
                "ln_op requires grad but input Arc Node unavailable".to_string(),
            ));
        }
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
#[path = "ln_test.rs"]
mod tests; // Link to the test file 