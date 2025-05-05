use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use std::fmt::Debug;
// Add Add trait needed for potential acc_grad, Send/Sync for BackwardOp
use std::sync::{Arc, RwLock};
use crate::buffer::{Buffer, CpuBuffer};
use crate::types::DType;

// --- Backward Operation Structure ---

/// Backward operation for the negation function.
#[derive(Debug)]
struct NegBackward {
    // Store the input tensor ID for the graph traversal
    input_node: Arc<RwLock<TensorData>>,
}

// --- Backward Operation Implementation ---

impl BackwardOp for NegBackward {
    /// Computes gradient for the negation operation z = -a.
    /// grad(a) = grad_output * (-1) = -neg_output
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // grad_a = -grad_output
        // Use the adapted neg_op which preserves DType
        let grad_a = neg_op(grad_output)?;
        Ok(vec![grad_a])
    }

    // inputs() method is no longer needed with the simplified Variable approach for now.
    // If needed later for graph construction, it would return pointers/IDs of input TensorData.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        // Return the pointer to the stored input node
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- Forward Operation ---

/// Performs element-wise negation on a tensor.
/// Handles F32 and F64 tensors on CPU.
pub fn neg_op(input: &Tensor) -> Result<Tensor, NeuraRustError> {
    let input_data_guard = input.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock input TensorData for read in neg_op".to_string(),
    })?;

    // --- Device Check ---
    if input_data_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "neg_op".to_string(),
            expected: StorageDevice::CPU,
            actual: input_data_guard.device,
        });
    }

    let input_dtype = input_data_guard.dtype;
    let output_shape = input_data_guard.shape.clone();
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };

    // --- DType Dispatch for Computation -> Create Output Tensor ---
    let output_tensor = match input_dtype {
        DType::F32 => {
            let output_data_vec: Vec<f32> = match &*input_data_guard.buffer {
                Buffer::Cpu(CpuBuffer::F32(data_arc)) => {
                    data_arc.iter().map(|&x| -x).collect()
                }
                _ => return Err(NeuraRustError::InternalError(
                    "Buffer type mismatch for F32 dtype in neg_op".to_string()
                ))
            };
            drop(input_data_guard); // Drop guard before creating tensor
            Tensor::new(output_data_vec, output_shape)?
        }
        DType::F64 => {
            let output_data_vec: Vec<f64> = match &*input_data_guard.buffer {
                Buffer::Cpu(CpuBuffer::F64(data_arc)) => {
                    data_arc.iter().map(|&x| -x).collect()
                }
                _ => return Err(NeuraRustError::InternalError(
                    "Buffer type mismatch for F64 dtype in neg_op".to_string()
                ))
            };
            drop(input_data_guard); // Drop guard before creating tensor
            Tensor::new_f64(output_data_vec, output_shape)?
        }
        // Add other types later if needed
    };

    // --- Autograd Setup --- (Keep as is, using output_tensor created above)
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                 lock_type: "write".to_string(),
                 reason: "Failed to lock output TensorData for write (autograd setup in neg_op)".to_string(),
             })?;
            output_data_write_guard.requires_grad = true;
            let backward_op = NegBackward { input_node: node_arc };
            output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in neg_op".to_string()));
        }
    }

    Ok(output_tensor)
}

// --- std::ops::Neg implementation ---
// Implement the Neg trait for Tensor by calling neg_op
/* // Remove the generic implementation for now
impl<T> Neg for &Tensor<T>
where
    // Bounds must match neg_op requirements
    T: Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    type Output = Result<Tensor<T>, NeuraRustError>;

    fn neg(self) -> Self::Output {
        neg_op(self)
    }
}
*/

// --- Tests ---
// Link the external test file
#[cfg(test)]
#[path = "neg_test.rs"]
mod tests;
