use crate::autograd::backward_op::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use std::fmt::Debug;
// Add Add trait needed for potential acc_grad, Send/Sync for BackwardOp
use std::sync::{Arc, RwLock};
use crate::autograd::graph::NodeId;
// Import necessary items based on API analysis
use crate::ops::traits::NeuraNumeric;
use crate::types::DType;
use crate::device::StorageDevice;
use crate::buffer::{Buffer, CpuBuffer};

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise negation operation.
///
/// Stores a reference to the input tensor node for graph linkage.
#[derive(Debug)]
struct NegBackward {
    /// Reference counted pointer to the input tensor's data for graph linkage.
    a_node: Option<Arc<RwLock<TensorData>>>,
}

// --- Backward Operation Implementation ---

impl BackwardOp for NegBackward {
    /// Computes the gradient for the negation operation \( z = -a \).
    ///
    /// Using the chain rule \( \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{dz}{da} \),
    /// where \( \frac{dz}{da} = -1 \), the gradient is:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \cdot (-1) = - \frac{dL}{dz} \\]
    ///
    /// This method simply negates the incoming gradient (`grad_output`).
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let input_grad = neg_op(grad_output)?; // Reuse neg_op
        Ok(vec![input_grad])
    }

    /// Returns the identifier of the input tensor node.
    fn inputs(&self) -> Vec<NodeId> {
        match &self.a_node {
            Some(node) => vec![Arc::as_ptr(node) as NodeId],
            None => vec![], // Should not happen if requires_grad was true
        }
    }
}

// --- Forward Operation ---

/// Generic kernel for element-wise negation.
fn neg_kernel<T: NeuraNumeric>(input_data: &[T]) -> Vec<T> {
    input_data.iter().map(|&x| -x).collect()
}

/// Computes the element-wise negation of the input tensor.
///
/// Computes the negative of each element in the input tensor.
/// Supports `DType::F32` and `DType::F64` tensors on the CPU.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `input`: The input `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` with the negated values, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - The tensor is not on the CPU (`DeviceMismatch`).
/// - The tensor's `DType` is not F32 or F64 (`UnsupportedOperation`).
/// - An internal error occurs.
pub fn neg_op(input: &Tensor) -> Result<Tensor, NeuraRustError> {
    let input_guard = input.read_data();

    // --- Get autograd context --- 
    let requires_grad = input_guard.requires_grad;
    // Clone the Arc needed for the backward pass *before* dropping the guard
    let input_data_arc_opt = if requires_grad { 
        Some(Arc::clone(&input.data)) // Clone the Arc from the input Tensor
    } else { 
        None 
    };

    // --- Forward Computation --- 
    // Check device
    if input_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "neg_op".to_string(),
            expected: StorageDevice::CPU,
            actual: input_guard.device, 
        });
    }

    // Check contiguity (required for simple buffer access via offset/numel)
    if !input_guard.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "neg_op (refactored) currently requires contiguous input tensor. Found strides: {:?}", 
            input_guard.strides
        )));
    }

    // Prepare shape and buffer details from guard
    let output_shape = input_guard.shape.clone();
    let offset = input_guard.offset;
    let numel = input_guard.numel(); // Safe because we checked for contiguity
    let buffer_arc = Arc::clone(input_guard.buffer()); // Clone Arc to buffer

    // Match DType and compute
    let output_tensor = match input_guard.dtype {
        DType::F32 => {
            match &*buffer_arc {
                Buffer::Cpu(CpuBuffer::F32(data_arc)) => {
                    let data_slice = &data_arc[offset .. offset + numel];
                    let output_data = neg_kernel::<f32>(data_slice);
                    Tensor::new(output_data, output_shape)?
                }
                _ => return Err(NeuraRustError::InternalError("Buffer type mismatch for F32 DType in neg_op".to_string())),
            }
        }
        DType::F64 => {
             match &*buffer_arc {
                Buffer::Cpu(CpuBuffer::F64(data_arc)) => {
                    let data_slice = &data_arc[offset .. offset + numel];
                    let output_data = neg_kernel::<f64>(data_slice);
                    Tensor::new_f64(output_data, output_shape)?
                }
                _ => return Err(NeuraRustError::InternalError("Buffer type mismatch for F64 DType in neg_op".to_string())),
            }
        }
    };
    
    // Drop the read guard explicitly *after* extracting all needed info
    drop(input_guard); 

    // --- Autograd Handling ---
    if requires_grad {
        let grad_fn = NegBackward { a_node: input_data_arc_opt }; // Pass the cloned Arc
        // Call set_grad_fn with only the grad_fn Arc
        output_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
        // Explicitly set requires_grad on the output tensor
        output_tensor.set_requires_grad(true)?; 
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
