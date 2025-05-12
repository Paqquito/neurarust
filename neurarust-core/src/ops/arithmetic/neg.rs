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
        let contiguous_grad_output = grad_output.contiguous()?;
        let input_grad = neg_op(&contiguous_grad_output)?;
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
pub fn neg_op(a: &Tensor) -> Result<Tensor, NeuraRustError> {
    if !a.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "neg_op requires contiguous input tensor. Found strides: {:?} for shape {:?}",
            a.strides(), a.shape()
        )));
    }

    let a_data_guard = a.data.read().map_err(|e| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: format!("Failed to lock tensor data in neg_op: {}", e),
    })?;

    if a_data_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "neg_op".to_string(),
            expected: StorageDevice::CPU,
            actual: a_data_guard.device,
        });
    }
    
    let output_shape = a_data_guard.shape.clone();
    let offset = a_data_guard.offset;
    let numel: usize = output_shape.iter().product();

    let new_data_buffer: Buffer = match a_data_guard.dtype {
        DType::F32 => {
            match &*a_data_guard.buffer {
                Buffer::Cpu(CpuBuffer::F32(data_arc)) => {
                    if offset + numel > data_arc.len() {
                        return Err(NeuraRustError::InternalError(format!(
                            "neg_op F32: offset + numel ({}+{}={}) exceeds buffer len ({}) for shape {:?} and strides {:?}",
                            offset, numel, offset + numel, data_arc.len(), &output_shape, a.strides()
                        )));
                    }
                    let data_slice = &data_arc[offset..offset + numel];
                    Buffer::Cpu(CpuBuffer::F32(neg_kernel::<f32>(data_slice).into()))
                }
                _ => return Err(NeuraRustError::InternalError("Buffer type mismatch for F32 DType in neg_op".to_string())),
            }
        }
        DType::F64 => {
            match &*a_data_guard.buffer {
                Buffer::Cpu(CpuBuffer::F64(data_arc)) => {
                     if offset + numel > data_arc.len() {
                        return Err(NeuraRustError::InternalError(format!(
                            "neg_op F64: offset + numel ({}+{}={}) exceeds buffer len ({}) for shape {:?} and strides {:?}",
                            offset, numel, offset + numel, data_arc.len(), &output_shape, a.strides()
                        )));
                    }
                    let data_slice = &data_arc[offset..offset + numel];
                    Buffer::Cpu(CpuBuffer::F64(neg_kernel::<f64>(data_slice).into()))
                }
                _ => return Err(NeuraRustError::InternalError("Buffer type mismatch for F64 DType in neg_op".to_string())),
            }
        }
        DType::I32 | DType::I64 | DType::Bool => {
            return Err(NeuraRustError::UnsupportedOperation(
                "neg_op n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string()
            ));
        }
    };
    
    let new_strides = if output_shape.is_empty() {
        vec![]
    } else {
        let mut strides = vec![0; output_shape.len()];
        strides[output_shape.len() - 1] = 1;
        for i in (0..output_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * output_shape[i + 1];
        }
        strides
    };

    let new_tensor_data = TensorData {
        buffer: Arc::new(new_data_buffer),
        shape: output_shape.clone(),
        strides: new_strides,
        offset: 0,
        dtype: a_data_guard.dtype,
        device: a_data_guard.device,
        requires_grad: a.requires_grad(),
        grad: None,
        grad_fn: None,
    };

    let output_tensor = Tensor {
        data: Arc::new(RwLock::new(new_tensor_data)),
    };

    if a.requires_grad() {
        let grad_fn = NegBackward { a_node: Some(Arc::clone(&a.data)) };
        output_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
        output_tensor.set_requires_grad(true)?;
    }
    
    drop(a_data_guard);

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
