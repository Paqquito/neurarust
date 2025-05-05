use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::buffer::{Buffer, CpuBuffer}; // Need Buffer types
use crate::types::DType; // Need DType
use std::fmt::Debug;
 // Keep Debug for recursive helper
 // Keep Arc

// Helper function for recursive multidimensional iteration used by contiguous()
// Made generic over numeric type T
fn copy_non_contiguous_recursive<T>(
    original_guard: &TensorData, // Keep non-generic TensorData ref
    original_data_slice: &[T],  // Generic slice
    new_buffer: &mut Vec<T>,    // Generic output buffer
    current_indices: &mut Vec<usize>,
    current_dim: usize,
) -> Result<(), NeuraRustError>
where
    T: Copy + Debug, // Add required traits
{
    if current_dim == original_guard.shape.len() {
        let original_offset = original_guard.get_offset(current_indices);
        if original_offset >= original_data_slice.len() {
            return Err(NeuraRustError::InternalError(format!(
                "Contiguous copy error: Offset {} out of bounds for buffer len {}",
                original_offset, original_data_slice.len()
            )));
        }
        new_buffer.push(original_data_slice[original_offset]); // Works for any T: Copy
    } else {
        for i in 0..original_guard.shape[current_dim] {
            current_indices[current_dim] = i;
            copy_non_contiguous_recursive(
                original_guard,
                original_data_slice,
                new_buffer,
                current_indices,
                current_dim + 1,
            )?;
        }
    }
    Ok(())
}

// Remove <T> and bounds from impl block
impl Tensor {
    /// Creates a view of the tensor by slicing along specified dimensions.
    /// Currently, only basic slicing is supported (no steps or negative indices).
    /// Requires the tensor data to be on the CPU.
    pub fn slice(&self, ranges: &[crate::ops::view::SliceArg]) -> Result<Self, NeuraRustError> {
        // Reactivate the call to the underlying slice_op function
        crate::ops::view::slice_op(self, ranges)
    }

    /// Creates a view of the tensor with two dimensions transposed.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, NeuraRustError> {
        crate::ops::view::transpose_op(self, dim1, dim2)
    }

    /// Creates a view of the tensor with dimensions permuted according to the specified order.
    pub fn permute(&self, dims: &[usize]) -> Result<Self, NeuraRustError> {
        crate::ops::view::permute_op(self, dims)
    }

    /// Creates a view of the tensor with a different shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        crate::ops::view::reshape_op(self, new_shape)
    }

    /// Returns a contiguous version of the tensor.
    pub fn contiguous(&self) -> Result<Self, NeuraRustError> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let guard = self.data.read().map_err(|_| NeuraRustError::LockError{
                lock_type: "read".to_string(),
                reason: "Failed to lock for contiguous()".to_string()
            })?;
            let td_ref = &*guard;

            let device = td_ref.device;
            let shape = td_ref.shape.clone();
            let numel = td_ref.numel();

            // Dispatch based on dtype and device
            match (td_ref.dtype, device) {
                (DType::F32, StorageDevice::CPU) => {
                    let mut new_buffer_vec = Vec::with_capacity(numel);
                    match &*td_ref.buffer {
                        Buffer::Cpu(CpuBuffer::F32(original_cpu_data_arc)) => {
                            let original_f32_data: &[f32] = original_cpu_data_arc;
                            let mut current_indices = vec![0; shape.len()];
                            // Call generic recursive function
                            copy_non_contiguous_recursive(
                                td_ref,
                                original_f32_data,
                                &mut new_buffer_vec,
                                &mut current_indices,
                                0,
                            )?;
                        }
                        _ => return Err(NeuraRustError::InternalError("Mismatched buffer type for F32 dtype in contiguous()".to_string()))
                    }
                    drop(guard);
                    Tensor::new(new_buffer_vec, shape)
                }
                (DType::F64, StorageDevice::CPU) => {
                    let mut new_buffer_vec: Vec<f64> = Vec::with_capacity(numel);
                    match &*td_ref.buffer {
                        Buffer::Cpu(CpuBuffer::F64(original_cpu_data_arc)) => {
                            let original_f64_data: &[f64] = original_cpu_data_arc;
                            let mut current_indices = vec![0; shape.len()];
                            // Call generic recursive function for F64
                            copy_non_contiguous_recursive(
                                td_ref,
                                original_f64_data,
                                &mut new_buffer_vec,
                                &mut current_indices,
                                0,
                            )?;
                        }
                        _ => return Err(NeuraRustError::InternalError("Mismatched buffer type for F64 dtype in contiguous()".to_string()))
                    }
                    drop(guard);
                    // Call F64 constructor
                    Tensor::new_f64(new_buffer_vec, shape)
                }
                 // TODO: Add cases for other DTypes (e.g., I64) later
                 // (DType::I64, StorageDevice::CPU) => { ... }
                (dtype, StorageDevice::GPU) => {
                    Err(NeuraRustError::UnsupportedOperation(
                        format!("GPU contiguous copy not yet implemented for dtype {:?}", dtype)
                    ))
                }
            }
        }
    }
}
