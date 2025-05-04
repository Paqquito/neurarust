use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::buffer::{Buffer, CpuBuffer}; // Need Buffer types
use crate::types::DType; // Need DType
 // Keep Debug for recursive helper
 // Keep Arc

// Helper function for recursive multidimensional iteration used by contiguous()
// Adapt for non-generic TensorData, assuming F32 for now
fn copy_non_contiguous_recursive(
    original_guard: &TensorData, // Use non-generic TensorData ref
    original_f32_data: &[f32], // Pass the specific slice
    new_buffer: &mut Vec<f32>, // Expect f32 output
    current_indices: &mut Vec<usize>,
    current_dim: usize,
) -> Result<(), NeuraRustError> { // Return Result for potential errors
    if current_dim == original_guard.shape.len() {
        let original_offset = original_guard.get_offset(current_indices);
        if original_offset >= original_f32_data.len() {
            return Err(NeuraRustError::InternalError(format!(
                "Contiguous copy error: Offset {} out of bounds for buffer len {}",
                original_offset, original_f32_data.len()
            )));
        }
        new_buffer.push(original_f32_data[original_offset]); // Clone handled by push
    } else {
        for i in 0..original_guard.shape[current_dim] {
            current_indices[current_dim] = i;
            copy_non_contiguous_recursive(
                original_guard,
                original_f32_data,
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
        // Call the slice_op function
        // Convert slice of SliceArg to Vec<(usize, usize)> if needed by slice_op
        // For now, assume SliceArg is compatible or slice_op handles it.
        // TODO: Adapt SliceArg if necessary.
        // Assuming SliceArg is just (usize, usize) for now.
        let simple_ranges: Vec<(usize, usize)> = ranges.iter().map(|arg| (arg.start, arg.end)).collect();
        // Use the path via the re-export in ops/view/mod.rs
        crate::ops::view::slice_op(self, simple_ranges)
    }

    /// Creates a view of the tensor with two dimensions transposed.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, NeuraRustError> {
        crate::ops::view::transpose_op(self, dim1, dim2)
    }

    /// Creates a view of the tensor with dimensions permuted according to the specified order.
    pub fn permute(&self, dims: &[usize]) -> Result<Self, NeuraRustError> {
        // permute_op expects Vec<usize>
        crate::ops::view::permute_op(self, dims.to_vec())
    }

    /// Creates a view of the tensor with a different shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        crate::ops::view::reshape_op(self, new_shape)
    }

    /// Returns a contiguous version of the tensor.
    pub fn contiguous(&self) -> Result<Self, NeuraRustError> {
        if self.is_contiguous() {
            Ok(self.clone()) // clone() is non-generic now
        } else {
            let guard = self.data.read().map_err(|_| NeuraRustError::LockError{
                lock_type: "read".to_string(),
                reason: "Failed to lock for contiguous()".to_string()
            })?;
            let td_ref = &*guard; // Get reference to TensorData

            let device = td_ref.device;
            let shape = td_ref.shape.clone();
            let numel = td_ref.numel();

            // Prepare output vec based on dtype
            match (td_ref.dtype, device) {
                (DType::F32, StorageDevice::CPU) => {
                    let mut new_buffer_vec = Vec::with_capacity(numel);
                    // Match on buffer to get the correct CpuBuffer variant
                    match &*td_ref.buffer {
                        Buffer::Cpu(CpuBuffer::F32(original_cpu_data_arc)) => {
                            let original_f32_data: &Vec<f32> = original_cpu_data_arc;
                            let mut current_indices = vec![0; shape.len()];
                            copy_non_contiguous_recursive(
                                td_ref, // Pass TensorData ref
                                original_f32_data, // Pass f32 slice
                                &mut new_buffer_vec,
                                &mut current_indices,
                                0,
                            )?;
                        }
                        _ => return Err(NeuraRustError::InternalError("Mismatched buffer type for F32 dtype in contiguous()".to_string()))
                    }
                    drop(guard);
                    // Call non-generic Tensor::new which handles F32
                    Tensor::new(new_buffer_vec, shape)
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
