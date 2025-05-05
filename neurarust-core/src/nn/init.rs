use crate::ops::traits::NeuraNumeric;
use std::sync::{Arc, RwLockWriteGuard};
use crate::buffer::CpuBuffer;

// TODO: Implement initialization functions (kaiming_uniform_, etc.)

/// Fills the input `Tensor` with the scalar value 0.
///
/// Operates in-place.
///
/// # Arguments
/// * `tensor`: The tensor to fill (mutable reference).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn zeros_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    fill_inplace(tensor, 0.0)
}

/// Fills the input `Tensor` with the scalar value 1.
///
/// Operates in-place.
///
/// # Arguments
/// * `tensor`: The tensor to fill (mutable reference).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn ones_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    fill_inplace(tensor, 1.0)
}

// --- Internal Helper for In-place Fill ---

/// Helper function to fill a tensor in-place with a scalar value.
/// Handles different data types.
fn fill_inplace<T>(tensor: &mut Tensor, value: T) -> Result<(), NeuraRustError>
where
    T: NeuraNumeric + Copy, // Value must be numeric and copyable
{
    let mut guard = tensor.write_data(); // Get write lock

    // Check for autograd safety: Cannot modify leaf tensor requiring grad in-place
    if guard.requires_grad && guard.grad_fn.is_none() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "fill_inplace".to_string(),
            reason: "Cannot fill_inplace on a leaf tensor that requires grad.".to_string(),
        });
    }

    // TODO: Later, handle non-CPU devices.
    if guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "In-place fill currently only supports CPU tensors.".to_string(),
        ));
    }

    // Match on DType and modify the buffer
    match guard.buffer {
        CpuBuffer::F32(ref mut arc_vec) => {
            let val_f32 = value.to_f32().ok_or_else(|| NeuraRustError::DataTypeMismatch {
                operation: "fill_inplace".to_string(),
                expected: DType::F32,
                actual: guard.dtype, // Should match T but good practice
            })?;
            if let Some(vec) = Arc::get_mut(arc_vec) {
                vec.fill(val_f32);
            } else {
                // Handle case where Arc has multiple owners (should not happen if called correctly?)
                // We might need to clone/reallocate if we can't get mutable access.
                // For now, return an error or potentially clone.
                 return Err(NeuraRustError::InternalError(
                    "Failed to get mutable access to F32 buffer for in-place fill.".to_string(),
                 ));
            }
        }
        CpuBuffer::F64(ref mut arc_vec) => {
            let val_f64 = value.to_f64().ok_or_else(|| NeuraRustError::DataTypeMismatch {
                operation: "fill_inplace".to_string(),
                expected: DType::F64,
                actual: guard.dtype,
            })?;
             if let Some(vec) = Arc::get_mut(arc_vec) {
                vec.fill(val_f64);
            } else {
                 return Err(NeuraRustError::InternalError(
                    "Failed to get mutable access to F64 buffer for in-place fill.".to_string(),
                 ));
            }
        }
        // Add other dtypes later
    }

    Ok(())
}

// --- Tests ---
#[cfg(test)]
#[path = "init_test.rs"]
mod tests; // Link to the test file 