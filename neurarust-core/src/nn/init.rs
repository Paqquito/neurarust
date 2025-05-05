use crate::ops::traits::NeuraNumeric;
use std::sync::{Arc, RwLockWriteGuard};
use crate::buffer::CpuBuffer;
use crate::tensor::TensorData;
use rand::{Rng, distributions::{Uniform, Distribution, StandardNormal}};

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

/// Fills the input `Tensor` with values according to the Kaiming uniform initialization method.
///
/// Operates in-place. Calculates the bound based on the tensor's fan-in.
///
/// # Arguments
/// * `tensor`: The tensor to initialize (typically a weight tensor).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn kaiming_uniform_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    let fan_in = calculate_fan_in(tensor)?;
    let bound = (6.0 / fan_in as f64).sqrt(); // Calculate bound as f64 for precision
    
    // Create distribution based on tensor dtype
    match tensor.dtype() {
        DType::F32 => {
            let bound_f32 = bound as f32;
            let dist = Uniform::new(-bound_f32, bound_f32);
            fill_inplace_distribution(tensor, &dist)
        }
        DType::F64 => {
            let dist = Uniform::new(-bound, bound); // Use f64 bound
            fill_inplace_distribution(tensor, &dist)
        }
        // Add other dtypes later if needed
    }
}

/// Fills the input `Tensor` with values according to the Kaiming normal initialization method.
///
/// Operates in-place. Calculates the standard deviation based on the tensor's fan-in.
///
/// # Arguments
/// * `tensor`: The tensor to initialize (typically a weight tensor).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn kaiming_normal_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    let fan_in = calculate_fan_in(tensor)?;
    let std = (2.0 / fan_in as f64).sqrt(); // Calculate std as f64 for precision

    // Use StandardNormal distribution and scale the result
    fill_inplace_distribution_scaled(tensor, &StandardNormal, std)
}

/// Fills the input `Tensor` with values according to the Xavier (Glorot) uniform initialization method.
///
/// Operates in-place. Calculates the bound based on the tensor's fan-in and fan-out.
///
/// # Arguments
/// * `tensor`: The tensor to initialize (typically a weight tensor).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn xavier_uniform_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor)?;
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt(); // Calculate bound as f64

    match tensor.dtype() {
        DType::F32 => {
            let bound_f32 = bound as f32;
            let dist = Uniform::new(-bound_f32, bound_f32);
            fill_inplace_distribution(tensor, &dist)
        }
        DType::F64 => {
            let dist = Uniform::new(-bound, bound);
            fill_inplace_distribution(tensor, &dist)
        }
    }
}

// --- Helper for calculating fan_in/fan_out ---
// Note: This is a simplified version assuming standard weight shapes.
// PyTorch has a more complex _calculate_fan_in_and_fan_out function.
fn calculate_fan_in(tensor: &Tensor) -> Result<usize, NeuraRustError> {
    let shape = tensor.shape();
    if shape.len() < 2 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Fan in/out calculation requires at least 2 dimensions".to_string(),
        ));
    }
    // Assuming standard weight shape (e.g., [out_features, in_features, ...])
    Ok(shape[1])
}

// --- Helper for calculating fan_in/fan_out ---
fn calculate_fan_in_and_fan_out(tensor: &Tensor) -> Result<(usize, usize), NeuraRustError> {
    let shape = tensor.shape();
    if shape.len() < 2 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Fan in/out calculation requires at least 2 dimensions".to_string(),
        ));
    }
    // Assuming standard weight shape [out_features, in_features, ...]
    let fan_out = shape[0];
    let fan_in = shape[1];
    Ok((fan_in, fan_out))
}

// --- Helper for filling with a distribution ---

/// Helper function to fill a tensor in-place with values from a distribution.
/// Assumes the Distribution `D` generates values of the correct type `T` matching the tensor buffer.
fn fill_inplace_distribution<D, T>(tensor: &mut Tensor, dist: &D) -> Result<(), NeuraRustError>
where
    D: Distribution<T>,
    T: NeuraNumeric + Copy, // Tensor element type
{
    let mut guard = tensor.write_data();
    if guard.requires_grad && guard.grad_fn.is_none() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "fill_inplace_distribution".to_string(),
            reason: "Cannot fill_inplace on a leaf tensor that requires grad.".to_string(),
        });
    }
    if guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "In-place distribution fill currently only supports CPU tensors.".to_string(),
        ));
    }

    let mut rng = rand::thread_rng();

    // Use the correct buffer based on T (which should match guard.dtype)
    match guard.buffer {
        CpuBuffer::F32(ref mut arc_vec) => {
            // Ensure T is f32, otherwise this is a logic error
             if let Some(vec) = Arc::get_mut(arc_vec) {
                for elem in vec.iter_mut() {
                     *elem = dist.sample(&mut rng); // D samples f32
                }
            } else {
                 return Err(NeuraRustError::InternalError(
                    "Failed to get mutable access to F32 buffer for in-place distribution fill.".to_string(),
                 ));
            }
        }
        CpuBuffer::F64(ref mut arc_vec) => {
             // Ensure T is f64
             if let Some(vec) = Arc::get_mut(arc_vec) {
                for elem in vec.iter_mut() {
                    *elem = dist.sample(&mut rng); // D samples f64
                }
            } else {
                 return Err(NeuraRustError::InternalError(
                    "Failed to get mutable access to F64 buffer for in-place distribution fill.".to_string(),
                 ));
            }
        }
    }
    Ok(())
}

// --- Helper for filling with a scaled distribution ---

/// Helper function to fill a tensor in-place with values from a distribution, scaled by a factor.
/// Assumes the Distribution `D` generates values of type f64 or f32.
fn fill_inplace_distribution_scaled<D, T>(tensor: &mut Tensor, dist: &D, scale: f64) -> Result<(), NeuraRustError>
where
    D: Distribution<T>,
    T: NeuraNumeric + Copy, // Type generated by the distribution (f32 or f64)
{
    let mut guard = tensor.write_data();
    if guard.requires_grad && guard.grad_fn.is_none() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "fill_inplace_distribution_scaled".to_string(),
            reason: "Cannot fill_inplace on a leaf tensor that requires grad.".to_string(),
        });
    }
    if guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "In-place distribution fill currently only supports CPU tensors.".to_string(),
        ));
    }

    let mut rng = rand::thread_rng();

    match guard.buffer {
        CpuBuffer::F32(ref mut arc_vec) => {
            let scale_f32 = scale as f32;
            if let Some(vec) = Arc::get_mut(arc_vec) {
                for elem in vec.iter_mut() {
                    // Assuming D samples f32 directly for StandardNormal
                    let sample: f32 = dist.sample(&mut rng);
                    *elem = sample * scale_f32;
                }
            } else {
                 return Err(NeuraRustError::InternalError(
                    "Failed to get mutable access to F32 buffer for scaled in-place fill.".to_string(),
                 ));
            }
        }
        CpuBuffer::F64(ref mut arc_vec) => {
            if let Some(vec) = Arc::get_mut(arc_vec) {
                for elem in vec.iter_mut() {
                     // Assuming D samples f64 directly for StandardNormal
                    let sample: f64 = dist.sample(&mut rng);
                    *elem = sample * scale; // Use f64 scale
                }
            } else {
                 return Err(NeuraRustError::InternalError(
                    "Failed to get mutable access to F64 buffer for scaled in-place fill.".to_string(),
                 ));
            }
        }
    }
    Ok(())
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