use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    device::StorageDevice,
    types::DType,
    tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64},
    tensor::utils::{broadcast_shapes, index_to_coord},
};
use std::sync::Arc;
// use std::ops::DerefMut; // DerefMut is not directly used by Arc::get_mut path
// Deref is implicitly used for other_buffer_read_guard.buffer.try_get_cpu_f32()
// but it's part of the prelude or automatically brought in scope usually.
// If the compiler complains about Deref, we'll add `use std::ops::Deref;`

// Renamed self_tensor to current_tensor to avoid confusion with self keyword in methods
pub fn perform_add_inplace(current_tensor: &mut Tensor, other: &Tensor) -> Result<(), NeuraRustError> {
    // Step 1: Autograd Check
    let self_read_guard = current_tensor.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock self.data for requires_grad check in add_".to_string(),
    })?;
    if self_read_guard.requires_grad {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "add_".to_string(),
            reason: "Cannot perform in-place operation on a tensor that requires gradients.".to_string(),
        });
    }

    // Step 2: Device and DType Checks, and gather initial info
    let other_read_guard = other.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock other.data for checks in add_".to_string(),
    })?;

    if self_read_guard.device != other_read_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "add_".to_string(),
            expected: self_read_guard.device,
            actual: other_read_guard.device,
        });
    }
    if self_read_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedDevice {
            device: self_read_guard.device,
            operation: "add_".to_string(),
        });
    }

    if self_read_guard.dtype != other_read_guard.dtype {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "add_".to_string(),
            expected: self_read_guard.dtype,
            actual: other_read_guard.dtype,
        });
    }
    let dtype = self_read_guard.dtype;

    let self_shape = self_read_guard.shape.clone();
    let self_strides = self_read_guard.strides.clone();
    let self_offset = self_read_guard.offset;
    let self_is_contiguous = self_read_guard.is_contiguous();

    let other_shape = other_read_guard.shape.clone();
    let other_strides = other_read_guard.strides.clone();
    let other_offset = other_read_guard.offset;
    
    drop(self_read_guard);
    drop(other_read_guard);
    
    // Broadcasting check (moved before write lock on current_tensor)
    // This uses cloned shapes, so no tensor data access is needed here.
    let broadcast_target_shape = broadcast_shapes(&self_shape, &other_shape)?;
    if broadcast_target_shape.as_slice() != self_shape.as_slice() {
        // This specific check ensures that `other` can be broadcast to `self` without `self` changing shape.
        // The generic `broadcast_shapes` might return a larger shape if both can be broadcast to it.
        // For in-place, `self` must dictate the final shape.
        return Err(NeuraRustError::BroadcastError {
            shape1: self_shape.clone(), // Or format!("{:?}", self_shape)
            shape2: other_shape.clone(), // Or format!("{:?}", other_shape)
        });
    }

    let mut self_write_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError {
        lock_type: "write".to_string(),
        reason: "Failed to lock self.data for in-place modification in add_".to_string(),
    })?;
    
    let other_buffer_read_guard = other.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to re-lock other.data for buffer access in add_".to_string(),
    })?;

    match dtype {
        DType::F32 => {
            let buffer_ref_mut = Arc::get_mut(&mut self_write_guard.buffer).ok_or_else(|| 
                NeuraRustError::BufferSharedError { 
                    operation: "add_ (TensorData.buffer is shared)".to_string() 
                }
            )?;
            let self_buffer_vec_mut = buffer_ref_mut.try_get_cpu_f32_mut()?;
            let other_buffer_f32 = other_buffer_read_guard.buffer.try_get_cpu_f32()?;

            let mut broadcast_iter = NdArrayBroadcastingIter::new(
                other_buffer_f32,
                &other_shape,
                &other_strides,
                other_offset,
                &self_shape, // Target shape for iteration is self's shape
            )?;

            let self_numel: usize = self_shape.iter().product();
            if self_is_contiguous {
                let start = self_offset; // self_offset was from the original self_read_guard
                let end = self_offset + self_numel;
                if end > self_buffer_vec_mut.len() { 
                    return Err(NeuraRustError::InternalError("Contiguous slice out of bounds in add_".to_string()));
                }
                let self_slice = &mut self_buffer_vec_mut[start..end];
                for (self_val, other_val) in self_slice.iter_mut().zip(broadcast_iter) {
                    *self_val += other_val;
                }
            } else {
                for i in 0..self_numel {
                    let coord = index_to_coord(i, &self_shape);
                    let mut self_idx = self_offset; // self_offset was from the original self_read_guard
                    for (d, s) in coord.iter().zip(self_strides.iter()) {
                        self_idx += d * s;
                    }
                    if self_idx >= self_buffer_vec_mut.len() { 
                        return Err(NeuraRustError::InternalError(format!("Index {} out of bounds for self_buffer in add_ (non-contiguous)", self_idx)));
                    }
                    match broadcast_iter.next() {
                        Some(other_val) => {
                            self_buffer_vec_mut[self_idx] += other_val;
                        }
                        None => {
                            return Err(NeuraRustError::InternalError("Broadcast iterator exhausted prematurely in add_ (non-contiguous).".to_string()));
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let buffer_ref_mut = Arc::get_mut(&mut self_write_guard.buffer).ok_or_else(|| 
                NeuraRustError::BufferSharedError { 
                    operation: "add_ (TensorData.buffer is shared for F64)".to_string() 
                }
            )?;
            let self_buffer_vec_mut = buffer_ref_mut.try_get_cpu_f64_mut()?;
            let other_buffer_f64 = other_buffer_read_guard.buffer.try_get_cpu_f64()?;

            let mut broadcast_iter = NdArrayBroadcastingIterF64::new(
                other_buffer_f64,
                &other_shape,
                &other_strides,
                other_offset,
                &self_shape, // Target shape for iteration is self's shape
            )?;

            let self_numel: usize = self_shape.iter().product();
            if self_is_contiguous {
                let start = self_offset; // self_offset was from the original self_read_guard
                let end = self_offset + self_numel;
                 if end > self_buffer_vec_mut.len() { 
                    return Err(NeuraRustError::InternalError("Contiguous slice out of bounds in add_ (f64)".to_string()));
                }
                let self_slice = &mut self_buffer_vec_mut[start..end];
                for (self_val, other_val) in self_slice.iter_mut().zip(broadcast_iter) {
                    *self_val += other_val;
                }
            } else {
                for i in 0..self_numel {
                    let coord = index_to_coord(i, &self_shape);
                    let mut self_idx = self_offset; // self_offset was from the original self_read_guard
                    for (d, s) in coord.iter().zip(self_strides.iter()) {
                        self_idx += d * s;
                    }
                    if self_idx >= self_buffer_vec_mut.len() { 
                         return Err(NeuraRustError::InternalError(format!("Index {} out of bounds for self_buffer in add_ (f64, non-contiguous)", self_idx)));
                    }
                    match broadcast_iter.next() {
                        Some(other_val) => {
                            self_buffer_vec_mut[self_idx] += other_val;
                        }
                        None => {
                            return Err(NeuraRustError::InternalError("Broadcast iterator exhausted prematurely in add_ (f64, non-contiguous).".to_string()));
                        }
                    }
                }
            }
        }
    }

    Ok(())
} 