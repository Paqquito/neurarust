use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    // buffer::{Buffer, CpuBuffer}, // Déjà géré par crate::buffer::
    tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64},
    tensor::utils::broadcast_shapes,
};
use std::sync::Arc;
// use std::ops::AddAssign; // Non utilisé directement ici, mais dans l'helper des scalaires
// use std::fmt::Debug; // Non utilisé directement ici, mais dans l'helper des scalaires
// Deref is implicitly used for other_buffer_read_guard.buffer.try_get_cpu_f32()
// but it's part of the prelude or automatically brought in scope usually.
// If the compiler complains about Deref, we'll add `use std::ops::Deref;`

// Renamed self_tensor to current_tensor to avoid confusion with self keyword in methods
pub fn perform_add_inplace(current_tensor: &mut Tensor, other: &Tensor) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != other.dtype() {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: current_tensor.dtype(),
            actual: other.dtype(),
            operation: "in-place addition (add_)".to_string(),
        });
    }

    // Autograd check: Disallow in-place if it's a non-leaf or a leaf that requires grad.
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "add_".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string(),
        });
    }

    let self_dtype = current_tensor.dtype();
    let self_shape = current_tensor.shape().clone();
    // Accéder à other.offset() via une garde de lecture temporaire si nécessaire, ou s'assurer qu'il est déjà disponible.
    // Pour l'instant, on suppose qu'il est passé correctement ou accessible.
    // La récupération de other_offset doit être faite AVANT la self_write_guard si other peut être current_tensor.
    // Mais pour add, other est un &Tensor différent.
    let other_read_guard_for_meta = other.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read (other for meta)".to_string(),
        reason: "Failed to acquire read lock for other.data (meta) in add_".to_string(),
    })?;
    let other_shape = other_read_guard_for_meta.shape.clone();
    let other_strides = other_read_guard_for_meta.strides.clone();
    let other_offset = other_read_guard_for_meta.offset;
    drop(other_read_guard_for_meta);

    let broadcast_output_shape = broadcast_shapes(&self_shape, &other_shape)?;
    if broadcast_output_shape != self_shape {
        return Err(NeuraRustError::BroadcastError {
            shape1: self_shape.clone(),
            shape2: other_shape.clone(),
        });
    }

    // --- Data Access and Clone-on-Write ---
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError {
        lock_type: "write (self)".to_string(),
        reason: "Failed to acquire write lock for self.data in add_".to_string(),
    })?;

    // Check for shared buffer before attempting CoW via Arc::make_mut
    // If the buffer is shared, in-place operation is disallowed as per test expectations.
    if Arc::strong_count(&self_tensor_data_guard.buffer) > 1 {
        return Err(NeuraRustError::BufferSharedError {
            operation: "add_ (buffer is shared)".to_string(), 
        });
    }

    // Clone strides from self_write_guard BEFORE the mutable borrow of its buffer for CoW
    let self_strides_cloned = self_tensor_data_guard.strides.clone(); 
    let self_offset_val = self_tensor_data_guard.offset;
    // It's better to also get self_write_guard.device here if needed in error messages within the match
    let self_device_for_error = self_tensor_data_guard.device; 

    let other_read_guard = other.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read (other)".to_string(),
        reason: "Failed to acquire read lock for other.data in add_".to_string(),
    })?;

    let numel_self: usize = self_shape.iter().product();

    // Get a mutable reference to the Buffer enum itself, cloning if the Arc<Buffer> is shared.
    let buffer_enum_mut_ref: &mut crate::buffer::Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

    match self_dtype {
        DType::F32 => {
            let self_buffer_vec_mut: &mut Vec<f32> = match buffer_enum_mut_ref {
                crate::buffer::Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    crate::buffer::CpuBuffer::F32(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch { 
                        expected: DType::F32, actual: self_dtype, operation: "add_ inplace (self buffer is not F32)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch { 
                    expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "add_ inplace (self buffer is not CPU)".to_string()
                }),
            };

            let other_buffer_f32 = other_read_guard.buffer.try_get_cpu_f32()?;

            let mut broadcast_iter = NdArrayBroadcastingIter::new(
                other_buffer_f32,
                &other_shape,
                &other_strides,
                other_offset, // other_offset est maintenant correctement récupéré
                &self_shape, 
            )?;

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, &self_shape);
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_cloned[d]; // Utiliser strides clonées
                }
                let other_val = broadcast_iter.next().ok_or_else(|| NeuraRustError::InternalError("Other iterator exhausted prematurely in add_ (F32).".to_string()))?;
                if physical_offset_self < self_buffer_vec_mut.len() {
                    self_buffer_vec_mut[physical_offset_self] += other_val;
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape.clone() });
                }
            }
        }
        DType::F64 => {
            let self_buffer_vec_mut: &mut Vec<f64> = match buffer_enum_mut_ref {
                crate::buffer::Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    crate::buffer::CpuBuffer::F64(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch { 
                        expected: DType::F64, actual: self_dtype, operation: "add_ inplace (self buffer is not F64)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch { 
                    expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "add_ inplace (self buffer is not CPU)".to_string()
                }),
            };

            let other_buffer_f64 = other_read_guard.buffer.try_get_cpu_f64()?;

            let mut broadcast_iter = NdArrayBroadcastingIterF64::new(
                other_buffer_f64,
                &other_shape,
                &other_strides,
                other_offset, // other_offset est maintenant correctement récupéré
                &self_shape, 
            )?;

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, &self_shape);
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_cloned[d]; // Utiliser strides clonées
                }
                let other_val = broadcast_iter.next().ok_or_else(|| NeuraRustError::InternalError("Other iterator exhausted prematurely in add_ (F64).".to_string()))?;
                if physical_offset_self < self_buffer_vec_mut.len() {
                    self_buffer_vec_mut[physical_offset_self] += other_val;
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape.clone() });
                }
            }
        }
        DType::I32 | DType::I64 | DType::Bool => todo!("add_ inplace non supporté pour ce DType"),
    }
    Ok(())
} 