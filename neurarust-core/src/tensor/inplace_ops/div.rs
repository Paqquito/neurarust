use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64},
    tensor::utils::broadcast_shapes,
};
use std::sync::Arc;

pub fn perform_div_inplace(current_tensor: &mut Tensor, other: &Tensor) -> Result<(), NeuraRustError> {
    // Initial checks: dtype and autograd requirements
    if current_tensor.dtype() != other.dtype() {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: current_tensor.dtype(), 
            actual: other.dtype(), 
            operation: "in-place division (div_)".to_string()
        });
    }

    if current_tensor.grad_fn().is_some() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "div_".to_string(),
            reason: "Tensor is not a leaf node (it has a grad_fn). In-place operations are only allowed on leaf tensors.".to_string()
        });
    }

    let self_dtype = current_tensor.dtype();
    let self_shape_vec = current_tensor.shape().clone(); 
    
    let other_read_guard_for_meta = other.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read (other for meta)".to_string(),
        reason: "Failed to acquire read lock for other.data (meta) in div_".to_string(),
    })?;
    let other_shape_vec = other_read_guard_for_meta.shape.clone();
    let other_strides_vec = other_read_guard_for_meta.strides.clone();
    let other_offset_val = other_read_guard_for_meta.offset;
    drop(other_read_guard_for_meta);
    
    let broadcast_output_shape = broadcast_shapes(self_shape_vec.as_slice(), &other_shape_vec)?;
    if broadcast_output_shape.as_slice() != self_shape_vec.as_slice() {
        return Err(NeuraRustError::BroadcastError {
            shape1: self_shape_vec.clone(), 
            shape2: other_shape_vec.clone(),
        });
    }
    
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in div_".to_string()
    })?;
    let self_strides_cloned = self_tensor_data_guard.strides.clone();
    let self_offset_val = self_tensor_data_guard.offset;
    let self_device_for_error = self_tensor_data_guard.device;
    
    let other_tensor_data_guard = other.data.read().map_err(|_| NeuraRustError::LockError{
        lock_type: "read (other)".to_string(),
        reason: "Failed to acquire read lock for other.data in div_".to_string()
    })?;
        
    let numel_self = self_shape_vec.iter().product();

    let buffer_enum_mut_ref: &mut crate::buffer::Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

    match self_dtype {
        DType::F32 => {
            let self_vec_mut: &mut Vec<f32> = match buffer_enum_mut_ref {
                crate::buffer::Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    crate::buffer::CpuBuffer::F32(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch { 
                        expected: DType::F32, actual: self_dtype, operation: "div_ inplace (self buffer is not F32)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch { 
                    expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "div_ inplace (self buffer is not CPU)".to_string()
                }),
            };

            let other_buffer_arc_f32 = other_tensor_data_guard.buffer.try_get_cpu_f32().map_err(|e_core| NeuraRustError::BufferAccessError { 
                buffer_type: "F32 Arc<Vec> (other)".to_string(), 
                details: format!("{:?}", e_core) 
            })?;
            
            let mut other_iter = NdArrayBroadcastingIter::new(
                other_buffer_arc_f32,         
                &other_shape_vec,   
                &other_strides_vec, 
                other_offset_val, 
                self_shape_vec.as_slice()       
            ).map_err(|e| NeuraRustError::InternalError(format!("Failed to create other_iter for div_ (F32): {:?}", e)))?;

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, self_shape_vec.as_slice());
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_cloned[d];
                }
                let other_val = other_iter.next().ok_or_else(|| NeuraRustError::InternalError("Other iterator exhausted prematurely in div_ (F32).".to_string()))?;
                if other_val == 0.0f32 {
                    return Err(NeuraRustError::ArithmeticError("Division by zero in div_ (F32).".to_string()));
                }
                if physical_offset_self < self_vec_mut.len() {
                    self_vec_mut[physical_offset_self] /= other_val;
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape_vec.clone() });
                }
            }
        }
        DType::F64 => {
            let self_vec_mut: &mut Vec<f64> = match buffer_enum_mut_ref {
                crate::buffer::Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    crate::buffer::CpuBuffer::F64(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch { 
                        expected: DType::F64, actual: self_dtype, operation: "div_ inplace (self buffer is not F64)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch { 
                    expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "div_ inplace (self buffer is not CPU)".to_string()
                }),
            };
            
            let other_buffer_arc_f64 = other_tensor_data_guard.buffer.try_get_cpu_f64().map_err(|e_core| NeuraRustError::BufferAccessError { 
                buffer_type: "F64 Arc<Vec> (other)".to_string(), 
                details: format!("{:?}", e_core) 
            })?;
            
            let mut other_iter = NdArrayBroadcastingIterF64::new(
                other_buffer_arc_f64,         
                &other_shape_vec,   
                &other_strides_vec, 
                other_offset_val, 
                self_shape_vec.as_slice()       
            ).map_err(|e| NeuraRustError::InternalError(format!("Failed to create other_iter for div_ (F64): {:?}", e)))?;

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, self_shape_vec.as_slice());
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_cloned[d];
                }
                let other_val = other_iter.next().ok_or_else(|| NeuraRustError::InternalError("Other iterator exhausted prematurely in div_ (F64).".to_string()))?;
                 if other_val == 0.0f64 {
                    return Err(NeuraRustError::ArithmeticError("Division by zero in div_ (F64).".to_string()));
                }
                if physical_offset_self < self_vec_mut.len() {
                    self_vec_mut[physical_offset_self] /= other_val;
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape_vec.clone() });
                }
            }
        }
    }
    Ok(())
} 