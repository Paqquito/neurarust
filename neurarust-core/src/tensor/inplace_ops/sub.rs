use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    // device::StorageDevice, // Not directly used here, check if tensor.device() is needed or handled by Tensor methods
    types::DType,
    tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64},
    tensor::utils::broadcast_shapes,
    // buffer::{Buffer, CpuBuffer} // Importé via crate::buffer
};
use std::sync::Arc;
// use std::ops::{Deref, DerefMut}; // Supprimés

// Renamed self_tensor to current_tensor to avoid confusion with self keyword in methods
pub fn perform_sub_inplace(current_tensor: &mut Tensor, other: &Tensor) -> Result<(), NeuraRustError> {
    // Initial checks: dtype and autograd requirements
    if current_tensor.dtype() != other.dtype() {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: current_tensor.dtype(), 
            actual: other.dtype(), 
            operation: "in-place subtraction (sub_)".to_string()
        });
    }

    // Autograd check: Disallow in-place if it's a non-leaf or a leaf that requires grad.
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "sub_".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string()
        });
    }

    // Gather all necessary shape, strides, and offset information using read access
    let self_dtype = current_tensor.dtype();
    let self_shape_vec = current_tensor.shape(); 
    let self_strides_vec = current_tensor.strides(); 
    
    let other_shape_vec = other.shape(); 
    let other_strides_vec = other.strides();
    
    let broadcast_output_shape = broadcast_shapes(self_shape_vec.as_slice(), other_shape_vec.as_slice())?;
    if broadcast_output_shape.as_slice() != self_shape_vec.as_slice() {
        return Err(NeuraRustError::BroadcastError {
            shape1: self_shape_vec.clone(), 
            shape2: other_shape_vec.clone(),
        });
    }
    
    // --- Data Access and Clone-on-Write --- 
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in sub_".to_string()
    })?;

    if Arc::strong_count(&self_tensor_data_guard.buffer) > 1 {
        return Err(NeuraRustError::BufferSharedError {
            operation: "sub_ (buffer is shared)".to_string(),
        });
    }

    let other_tensor_data_guard = other.data.read().map_err(|_| NeuraRustError::LockError{
        lock_type: "read (other)".to_string(),
        reason: "Failed to acquire read lock for other.data in sub_".to_string()
    })?;

    let self_offset_val = self_tensor_data_guard.offset;
    let numel_self = self_shape_vec.iter().product();

    // Get a mutable reference to the Buffer enum itself, cloning if the Arc<Buffer> is shared.
    let buffer_enum_mut_ref: &mut crate::buffer::Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

    match self_dtype {
        DType::F32 => {
            // Now, get a mutable reference to the Vec<f32> from the &mut Buffer enum,
            // cloning the Arc<Vec<f32>> if it's shared.
            let self_vec_mut: &mut Vec<f32> = match buffer_enum_mut_ref {
                crate::buffer::Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    crate::buffer::CpuBuffer::F32(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32, actual: self_dtype, operation: "sub_ inplace (self buffer is not F32)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch {
                    expected: crate::device::StorageDevice::CPU, actual: self_tensor_data_guard.device, operation: "sub_ inplace (self buffer is not CPU)".to_string()
                }),
            };
            
            let other_buffer_concrete_ref: &crate::buffer::Buffer = &*other_tensor_data_guard.buffer; // This is Arc<Buffer>
            let other_buffer_arc_f32 = other_buffer_concrete_ref.try_get_cpu_f32().map_err(|e_core| NeuraRustError::BufferAccessError { 
                buffer_type: "F32 Arc<Vec> (other)".to_string(), 
                details: format!("{:?}", e_core) 
            })?;
            
            let mut other_iter = NdArrayBroadcastingIter::new(
                other_buffer_arc_f32,         
                other_shape_vec.as_slice(),   
                other_strides_vec.as_slice(), 
                other_tensor_data_guard.offset, 
                self_shape_vec.as_slice()       
            ).map_err(|e| NeuraRustError::InternalError(format!("Failed to create other_iter for sub_ (F32): {:?}", e)))?;

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, self_shape_vec.as_slice());
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_vec[d];
                }

                let other_val = other_iter.next().ok_or_else(|| NeuraRustError::InternalError("Other iterator exhausted prematurely in sub_ (F32).".to_string()))?;

                if physical_offset_self < self_vec_mut.len() {
                    self_vec_mut[physical_offset_self] -= other_val;
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
                        expected: DType::F64, actual: self_dtype, operation: "sub_ inplace (self buffer is not F64)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch {
                    expected: crate::device::StorageDevice::CPU, actual: self_tensor_data_guard.device, operation: "sub_ inplace (self buffer is not CPU)".to_string()
                }),
            };
            
            let other_buffer_concrete_ref: &crate::buffer::Buffer = &*other_tensor_data_guard.buffer; // This is Arc<Buffer>
            let other_buffer_arc_f64 = other_buffer_concrete_ref.try_get_cpu_f64().map_err(|e_core| NeuraRustError::BufferAccessError { 
                buffer_type: "F64 Arc<Vec> (other)".to_string(), 
                details: format!("{:?}", e_core) 
            })?;
            
            let mut other_iter = NdArrayBroadcastingIterF64::new(
                other_buffer_arc_f64,         
                other_shape_vec.as_slice(),   
                other_strides_vec.as_slice(), 
                other_tensor_data_guard.offset, 
                self_shape_vec.as_slice()       
            ).map_err(|e| NeuraRustError::InternalError(format!("Failed to create other_iter for sub_ (F64): {:?}", e)))?;

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, self_shape_vec.as_slice());
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_vec[d];
                }

                let other_val = other_iter.next().ok_or_else(|| NeuraRustError::InternalError("Other iterator exhausted prematurely in sub_ (F64).".to_string()))?;

                if physical_offset_self < self_vec_mut.len() {
                    self_vec_mut[physical_offset_self] -= other_val;
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape_vec.clone() });
                }
            }
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    }
    Ok(())
} 