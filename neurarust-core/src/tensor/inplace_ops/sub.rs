use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    // device::StorageDevice, // Not directly used here, check if tensor.device() is needed or handled by Tensor methods
    types::DType,
    tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64},
    tensor::utils::broadcast_shapes,
};
use std::sync::Arc;
use std::ops::{Deref, DerefMut};

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

    if current_tensor.requires_grad() || current_tensor.grad_fn().is_some() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "sub_".to_string(),
            reason: "Tensor requires grad or has grad_fn.".to_string()
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
    
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in sub_".to_string()
    })?;
    let other_tensor_data_guard = other.data.read().map_err(|_| NeuraRustError::LockError{
        lock_type: "read (other)".to_string(),
        reason: "Failed to acquire read lock for other.data in sub_".to_string()
    })?;

    let self_offset_val = self_tensor_data_guard.offset;
    let numel_self = self_shape_vec.iter().product();

    match self_dtype {
        DType::F32 => {
            let self_buffer_mut_ref = Arc::get_mut(&mut self_tensor_data_guard.deref_mut().buffer)
                .ok_or_else(|| NeuraRustError::BufferSharedError { operation: "sub_ (self, F32 - Arc::get_mut on TensorData.buffer failed)".to_string() })?;
            let self_vec_mut = self_buffer_mut_ref.try_get_cpu_f32_mut().map_err(|e_core| NeuraRustError::BufferAccessError { 
                buffer_type: "F32 mut vec (self)".to_string(), 
                details: format!("{:?}", e_core) 
            })?;
            
            let other_buffer_concrete_ref: &crate::buffer::Buffer = &*other_tensor_data_guard.deref().buffer;
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
            let self_buffer_mut_ref = Arc::get_mut(&mut self_tensor_data_guard.deref_mut().buffer)
                .ok_or_else(|| NeuraRustError::BufferSharedError { operation: "sub_ (self, F64 - Arc::get_mut on TensorData.buffer failed)".to_string() })?;
            let self_vec_mut = self_buffer_mut_ref.try_get_cpu_f64_mut().map_err(|e_core| NeuraRustError::BufferAccessError { 
                buffer_type: "F64 mut vec (self)".to_string(), 
                details: format!("{:?}", e_core) 
            })?;

            let other_buffer_concrete_ref: &crate::buffer::Buffer = &*other_tensor_data_guard.deref().buffer;
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
        _ => {
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "In-place subtraction not supported for dtype {:?}",
                self_dtype
            )));
        }
    }
    Ok(())
} 