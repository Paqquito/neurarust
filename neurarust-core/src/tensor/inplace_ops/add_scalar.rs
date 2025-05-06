use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    buffer::{Buffer, CpuBuffer}
};
use std::sync::Arc;
// use num_traits::Float; // Float might not be strictly necessary if we only add, but good for consistency with pow

pub fn perform_add_scalar_inplace_f32(current_tensor: &mut Tensor, scalar: f32) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::F32 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::F32,
            actual: current_tensor.dtype(),
            operation: "in-place add_scalar_f32".to_string()
        });
    }

    if current_tensor.requires_grad() || current_tensor.grad_fn().is_some() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "add_scalar_f32".to_string(),
            reason: "Tensor requires grad or has grad_fn.".to_string()
        });
    }

    let self_shape_vec = current_tensor.shape(); // Read properties before mutable borrow
    let self_strides_vec = current_tensor.strides();
    let self_offset_val;
    let numel_self;

    {
        let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
            lock_type: "write (self)".to_string(), 
            reason: "Failed to acquire write lock for self.data in add_scalar_f32".to_string()
        })?;
        
        self_offset_val = self_tensor_data_guard.offset;
        numel_self = self_shape_vec.iter().product(); // numel depends on shape, which is read-only

        let buffer_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

        let self_vec_mut: &mut Vec<f32> = match buffer_mut_ref {
            Buffer::Cpu(CpuBuffer::F32(arc_vec)) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::UnsupportedOperation(
                "add_scalar_f32 expects CPU F32 buffer".to_string()
            )),
        };
        
        apply_add_scalar_to_vec(self_vec_mut, scalar, self_shape_vec.as_slice(), self_strides_vec.as_slice(), self_offset_val, numel_self)?;
    }
    Ok(())
}

pub fn perform_add_scalar_inplace_f64(current_tensor: &mut Tensor, scalar: f64) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::F64 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::F64,
            actual: current_tensor.dtype(),
            operation: "in-place add_scalar_f64".to_string()
        });
    }

    if current_tensor.requires_grad() || current_tensor.grad_fn().is_some() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "add_scalar_f64".to_string(),
            reason: "Tensor requires grad or has grad_fn.".to_string()
        });
    }
    
    let self_shape_vec = current_tensor.shape();
    let self_strides_vec = current_tensor.strides();
    let self_offset_val;
    let numel_self;

    {
        let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
            lock_type: "write (self)".to_string(), 
            reason: "Failed to acquire write lock for self.data in add_scalar_f64".to_string()
        })?;
        
        self_offset_val = self_tensor_data_guard.offset;
        numel_self = self_shape_vec.iter().product();

        let buffer_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

        let self_vec_mut: &mut Vec<f64> = match buffer_mut_ref {
            Buffer::Cpu(CpuBuffer::F64(arc_vec)) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::UnsupportedOperation(
                "add_scalar_f64 expects CPU F64 buffer".to_string()
            )),
        };

        apply_add_scalar_to_vec(self_vec_mut, scalar, self_shape_vec.as_slice(), self_strides_vec.as_slice(), self_offset_val, numel_self)?;
    }
    Ok(())
}

fn apply_add_scalar_to_vec<T>(
    vec_data: &mut Vec<T>, 
    scalar: T, 
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    numel: usize
) -> Result<(), NeuraRustError>
where
    T: Copy + std::ops::AddAssign + std::fmt::Debug, // AddAssign for +=, Debug for potential errors
{
    for i in 0..numel {
        let logical_coords = crate::tensor::utils::index_to_coord(i, shape);
        let mut physical_offset = offset;
        for d in 0..logical_coords.len() {
            physical_offset += logical_coords[d] * strides[d];
        }

        if physical_offset < vec_data.len() {
            vec_data[physical_offset] += scalar;
        } else {
            // This should ideally not happen if numel, shape, strides, and offset are consistent
            // with the actual buffer length obtained after Arc::make_mut.
            return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: shape.to_vec() });
        }
    }
    Ok(())
} 