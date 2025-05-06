use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    buffer::{Buffer, CpuBuffer}
};
use std::sync::Arc;
use num_traits::Float;

pub fn perform_pow_inplace_f32(current_tensor: &mut Tensor, exponent: f32) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::F32 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::F32,
            actual: current_tensor.dtype(),
            operation: "in-place power (pow_) with f32 exponent".to_string()
        });
    }

    if current_tensor.requires_grad() || current_tensor.grad_fn().is_some() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "pow_".to_string(),
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
            reason: "Failed to acquire write lock for self.data in pow_ (f32)".to_string()
        })?;
        
        self_offset_val = self_tensor_data_guard.offset;
        numel_self = self_shape_vec.iter().product();

        let buffer_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

        let self_vec_mut: &mut Vec<f32> = match buffer_mut_ref {
            Buffer::Cpu(CpuBuffer::F32(arc_vec)) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::UnsupportedOperation(
                "pow_ (f32) expects CPU F32 buffer".to_string()
            )),
        };
        
        apply_pow_to_vec(self_vec_mut, exponent, self_shape_vec.as_slice(), self_strides_vec.as_slice(), self_offset_val, numel_self)?;
    }
    Ok(())
}

pub fn perform_pow_inplace_f64(current_tensor: &mut Tensor, exponent: f64) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::F64 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::F64,
            actual: current_tensor.dtype(),
            operation: "in-place power (pow_) with f64 exponent".to_string()
        });
    }

    if current_tensor.requires_grad() || current_tensor.grad_fn().is_some() {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "pow_".to_string(),
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
            reason: "Failed to acquire write lock for self.data in pow_ (f64)".to_string()
        })?;
        
        self_offset_val = self_tensor_data_guard.offset;
        numel_self = self_shape_vec.iter().product();

        let buffer_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

        let self_vec_mut: &mut Vec<f64> = match buffer_mut_ref {
            Buffer::Cpu(CpuBuffer::F64(arc_vec)) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::UnsupportedOperation(
                "pow_ (f64) expects CPU F64 buffer".to_string()
            )),
        };

        apply_pow_to_vec(self_vec_mut, exponent, self_shape_vec.as_slice(), self_strides_vec.as_slice(), self_offset_val, numel_self)?;
    }
    Ok(())
}


fn apply_pow_to_vec<T>(
    vec_data: &mut Vec<T>, 
    exponent: T, 
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    numel: usize
) -> Result<(), NeuraRustError>
where
    T: Float + Copy + std::fmt::Debug, // Debug for error messages
{
    for i in 0..numel {
        let logical_coords = crate::tensor::utils::index_to_coord(i, shape);
        let mut physical_offset = offset;
        for d in 0..logical_coords.len() {
            physical_offset += logical_coords[d] * strides[d];
        }

        if physical_offset < vec_data.len() {
            let base_val = vec_data[physical_offset];
            
            if base_val.is_zero() && exponent.is_zero() {
                vec_data[physical_offset] = T::one(); // 0^0 = 1
            } else if base_val < T::zero() && exponent.fract() != T::zero() {
                return Err(NeuraRustError::ArithmeticError(
                    format!("pow_ error: negative base ({:?}) with non-integer exponent ({:?}).", base_val, exponent)
                ));
            } else {
                vec_data[physical_offset] = base_val.powf(exponent);
            }
        } else {
            return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: shape.to_vec() });
        }
    }
    Ok(())
} 