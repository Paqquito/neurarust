use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    buffer::{Buffer, CpuBuffer}
};
use std::sync::Arc;
use num_traits::Zero; // For checking division by zero
use std::ops::DivAssign;
use std::fmt::Debug;

fn apply_div_scalar_to_vec<T>(
    vec_data: &mut Vec<T>, 
    scalar: T, 
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    numel: usize
) -> Result<(), NeuraRustError>
where
    T: Copy + DivAssign + Zero + PartialEq + Debug,
{
    if scalar.is_zero() {
        return Err(NeuraRustError::ArithmeticError("Division by zero.".to_string()));
    }
    for i in 0..numel {
        let logical_coords = crate::tensor::utils::index_to_coord(i, shape);
        let mut physical_offset = offset;
        for d in 0..logical_coords.len() {
            physical_offset += logical_coords[d] * strides[d];
        }

        if physical_offset < vec_data.len() {
            vec_data[physical_offset] /= scalar;
        } else {
            return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: shape.to_vec() });
        }
    }
    Ok(())
}

pub fn perform_div_scalar_inplace_f32(current_tensor: &mut Tensor, scalar: f32) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::F32 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::F32,
            actual: current_tensor.dtype(),
            operation: "in-place div_scalar_f32".to_string()
        });
    }
    if scalar == 0.0f32 { // Specific check for f32 before helper
        return Err(NeuraRustError::ArithmeticError("Division by zero in div_scalar_f32.".to_string()));
    }

    // Autograd check: Disallow in-place if it's a non-leaf or a leaf that requires grad.
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "div_scalar_f32".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string()
        });
    }

    let self_shape_vec = current_tensor.shape().clone();
    
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in div_scalar_f32".to_string()
    })?;
    
    let self_strides_cloned = self_tensor_data_guard.strides.clone();
    let self_offset_val = self_tensor_data_guard.offset;
    let self_device_for_error = self_tensor_data_guard.device;
    let numel_self = self_shape_vec.iter().product();

    let buffer_enum_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

    let self_vec_mut: &mut Vec<f32> = match buffer_enum_mut_ref {
        Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
            CpuBuffer::F32(ref mut arc_vec) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::DataTypeMismatch { 
                expected: DType::F32, actual: current_tensor.dtype(), operation: "div_scalar_f32 inplace (self buffer is not F32)".to_string()
            }),
        },
        _ => return Err(NeuraRustError::DeviceMismatch { 
            expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "div_scalar_f32 inplace (self buffer is not CPU)".to_string()
        }),
    };
    // Division by zero already checked for f32, helper's generic check is fine but redundant here.
    apply_div_scalar_to_vec(self_vec_mut, scalar, self_shape_vec.as_slice(), &self_strides_cloned, self_offset_val, numel_self)?;
    Ok(())
}

pub fn perform_div_scalar_inplace_f64(current_tensor: &mut Tensor, scalar: f64) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::F64 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::F64,
            actual: current_tensor.dtype(),
            operation: "in-place div_scalar_f64".to_string()
        });
    }
    if scalar == 0.0f64 { // Specific check for f64 before helper
        return Err(NeuraRustError::ArithmeticError("Division by zero in div_scalar_f64.".to_string()));
    }

    // Autograd check: Disallow in-place if it's a non-leaf or a leaf that requires grad.
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "div_scalar_f64".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string()
        });
    }
    
    let self_shape_vec = current_tensor.shape().clone();

    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in div_scalar_f64".to_string()
    })?;
        
    let self_strides_cloned = self_tensor_data_guard.strides.clone();
    let self_offset_val = self_tensor_data_guard.offset;
    let self_device_for_error = self_tensor_data_guard.device;
    let numel_self = self_shape_vec.iter().product();

    let buffer_enum_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

    let self_vec_mut: &mut Vec<f64> = match buffer_enum_mut_ref {
        Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
            CpuBuffer::F64(ref mut arc_vec) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::DataTypeMismatch { 
                expected: DType::F64, actual: current_tensor.dtype(), operation: "div_scalar_f64 inplace (self buffer is not F64)".to_string()
            }),
        },
        _ => return Err(NeuraRustError::DeviceMismatch { 
            expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "div_scalar_f64 inplace (self buffer is not CPU)".to_string()
        }),
    };

    apply_div_scalar_to_vec(self_vec_mut, scalar, self_shape_vec.as_slice(), &self_strides_cloned, self_offset_val, numel_self)?;
    Ok(())
}

pub fn perform_div_scalar_inplace_i32(current_tensor: &mut Tensor, scalar: i32) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::I32 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::I32,
            actual: current_tensor.dtype(),
            operation: "in-place div_scalar_i32".to_string()
        });
    }
    if scalar == 0 {
        return Err(NeuraRustError::ArithmeticError("Division by zero in div_scalar_i32.".to_string()));
    }
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "div_scalar_i32".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string()
        });
    }
    let self_shape_vec = current_tensor.shape().clone();
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in div_scalar_i32".to_string()
    })?;
    let self_strides_cloned = self_tensor_data_guard.strides.clone();
    let self_offset_val = self_tensor_data_guard.offset;
    let self_device_for_error = self_tensor_data_guard.device;
    let numel_self = self_shape_vec.iter().product();
    let buffer_enum_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);
    let self_vec_mut: &mut Vec<i32> = match buffer_enum_mut_ref {
        Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
            CpuBuffer::I32(ref mut arc_vec) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::DataTypeMismatch { 
                expected: DType::I32, actual: current_tensor.dtype(), operation: "div_scalar_i32 inplace (self buffer is not I32)".to_string()
            }),
        },
        _ => return Err(NeuraRustError::DeviceMismatch { 
            expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "div_scalar_i32 inplace (self buffer is not CPU)".to_string()
        }),
    };
    apply_div_scalar_to_vec(self_vec_mut, scalar, self_shape_vec.as_slice(), &self_strides_cloned, self_offset_val, numel_self)?;
    Ok(())
}

pub fn perform_div_scalar_inplace_i64(current_tensor: &mut Tensor, scalar: i64) -> Result<(), NeuraRustError> {
    if current_tensor.dtype() != DType::I64 {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::I64,
            actual: current_tensor.dtype(),
            operation: "in-place div_scalar_i64".to_string()
        });
    }
    if scalar == 0 {
        return Err(NeuraRustError::ArithmeticError("Division by zero in div_scalar_i64.".to_string()));
    }
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "div_scalar_i64".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string()
        });
    }
    let self_shape_vec = current_tensor.shape().clone();
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in div_scalar_i64".to_string()
    })?;
    let self_strides_cloned = self_tensor_data_guard.strides.clone();
    let self_offset_val = self_tensor_data_guard.offset;
    let self_device_for_error = self_tensor_data_guard.device;
    let numel_self = self_shape_vec.iter().product();
    let buffer_enum_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);
    let self_vec_mut: &mut Vec<i64> = match buffer_enum_mut_ref {
        Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
            CpuBuffer::I64(ref mut arc_vec) => Arc::make_mut(arc_vec),
            _ => return Err(NeuraRustError::DataTypeMismatch { 
                expected: DType::I64, actual: current_tensor.dtype(), operation: "div_scalar_i64 inplace (self buffer is not I64)".to_string()
            }),
        },
        _ => return Err(NeuraRustError::DeviceMismatch { 
            expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "div_scalar_i64 inplace (self buffer is not CPU)".to_string()
        }),
    };
    apply_div_scalar_to_vec(self_vec_mut, scalar, self_shape_vec.as_slice(), &self_strides_cloned, self_offset_val, numel_self)?;
    Ok(())
} 