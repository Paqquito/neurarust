use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    buffer::{Buffer, CpuBuffer},
    ops::traits::NeuraNumeric // Pour la conversion de l'exposant
};
use std::sync::Arc;
use std::fmt::Debug;

pub fn perform_pow_inplace<E: NeuraNumeric + Copy + Debug>(
    current_tensor: &mut Tensor, 
    exponent: E
) -> Result<(), NeuraRustError> {
    // Autograd check: Disallow in-place if it's a non-leaf or a leaf that requires grad.
    if current_tensor.grad_fn().is_some() || (current_tensor.grad_fn().is_none() && current_tensor.requires_grad()) {
        return Err(NeuraRustError::InplaceModificationError {
            operation: "pow_".to_string(),
            reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string()
        });
    }

    let self_dtype = current_tensor.dtype();
    let self_shape_vec = current_tensor.shape().clone();
    
    let mut self_tensor_data_guard = current_tensor.data.write().map_err(|_| NeuraRustError::LockError{
        lock_type: "write (self)".to_string(), 
        reason: "Failed to acquire write lock for self.data in pow_".to_string()
    })?;
    let self_strides_cloned = self_tensor_data_guard.strides.clone();
    let self_offset_val = self_tensor_data_guard.offset;
    let self_device_for_error = self_tensor_data_guard.device;

    let numel_self = self_shape_vec.iter().product();

    let buffer_enum_mut_ref: &mut Buffer = Arc::make_mut(&mut self_tensor_data_guard.buffer);

    match self_dtype {
        DType::F32 => {
            let exp_f32 = exponent.to_f32().ok_or_else(|| NeuraRustError::DataTypeMismatch {
                operation: "pow_ (exponent conversion to f32)".to_string(),
                expected: DType::F32,
                actual: self_dtype, // Ou un DType dérivé de E si possible
            })?;

            let self_vec_mut: &mut Vec<f32> = match buffer_enum_mut_ref {
                Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    CpuBuffer::F32(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch { 
                        expected: DType::F32, actual: self_dtype, operation: "pow_ inplace (self buffer is not F32)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch { 
                    expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "pow_ inplace (self buffer is not CPU)".to_string()
                }),
            };

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, &self_shape_vec);
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_cloned[d];
                }
                if physical_offset_self < self_vec_mut.len() {
                    let val = self_vec_mut[physical_offset_self];
                    if val < 0.0 && exp_f32.fract() != 0.0 { // Base négative à une puissance non entière
                        return Err(NeuraRustError::ArithmeticError(
                            format!("pow_: Negative base ({}) to non-integer exponent ({}) results in complex number or NaN.", val, exp_f32)
                        ));
                    }
                    self_vec_mut[physical_offset_self] = val.powf(exp_f32);
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape_vec.clone() });
                }
            }
        }
        DType::F64 => {
            let exp_f64 = exponent.to_f64().ok_or_else(|| NeuraRustError::DataTypeMismatch {
                operation: "pow_ (exponent conversion to f64)".to_string(),
                expected: DType::F64,
                actual: self_dtype,
            })?;

            let self_vec_mut: &mut Vec<f64> = match buffer_enum_mut_ref {
                Buffer::Cpu(ref mut cpu_buf) => match cpu_buf {
                    CpuBuffer::F64(ref mut arc_vec) => Arc::make_mut(arc_vec),
                    _ => return Err(NeuraRustError::DataTypeMismatch { 
                        expected: DType::F64, actual: self_dtype, operation: "pow_ inplace (self buffer is not F64)".to_string()
                    }),
                },
                _ => return Err(NeuraRustError::DeviceMismatch { 
                    expected: crate::device::StorageDevice::CPU, actual: self_device_for_error, operation: "pow_ inplace (self buffer is not CPU)".to_string()
                }),
            };

            for i in 0..numel_self {
                let logical_coords = crate::tensor::utils::index_to_coord(i, &self_shape_vec);
                let mut physical_offset_self = self_offset_val;
                for d in 0..logical_coords.len() {
                    physical_offset_self += logical_coords[d] * self_strides_cloned[d];
                }
                if physical_offset_self < self_vec_mut.len() {
                    let val = self_vec_mut[physical_offset_self];
                     if val < 0.0 && exp_f64.fract() != 0.0 { // Base négative à une puissance non entière
                        return Err(NeuraRustError::ArithmeticError(
                            format!("pow_: Negative base ({}) to non-integer exponent ({}) results in complex number or NaN.", val, exp_f64)
                        ));
                    }
                    self_vec_mut[physical_offset_self] = val.powf(exp_f64);
                } else {
                    return Err(NeuraRustError::IndexOutOfBounds{ index: logical_coords, shape: self_shape_vec.clone() });
                }
            }
        }
        DType::I32 | DType::I64 | DType::Bool => todo!("pow_ inplace non supporté pour ce DType"),
    }
    Ok(())
} 