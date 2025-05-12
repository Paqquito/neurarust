use crate::{
    error::NeuraRustError,
    ops::traits::numeric::NeuraNumeric,
    DType,
    buffer::{Buffer, CpuBuffer},
    tensor_data::TensorData,
};
use std::{fmt::Debug, sync::Arc, any::TypeId};

/// Clamps a single value between an optional minimum and maximum.
fn clamp_value<T: NeuraNumeric + PartialOrd>(val: T, min: Option<T>, max: Option<T>) -> T {
    let mut clamped_val = val;
    if let Some(min_val) = min {
        if clamped_val < min_val {
            clamped_val = min_val;
        }
    }
    if let Some(max_val) = max {
        if clamped_val > max_val {
            clamped_val = max_val;
        }
    }
    clamped_val
}

fn process_clamp_for_typed_buffer<T>(
    buffer_data: &mut [T],
    min_opt: Option<T>,
    max_opt: Option<T>,
) where T: NeuraNumeric + PartialOrd + Debug + Copy {
    for val_mut_ref in buffer_data.iter_mut() {
        *val_mut_ref = clamp_value(*val_mut_ref, min_opt, max_opt);
    }
}

/// Implementation for the in-place clamp operation.
/// THIS MUST BE pub(crate) for visibility from tensor module.
///
/// This function modifies the tensor's buffer directly after performing safety checks.
///
/// # Arguments
///
/// * `tensor_data_guard`: A mutable reference to the `TensorData`.
/// * `min_opt`: An optional minimum value of type `T`.
/// * `max_opt`: An optional maximum value of type `T`.
///
/// # Errors
///
/// Returns `NeuraRustError` if the in-place modification is unsafe (e.g., on a leaf tensor
/// requiring gradients or a non-leaf tensor part of a graph), or if the buffer
/// cannot be accessed as the specified type `T`.
pub(crate) fn clamp_tensor_data<T>(
    tensor_data_guard: &mut TensorData,
    min_opt: Option<T>,
    max_opt: Option<T>,
) -> Result<(), NeuraRustError>
where
    T: NeuraNumeric + PartialOrd + Debug + Copy + Send + Sync + 'static,
{
    let current_tensor_dtype = tensor_data_guard.dtype;
    let t_as_dtype: DType;

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        t_as_dtype = DType::F32;
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        t_as_dtype = DType::F64;
    } else {
        return Err(NeuraRustError::InternalError(format!(
            "clamp_tensor_data called with unexpected generic type T: {}", std::any::type_name::<T>()
        )));
    }

    if current_tensor_dtype != t_as_dtype {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "clamp_ (internal type vs tensor dtype)".to_string(),
            expected: current_tensor_dtype,
            actual: t_as_dtype,
        });
    }

    let buffer_enum_mut = Arc::make_mut(&mut tensor_data_guard.buffer);

    match buffer_enum_mut {
        Buffer::Cpu(cpu_buffer) => {
            let (untyped_vec_slice_all_elements, vec_len): (&mut [u8], usize) = match (cpu_buffer, t_as_dtype) {
                (CpuBuffer::F32(data_arc_f32), DType::F32) => {
                    let data_vec_f32 = Arc::make_mut(data_arc_f32);
                    let len = data_vec_f32.len();
                    let ptr = data_vec_f32.as_mut_ptr() as *mut u8;
                    (unsafe { std::slice::from_raw_parts_mut(ptr, len * std::mem::size_of::<f32>()) }, len)
                }
                (CpuBuffer::F64(data_arc_f64), DType::F64) => {
                    let data_vec_f64 = Arc::make_mut(data_arc_f64);
                    let len = data_vec_f64.len();
                    let ptr = data_vec_f64.as_mut_ptr() as *mut u8;
                    (unsafe { std::slice::from_raw_parts_mut(ptr, len * std::mem::size_of::<f64>()) }, len)
                }
                _ => { 
                    return Err(NeuraRustError::InternalError(format!(
                        "Mismatched CpuBuffer variant and DType T ({:?}) in clamp_tensor_data after initial checks. Tensor DType: {:?}.",
                        t_as_dtype, // DType of T
                        current_tensor_dtype // Actual DType of the tensor's buffer
                    )));
                }
            };

            let typed_vec_slice_all_elements = unsafe {
                std::slice::from_raw_parts_mut(untyped_vec_slice_all_elements.as_mut_ptr() as *mut T, vec_len)
            };

            if tensor_data_guard.is_contiguous() {
                let offset = tensor_data_guard.offset;
                let numel = tensor_data_guard.numel();
                if offset + numel > typed_vec_slice_all_elements.len() {
                    return Err(NeuraRustError::InternalError(format!(
                        "Contiguous slice [{}..{}] out of bounds for buffer len {} in clamp_tensor_data",
                        offset, offset + numel, typed_vec_slice_all_elements.len()
                    )));
                }
                let logical_slice = &mut typed_vec_slice_all_elements[offset .. offset + numel];
                process_clamp_for_typed_buffer(logical_slice, min_opt, max_opt);
            } else {
                return Err(NeuraRustError::UnsupportedOperation(
                    "clamp_ on non-contiguous tensors is not yet implemented. Tensor must be contiguous.".to_string()
                ));
            }
        }
        Buffer::Gpu { .. } => {
            return Err(NeuraRustError::UnsupportedDevice {
                device: tensor_data_guard.device,
                operation: "clamp_".to_string(),
            });
        }
        &mut Buffer::Cuda(_) => todo!(),
    }
    Ok(())
} 