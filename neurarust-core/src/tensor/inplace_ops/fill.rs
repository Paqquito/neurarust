// Logic for the in-place fill operation

use crate::{tensor::Tensor, buffer::CpuBuffer, error::NeuraRustError, types::DType, ops::traits::numeric::NeuraNumeric, Buffer};
use std::sync::Arc;
use std::any::TypeId; // Added for TypeId::of

/// Internal implementation for the in-place fill operation.
/// Ensures the tensor buffer is uniquely owned (handling CoW) and modifies it.
pub(crate) fn perform_fill_inplace<S: NeuraNumeric>(
    tensor: &mut Tensor,
    value: S,
) -> Result<(), NeuraRustError> {
    
    if !tensor.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(
            "fill_ requires the tensor to be contiguous.".to_string()
        ));
    }

    // Get mutable access guard to the TensorData
    let mut tensor_data_guard = tensor.data.write().map_err(|e| NeuraRustError::LockError {
        lock_type: "write".to_string(),
        reason: format!("Failed to lock tensor data for fill_: {}", e),
    })?;

    // Get dtype from TensorData before potentially modifying the buffer structure via Arc::make_mut
    let tensor_dtype = tensor_data_guard.dtype; 

    // Get mutable access to the Buffer via the guard and Arc
    // The Arc::make_mut ensures we have a unique mutable reference, handling CoW.
    let buffer_mut = Arc::make_mut(&mut tensor_data_guard.buffer); 

    match buffer_mut {
        Buffer::Cpu(cpu_buffer) => { // Handle CPU case
            match cpu_buffer {
                CpuBuffer::F32(_) => {
                    // Ensure the tensor is actually F32
                    if tensor_dtype != DType::F32 {
                         return Err(NeuraRustError::DataTypeMismatch {
                            operation: "fill_".to_string(),
                            expected: DType::F32,
                            actual: tensor_dtype, 
                        });
                    }
                    // Explicitly check if the type of S is f32
                    if TypeId::of::<S>() != TypeId::of::<f32>() {
                        // If tensor is F32 and S is not f32, then S must be f64 (given NeuraNumeric constraints)
                        return Err(NeuraRustError::DataTypeMismatch {
                            operation: "fill_".to_string(),
                            expected: DType::F32, // Tensor's dtype
                            actual: DType::F64,   // S must be f64 if not f32
                        });
                    }
                    // Try converting the input value S to f32
                    let scalar_val = value.to_f32().ok_or_else(|| {
                        // This case should ideally not be reached if TypeId check is robust
                        // and NeuraNumeric implies convertibility to its own type.
                        NeuraRustError::DataTypeMismatch {
                            operation: "fill_ value conversion".to_string(),
                            expected: DType::F32, 
                            actual: DType::F32, // If TypeId matched, S is f32
                        }
                    })?;
                    let data_slice = buffer_mut.try_get_cpu_f32_mut()?;
                    data_slice.fill(scalar_val);
                }
                CpuBuffer::F64(_) => {
                     // Ensure the tensor is actually F64
                     if tensor_dtype != DType::F64 {
                         return Err(NeuraRustError::DataTypeMismatch {
                            operation: "fill_".to_string(),
                            expected: DType::F64,
                            actual: tensor_dtype,
                        });
                    }
                    // Explicitly check if the type of S is f64
                    if TypeId::of::<S>() != TypeId::of::<f64>() {
                        // If tensor is F64 and S is not f64, then S must be f32 (given NeuraNumeric constraints)
                        return Err(NeuraRustError::DataTypeMismatch {
                            operation: "fill_".to_string(),
                            expected: DType::F64, // Tensor's dtype
                            actual: DType::F32,   // S must be f32 if not f64
                        });
                    }
                    // Try converting the input value S to f64
                    let scalar_val = value.to_f64().ok_or_else(|| {
                        // This case should ideally not be reached if TypeId check is robust
                        NeuraRustError::DataTypeMismatch {
                           operation: "fill_ value conversion".to_string(),
                           expected: DType::F64, 
                           actual: DType::F64, // If TypeId matched, S is f64
                       }
                    })?;
                    let data_slice = buffer_mut.try_get_cpu_f64_mut()?;
                    data_slice.fill(scalar_val);
                }
                CpuBuffer::I32(_) | CpuBuffer::I64(_) | CpuBuffer::Bool(_) => todo!("fill_ non supporté pour ce DType"),
                // Add other CpuBuffer variants here when supported
            }
        }
        Buffer::Gpu { .. } => { // Handle GPU case
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "fill_ is not yet supported for DType {:?} on GPU",
                tensor_dtype
            )));
        }
    }

    Ok(())
} 