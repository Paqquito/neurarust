use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::ops::traits::NeuraNumeric;
use crate::tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64};
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

/// Kernel for element-wise greater than or equal comparison.
/// Returns 1.0f32 if a >= b, 0.0f32 otherwise.
#[inline]
fn ge_kernel<T>(a: T, b: T) -> f32
where
    T: NeuraNumeric + PartialOrd, // PartialOrd is sufficient for >=
{
    if a >= b {
        1.0f32
    } else {
        0.0f32
    }
}

/// Performs element-wise greater than or equal comparison (`a >= b`) between two tensors.
///
/// Compares elements of `a` and `b` after broadcasting to a common shape.
///
/// The result is a tensor with the broadcasted shape and `DType::F32`, containing
/// `1.0` where `a >= b` is true and `0.0` otherwise.
///
/// This operation **does not** support automatic differentiation.
///
/// # Arguments
/// * `a`: The first input `Tensor`.
/// * `b`: The second input `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` (DType F32) with the comparison result (1.0 or 0.0),
/// or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - Tensors are not on the CPU (`DeviceMismatch`).
/// - Tensors have different `DType`s (`DataTypeMismatch`).
/// - Tensors have incompatible shapes for broadcasting (`BroadcastError`).
/// - An internal error occurs.
pub fn ge_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check (CPU only for now) ---
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "ge_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU {
                a_guard.device
            } else {
                b_guard.device
            },
        });
    }

    // --- DType Check (Currently F32/F64) ---
    // TODO: Support Integer/Boolean DTypes later if needed.
    if a_guard.dtype != b_guard.dtype
        || (a_guard.dtype != DType::F32 && a_guard.dtype != DType::F64)
    {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "ge_op currently only supports matching F32 or F64 inputs, got {:?} and {:?}",
            a_guard.dtype,
            b_guard.dtype
        )));
    }
    let dtype = a_guard.dtype;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    // --- DType Dispatch for Computation using Broadcasting Iterators ---
    let result_data_vec: Vec<f32> = match dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            let iter_a = NdArrayBroadcastingIter::new(
                a_buffer,
                &a_guard.shape,
                &a_guard.strides,
                a_guard.offset,
                &output_shape,
            )?;
            let iter_b = NdArrayBroadcastingIter::new(
                b_buffer,
                &b_guard.shape,
                &b_guard.strides,
                b_guard.offset,
                &output_shape,
            )?;
            iter_a
                .zip(iter_b)
                .map(|(va, vb)| ge_kernel::<f32>(va, vb))
                .collect()
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;
            let iter_a = NdArrayBroadcastingIterF64::new(
                a_buffer,
                &a_guard.shape,
                &a_guard.strides,
                a_guard.offset,
                &output_shape,
            )?;
            let iter_b = NdArrayBroadcastingIterF64::new(
                b_buffer,
                &b_guard.shape,
                &b_guard.strides,
                b_guard.offset,
                &output_shape,
            )?;
            iter_a
                .zip(iter_b)
                .map(|(va, vb)| ge_kernel::<f64>(va, vb))
                .collect()
        }
    };

    if result_data_vec.len() != numel {
        return Err(NeuraRustError::InternalError(format!(
            "ge_op: Output vec len {} mismatch with expected numel {}",
            result_data_vec.len(),
            numel
        )));
    }

    drop(a_guard);
    drop(b_guard);

    // --- Create Output Tensor (Always F32, no autograd) ---
    Tensor::new(result_data_vec, output_shape)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::from_vec_f32;
    use crate::utils::testing::check_tensor_near;

    #[test]
    fn test_ge_simple() -> Result<(), NeuraRustError> {
        let a = from_vec_f32(vec![1.0, 5.0, 3.0], vec![3])?;
        let b = from_vec_f32(vec![1.0, 2.0, 4.0], vec![3])?;
        let result = ge_op(&a, &b)?;
        check_tensor_near(&result, &[3], &[1.0, 1.0, 0.0], 1e-7);
        assert!(!result.requires_grad());
        Ok(())
    }

    #[test]
    fn test_ge_broadcast() -> Result<(), NeuraRustError> {
        let a = from_vec_f32(vec![3.0], vec![1])?;
        let b = from_vec_f32(vec![1.0, 3.0, 4.0, 2.0], vec![2, 2])?;
        let result = ge_op(&a, &b)?;
        check_tensor_near(&result, &[2, 2], &[1.0, 1.0, 0.0, 1.0], 1e-7);
        Ok(())
    }

     #[test]
    fn test_ge_f64() -> Result<(), NeuraRustError> {
        let a = Tensor::new_f64(vec![1.0, 5.0, 3.0], vec![3])?;
        let b = Tensor::new_f64(vec![1.0, 2.0, 4.0], vec![3])?;
        let result = ge_op(&a, &b)?;
        check_tensor_near(&result, &[3], &[1.0, 1.0, 0.0], 1e-7);
        assert_eq!(result.dtype(), DType::F32); // Output is always F32 for comparisons for now
        Ok(())
    }
} 