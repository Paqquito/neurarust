use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;
use crate::ops::traits::NeuraNumeric;
use approx::AbsDiffEq;
use crate::tensor::iter_utils::NdArrayBroadcastingIter;

/// Compares two NeuraNumeric values for equality, returning 1.0f32 or 0.0f32.
/// Uses absolute difference comparison with a default epsilon for floating types.
#[inline]
fn equal_kernel<T>(a: T, b: T) -> f32
where
    T: NeuraNumeric + AbsDiffEq<Epsilon = T>,
{
    // Use approx crate's abs_diff_eq for robust float comparison
    // The default epsilon is likely suitable.
    if a.abs_diff_eq(&b, T::default_epsilon()) {
        1.0f32
    } else {
        0.0f32
    }
}

/// Performs element-wise equality comparison (`a == b`) between two tensors.
///
/// Compares elements of `a` and `b` after broadcasting to a common shape.
/// Due to floating-point inaccuracies, equality for `F32` is checked using
/// a small tolerance (epsilon, currently `1e-6`):
/// `|a_val - b_val| < epsilon`.
///
/// The result is a tensor with the broadcasted shape and `DType::F32`, containing
/// `1.0` where the elements are considered equal and `0.0` otherwise.
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
/// - Tensors are not `DType::F32` (`UnsupportedOperation`).
/// - Tensors have incompatible shapes for broadcasting (`BroadcastError`).
/// - An internal error occurs.
///
/// # Example
/// ```
/// use neurarust_core::tensor::Tensor;
/// use neurarust_core::ops::comparison::equal_op;
///
/// let t1 = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let t2 = Tensor::new(vec![1.0f32, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
/// let t3 = Tensor::new(vec![1.0f32], vec![1]).unwrap(); // Broadcastable scalar-like
///
/// let eq12 = equal_op(&t1, &t2).unwrap();
/// assert_eq!(eq12.shape(), vec![2, 2]);
/// assert_eq!(eq12.get_f32_data().unwrap(), vec![1.0, 0.0, 1.0, 0.0]);
///
/// let eq13 = equal_op(&t1, &t3).unwrap(); // t3 broadcasts to [[1., 1.], [1., 1.]]
/// assert_eq!(eq13.shape(), vec![2, 2]);
/// assert_eq!(eq13.get_f32_data().unwrap(), vec![1.0, 0.0, 0.0, 0.0]);
/// ```
pub fn equal_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check (CPU only) ---
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "equal_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }
    
    // --- DType Check (Require F32 for now) ---
    // TODO: Extend to F64 or other types if needed, kernel supports it.
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("equal_op currently only supports F32, got {:?} and {:?}", a_guard.dtype, b_guard.dtype)
        ));
    }
    let dtype = a_guard.dtype; // Keep track of input dtype

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    // --- DType Dispatch for Computation using Broadcasting Iterators --- 
    // Result is always F32, but iteration depends on input dtype
    let result_data_vec: Vec<f32> = match dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            
            let iter_a = NdArrayBroadcastingIter::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIter::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            
            // Use the generic equal_kernel, result is f32
            iter_a.zip(iter_b)
                .map(|(va, vb)| equal_kernel::<f32>(va, vb))
                .collect()
        }
        DType::F64 => { // Could be enabled later
            // let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            // let b_buffer = b_guard.buffer.try_get_cpu_f64()?;
            // let iter_a = NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            // let iter_b = NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            // iter_a.zip(iter_b).map(|(va, vb)| equal_kernel::<f64>(va, vb)).collect()
            return Err(NeuraRustError::UnsupportedOperation(
                 "equal_op currently only supports F32".to_string()
            )); // Keep F32 only for now
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    };

    if result_data_vec.len() != numel {
         return Err(NeuraRustError::InternalError(format!(
            "equal_op: Output vec len {} mismatch with expected numel {}",
             result_data_vec.len(), numel
        )));
    }
    
    drop(a_guard);
    drop(b_guard);
    
    // --- Create Output Tensor (Always F32, no autograd) --- 
    Tensor::new(result_data_vec, output_shape)
}


// --- Tests ---
#[cfg(test)]
#[path = "equal_test.rs"]
mod tests; // Link to the test file 