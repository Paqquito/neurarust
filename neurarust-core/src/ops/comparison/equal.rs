use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

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

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "equal_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "equal_op currently only supports CPU".to_string(),
        ));
    }

    // --- DType Check (Allow comparison between same supported types) ---
    // For now, strictly require F32 for both, but this could be expanded later.
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("equal_op currently only supports F32, got {:?} and {:?}", a_guard.dtype, b_guard.dtype)
        ));
    }
    // Output is always F32 (for masking purposes)
    let _output_dtype = DType::F32;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Extract Data & Metadata ---
    let a_shape = a_guard.shape.clone();
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;

    let a_buffer_data_arc = a_guard.buffer().try_get_cpu_f32()?.clone(); 
    let b_buffer_data_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    
    // No need to clone nodes, comparison ops don't participate in autograd graph

    drop(a_guard);
    drop(b_guard);

    // --- Calculation Logic (Manual Broadcasting) ---
    let numel_out = output_shape.iter().product();
    let mut result_data_vec: Vec<f32> = Vec::with_capacity(numel_out);

    let a_data = a_buffer_data_arc.as_slice();
    let b_data = b_buffer_data_arc.as_slice();

    let mut a_indices = vec![0; a_shape.len()];
    let mut b_indices = vec![0; b_shape.len()];
    let mut current_indices = vec![0; output_shape.len()];
    let output_rank = output_shape.len();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    // Epsilon for float comparison
    // TODO: Make this configurable or use a crate like `approx`?
    const EPSILON: f32 = 1e-6; 

    for i in 0..numel_out {
        let mut current_linear = i;
        for dim in (0..output_rank).rev() {
            let shape_val = output_shape[dim];
            if shape_val > 0 { current_indices[dim] = current_linear % shape_val; current_linear /= shape_val; } else { current_indices[dim] = 0; }
        }
        for dim in 0..output_rank {
            let out_idx = current_indices[dim];
            let a_dim_idx = (dim as isize) - (output_rank as isize - a_rank as isize); if a_dim_idx >= 0 { a_indices[a_dim_idx as usize] = if a_shape[a_dim_idx as usize] == 1 { 0 } else { out_idx }; }
            let b_dim_idx = (dim as isize) - (output_rank as isize - b_rank as isize); if b_dim_idx >= 0 { b_indices[b_dim_idx as usize] = if b_shape[b_dim_idx as usize] == 1 { 0 } else { out_idx }; }
        }
        let a_physical_offset = a_offset + a_indices.iter().zip(a_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        let b_physical_offset = b_offset + b_indices.iter().zip(b_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        
        // Perform comparison (using epsilon for floats)
        let a_val = a_data[a_physical_offset];
        let b_val = b_data[b_physical_offset];
        let are_equal = (a_val - b_val).abs() < EPSILON;
        result_data_vec.push(if are_equal { 1.0 } else { 0.0 });
    }
    
    // --- Create Output Tensor --- 
    // Output tensor DOES NOT require grad and has no grad_fn.
    Tensor::new(result_data_vec, output_shape)
}


// --- Tests ---
#[cfg(test)]
#[path = "equal_test.rs"]
mod tests; // Link to the test file 