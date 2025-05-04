use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

/// Performs element-wise equality comparison between two tensors.
///
/// Returns a new tensor containing 1.0f32 where elements are equal, and 0.0f32 otherwise.
/// Supports broadcasting.
/// Currently only supports F32 CPU tensors.
/// This operation does not support automatic differentiation.
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