// neurarust-core/src/ops/arithmetic/add.rs

use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use crate::types::DType;
use std::sync::RwLock;
use crate::tensor::utils::broadcast_shapes;
use std::sync::Arc;

use std::fmt::Debug;

// --- Backward Operation Structure ---
#[derive(Debug)]
struct AddBackward { // Removed <T>
    // Store context needed for backward pass and graph traversal
    a_node: Arc<RwLock<TensorData>>,
    b_node: Arc<RwLock<TensorData>>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    a_requires_grad: bool, // To know which grads to compute/return
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for AddBackward { // Remove <T>
    /// Computes gradients for the addition operation z = a + b.
    /// grad(a) = grad_output * dz/da = grad_output * 1 = grad_output
    /// grad(b) = grad_output * dz/db = grad_output * 1 = grad_output
    /// Need to handle broadcasting by reducing gradients back to original shapes.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // TODO: Implement gradient reduction for broadcasting
        // let grad_a = reduce_gradient_to_shape(grad_output, &self.a_shape)?;
        // let grad_b = reduce_gradient_to_shape(grad_output, &self.b_shape)?;
        // For now, assume no broadcasting or handle later
        let mut grads = Vec::with_capacity(2);

        if self.a_requires_grad {
            let grad_a = reduce_gradient_to_shape(grad_output, &self.a_shape)?;
            grads.push(grad_a);
        }
        if self.b_requires_grad {
             let grad_b = reduce_gradient_to_shape(grad_output, &self.b_shape)?;
            // If a didn't require grad, b is the first element needed
             grads.push(grad_b);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> { // Keep non-generic
        // Return IDs of inputs that required grad, in the order (a, b)
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

/// Helper function to reduce a gradient tensor to match a target shape.
/// This is used when backpropagating through broadcast operations.
/// It sums the gradient along the broadcasted dimensions.
fn reduce_gradient_to_shape(
    grad: &Tensor,
    target_shape: &[usize],
) -> Result<Tensor, NeuraRustError> {
    let grad_shape = grad.shape();
    let grad_rank = grad_shape.len();
    let target_rank = target_shape.len();

    if grad_rank == 0 {
        // If grad is scalar, return it directly (no reduction needed)
        return Ok(grad.clone());
    }
    if target_rank > grad_rank {
         return Err(NeuraRustError::InternalError(format!(
             "Cannot reduce gradient of rank {} to target shape of rank {}",
             grad_rank, target_rank
         )));
    }

    // Identify axes to sum over
    let mut axes_to_sum = Vec::new();
    for i in 0..(grad_rank - target_rank) {
        axes_to_sum.push(i); // Sum leading dimensions added by broadcasting
    }
    for i in 0..target_rank {
        let target_dim = target_shape[i];
        let grad_dim = grad_shape[i + (grad_rank - target_rank)];
        if target_dim == 1 && grad_dim > 1 {
            axes_to_sum.push(i + (grad_rank - target_rank)); // Sum broadcasted dimension
        } else if target_dim != grad_dim {
            // This case should ideally be caught by broadcast check earlier
             return Err(NeuraRustError::InternalError(format!(
                 "Shape mismatch during gradient reduction: grad_dim={}, target_dim={} at index {}",
                 grad_dim, target_dim, i
             )));
        }
    }

    // Perform the sum reduction
    let reduced_grad = if !axes_to_sum.is_empty() {
        crate::ops::reduction::sum_op(grad, Some(&axes_to_sum), true)? // Keep dims true for now
    } else {
        grad.clone()
    };

    // Reshape to remove summed dimensions if keep_dims was true and rank differs
    // or if target_shape contained ones that were summed.
     // Check if final shape matches target_shape after potential reduction
     let final_shape: Vec<usize> = target_shape.iter().cloned().collect();
     if reduced_grad.shape() != final_shape {
         // If keep_dims=true created shape like [1, 5] instead of [5],
         // or if target was [1, 5] and we summed axis 0 -> [1, 5]
         // We need a reshape.
         crate::ops::view::reshape_op(&reduced_grad, final_shape)
     } else {
        Ok(reduced_grad)
    }
}

// --- Forward Operation ---

/// Performs element-wise addition of two tensors, supporting broadcasting.
pub fn add_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.data.read().unwrap();
    let b_guard = b.data.read().unwrap();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "add_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    if a_guard.device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            "add_op currently only supports CPU tensors.".to_string(),
        ));
    }

    // --- DType Check ---
    // For now, assume both are F32 or return error
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "add_op currently only supports F32 tensors.".to_string(),
        ));
    }
    // Keep track of the output dtype even if only F32 for now
    let _output_dtype = DType::F32;

    // --- Shape Broadcasting --- 
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Data Access & Metadata Extraction --- 
    // Extract ALL necessary info while guards are held
    let _device = a_guard.device; // Devices are checked to be the same
    let a_shape = a_guard.shape.clone();
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;
    let a_requires_grad = a_guard.requires_grad;
    let b_requires_grad = b_guard.requires_grad;
    
    // Get buffer Arcs (assuming F32 CPU based on checks above)
    // Clone the INNER Arc<Vec<f32>> here to own the data reference
    let a_buffer_data_arc = a_guard.buffer().try_get_cpu_f32()?.clone(); 
    let b_buffer_data_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    let a_data = a_buffer_data_arc.as_slice(); // Get slices from owned Arcs
    let b_data = b_buffer_data_arc.as_slice();

    // Keep input TensorData Arcs if needed for backward pass
    let a_node_arc = if a_requires_grad || b_requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if a_requires_grad || b_requires_grad { Some(b.data.clone()) } else { None };

    // Drop guards explicitly NOW after extracting everything
    drop(a_guard);
    drop(b_guard);

    // --- Computation --- 
    let numel_out = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_out);

    // Prepare indices and strides for iteration
    let mut a_indices = vec![0; a_shape.len()];
    let mut b_indices = vec![0; b_shape.len()];
    let mut current_indices = vec![0; output_shape.len()];
    let output_rank = output_shape.len();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    for i in 0..numel_out {
        // Calculate multi-dimensional index from linear index i for output_shape
        let mut current_linear = i;
        for dim in (0..output_rank).rev() {
            let shape_val = output_shape[dim];
            if shape_val > 0 { // Avoid division by zero for empty dimensions
                 current_indices[dim] = current_linear % shape_val;
                 current_linear /= shape_val;
            } else {
                 current_indices[dim] = 0;
                 // current_linear remains 0 if shape_val is 0
            }
        }

        // Calculate corresponding indices for a and b considering broadcasting rules
        for dim in 0..output_rank {
            let out_idx = current_indices[dim];
            
            // Index for a (handle rank difference)
            let a_dim_idx = (dim as isize) - (output_rank as isize - a_rank as isize);
            if a_dim_idx >= 0 {
                let a_dim_idx = a_dim_idx as usize;
                // If dim size in a is 1, index is 0, else it's the output index
                a_indices[a_dim_idx] = if a_shape[a_dim_idx] == 1 { 0 } else { out_idx };
            }

            // Index for b (handle rank difference)
            let b_dim_idx = (dim as isize) - (output_rank as isize - b_rank as isize);
            if b_dim_idx >= 0 {
                 let b_dim_idx = b_dim_idx as usize;
                // If dim size in b is 1, index is 0, else it's the output index
                b_indices[b_dim_idx] = if b_shape[b_dim_idx] == 1 { 0 } else { out_idx };
            }
        }

        // Calculate physical offsets using strides
        let a_physical_offset = a_offset + a_indices.iter().zip(a_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        let b_physical_offset = b_offset + b_indices.iter().zip(b_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        
        // Perform addition
        result_data_vec.push(a_data[a_physical_offset] + b_data[b_physical_offset]);
    }
    
    // --- Create Result Tensor --- 
    let output_tensor = Tensor::from_vec_f32(result_data_vec, output_shape)?; // Use helper

    // --- Autograd Linkage --- 
    // Link if either input requires gradient
    if a_requires_grad || b_requires_grad {
         if let (Some(a_arc), Some(b_arc)) = (a_node_arc, b_node_arc) {
             // Create AddBackward with correct context (shapes, node arcs, flags)
             let backward_context = AddBackward {
                 a_node: a_arc,
                 b_node: b_arc,
                 a_shape, // Pass original shapes
                 b_shape,
                 a_requires_grad,
                 b_requires_grad,
             };
             // Set requires_grad and grad_fn on the output tensor
             let mut output_guard = output_tensor.write_data();
             output_guard.requires_grad = true;
             output_guard.grad_fn = Some(Arc::new(backward_context));
         } else {
             // This case should not be reachable if requires_grad flags are correct
             return Err(NeuraRustError::InternalError("Add op requires grad but Arc nodes unavailable".to_string()));
         }
    }

    Ok(output_tensor)
}

// Re-enable the test module link
#[cfg(test)]
#[path = "add_test.rs"]
mod tests;
