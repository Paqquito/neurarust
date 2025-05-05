use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use std::sync::Arc;

use std::fmt::Debug;
use std::sync::RwLock;

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise subtraction operation.
///
/// Stores references to input tensor nodes (`a_node`, `b_node`) and flags
/// indicating if they required gradients (`a_requires_grad`, `b_requires_grad`).
/// Original shapes are implicitly handled by `Tensor::reduce_to_shape` in the backward pass.
#[derive(Debug)]
struct SubBackward {
    /// Reference counted pointer to the first input tensor's data (`a` in `a - b`).
    a_node: Arc<RwLock<TensorData>>,
    /// Reference counted pointer to the second input tensor's data (`b` in `a - b`).
    b_node: Arc<RwLock<TensorData>>,
    /// Flag indicating if the first input (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the second input (`b`) required gradients.
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for SubBackward {
    /// Computes gradients for the subtraction operation \( z = a - b \).
    ///
    /// The gradients are:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \quad \text{and} \quad \frac{dL}{db} = - \frac{dL}{dz} \\]
    /// Where \( \frac{dL}{dz} \) is `grad_output`.
    ///
    /// If broadcasting occurred, gradients are reduced back to the original input shapes
    /// using [`Tensor::reduce_to_shape`](../../tensor/broadcast_utils/struct.Tensor.html#method.reduce_to_shape).
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::with_capacity(2);

        // Use reduce_to_shape for gradients
        if self.a_requires_grad {
            let a_guard = self.a_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock A node in SubBackward".to_string()))?;
            let grad_a = grad_output.reduce_to_shape(&a_guard.shape)?;
            grads.push(grad_a);
        }

        if self.b_requires_grad {
            let b_guard = self.b_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock B node in SubBackward".to_string()))?;
            let grad_b_unreduced = crate::ops::arithmetic::neg_op(grad_output)?;
            let grad_b = grad_b_unreduced.reduce_to_shape(&b_guard.shape)?;
            grads.push(grad_b);
        }
        
        Ok(grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` and `b` of the forward `sub_op`.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        // Return pointers only for inputs that required grad
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

// --- Forward Operation ---

/// Performs element-wise subtraction of two tensors (`a - b`), supporting broadcasting.
///
/// Computes the difference between two tensors, element by element. If the tensors have different
/// but compatible shapes, broadcasting rules are applied.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `a`: The first input `Tensor` (minuend).
/// * `b`: The second input `Tensor` (subtrahend).
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise difference, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - Tensors are not on the CPU (`DeviceMismatch`).
/// - Tensors are not `DType::F32` (`UnsupportedOperation`).
/// - Tensors have incompatible shapes for broadcasting (`BroadcastError`).
/// - An internal error occurs during computation or memory allocation.
pub fn sub_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor A data for reading".to_string()))?;
    let b_guard = b.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor B data for reading".to_string()))?;

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "sub_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "sub_op currently only supports CPU".to_string()
        ));
    }

    // --- DType Check & Promotion (Simplified for F32 only) ---
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("sub_op currently only supports F32, got {:?} and {:?}", a_guard.dtype, b_guard.dtype)
        ));
    }
    let _output_dtype = DType::F32;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Extract Data & Metadata ---
    let a_shape = a_guard.shape.clone(); // Keep original shapes for potential backward reduction
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;
    let a_requires_grad = a_guard.requires_grad;
    let b_requires_grad = b_guard.requires_grad;

    let a_buffer_data_arc = a_guard.buffer().try_get_cpu_f32()?.clone(); 
    let b_buffer_data_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    // Keep input TensorData Arcs if needed for backward pass
    let a_node_arc = if a_requires_grad || b_requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if a_requires_grad || b_requires_grad { Some(b.data.clone()) } else { None };

    // Drop guards before computation
    drop(a_guard);
    drop(b_guard);

    // --- Calculation Logic (Manual Broadcasting) ---
    let numel_out = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_out);

    let a_data = a_buffer_data_arc.as_slice();
    let b_data = b_buffer_data_arc.as_slice();

    // Prepare indices and strides for iteration (similar to add_op)
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
            }
        }

        // Calculate corresponding indices for a and b considering broadcasting rules
        for dim in 0..output_rank {
            let out_idx = current_indices[dim];
            
            // Index for a (handle rank difference)
            let a_dim_idx = (dim as isize) - (output_rank as isize - a_rank as isize);
            if a_dim_idx >= 0 {
                let a_dim_idx = a_dim_idx as usize;
                a_indices[a_dim_idx] = if a_shape[a_dim_idx] == 1 { 0 } else { out_idx };
            }

            // Index for b (handle rank difference)
            let b_dim_idx = (dim as isize) - (output_rank as isize - b_rank as isize);
            if b_dim_idx >= 0 {
                 let b_dim_idx = b_dim_idx as usize;
                b_indices[b_dim_idx] = if b_shape[b_dim_idx] == 1 { 0 } else { out_idx };
            }
        }

        // Calculate physical offsets using strides
        let a_physical_offset = a_offset + a_indices.iter().zip(a_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        let b_physical_offset = b_offset + b_indices.iter().zip(b_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        
        // Perform subtraction
        result_data_vec.push(a_data[a_physical_offset] - b_data[b_physical_offset]);
    }
    let result_buffer_arc = Arc::new(result_data_vec); // Arc the final Vec

    // --- Create Output TensorData ---
    // Correct call to TensorData::new using the signature from tensor_data.rs
    let output_td = TensorData::new(
        result_buffer_arc.as_ref().clone(), // Pass the owned Vec<f32> 
        output_shape, // Pass the shape
    )?;
    let result_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    // --- Autograd Setup ---
    if a_requires_grad || b_requires_grad {
         if let (Some(a_arc), Some(b_arc)) = (a_node_arc, b_node_arc) {
             let mut output_guard = result_tensor.data.write().map_err(|_| NeuraRustError::InternalError("Failed to lock output tensor data for writing".to_string()))?;
             output_guard.requires_grad = true;
             output_guard.grad_fn = Some(Arc::new(SubBackward {
                 a_node: a_arc,
                 b_node: b_arc,
                 a_requires_grad, // Pass flags
                 b_requires_grad,
             }));
             println!("SubBackward grad_fn set for sub result."); // Temporary debug print
         }
    }

    Ok(result_tensor)
}

// --- Tests --- 
// Link the external test file
#[cfg(test)]
#[path = "sub_test.rs"]
mod tests;
