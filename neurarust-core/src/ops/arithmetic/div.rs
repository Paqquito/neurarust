use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use crate::ops::arithmetic::mul::mul_op;
use crate::ops::arithmetic::neg::neg_op;

// Keep Zero trait for division check
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use num_traits::Zero;

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise division operation.
///
/// Stores references to input tensors (`a_node`, `b_node`) and a clone of `b` (`b_tensor_clone`)
/// needed for calculating gradients. Also stores flags indicating if inputs required gradients.
#[derive(Debug)]
struct DivBackward {
    /// Reference counted pointer to the numerator tensor's data (`a`).
    a_node: Arc<RwLock<TensorData>>,
    /// Reference counted pointer to the denominator tensor's data (`b`).
    b_node: Arc<RwLock<TensorData>>,
    /// Clone of the denominator tensor (`b`), needed for gradient calculation.
    b_tensor_clone: Tensor,
    /// Flag indicating if the numerator tensor (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the denominator tensor (`b`) required gradients.
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for DivBackward {
    /// Computes gradients for the division operation \( z = a / b \).
    ///
    /// Using the chain rule \( \frac{dL}{dx} = \frac{dL}{dz} \cdot \frac{dz}{dx} \), the gradients are:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{dz}{da} = \frac{dL}{dz} \cdot \frac{1}{b} \\]
    /// \\[ \frac{dL}{db} = \frac{dL}{dz} \cdot \frac{dz}{db} = \frac{dL}{dz} \cdot \left( -\frac{a}{b^2} \right) \\]
    ///
    /// Where \( \frac{dL}{dz} \) is `grad_output`.
    ///
    /// **Broadcasting Handling:** Similar to addition, if broadcasting occurred, the computed gradients
    /// (dL/da and dL/db) are reduced to the original shapes of `a` and `b` using the
    /// [`reduce_to_shape`](../broadcast_utils/fn.reduce_to_shape.html) helper function.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::with_capacity(2);

        if self.a_requires_grad {
            let a_guard = self.a_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock A node in DivBackward".to_string()))?;
            // grad(a) = grad_output / b
            let grad_a_unreduced = div_op(grad_output, &self.b_tensor_clone)?;
            // Reduce gradient if broadcasting occurred using the Tensor method
            let grad_a = grad_a_unreduced.reduce_to_shape(&a_guard.shape)?;
            grads.push(grad_a);
        }

        if self.b_requires_grad {
            let b_guard = self.b_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock B node in DivBackward".to_string()))?;
            let a_tensor = Tensor { data: self.a_node.clone() };
            
            // grad(b) = grad_output * (-a / b^2)
            let b_squared = mul_op(&self.b_tensor_clone, &self.b_tensor_clone)?;
            let neg_a = neg_op(&a_tensor)?;
            let inner_term = div_op(&neg_a, &b_squared)?;
            let grad_b_unreduced = mul_op(grad_output, &inner_term)?;
             // Reduce gradient if broadcasting occurred using the Tensor method
            let grad_b = grad_b_unreduced.reduce_to_shape(&b_guard.shape)?;
            grads.push(grad_b);
        }

        Ok(grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` (numerator) and `b` (denominator).
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

// --- Forward Operation ---

/// Performs element-wise division of two tensors (`a / b`), supporting broadcasting.
///
/// Computes the division of `a` by `b`, element by element. If the tensors have different
/// but compatible shapes, broadcasting rules are applied.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `a`: The numerator `Tensor`.
/// * `b`: The denominator `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise division, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - Tensors are not on the CPU (`DeviceMismatch`).
/// - Tensors are not `DType::F32` (`UnsupportedOperation`).
/// - Tensors have incompatible shapes for broadcasting (`BroadcastError`).
/// - Division by zero occurs (`DivisionByZero`).
/// - An internal error occurs during computation or memory allocation.
pub fn div_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor A data for reading".to_string()))?;
    let b_guard = b.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor B data for reading".to_string()))?;

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "div_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            "div_op currently only supports CPU tensors.".to_string(),
        ));
    }

    // --- DType Check ---
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "div_op currently only supports F32 tensors.".to_string(),
        ));
    }
    let _output_dtype = DType::F32;

    // --- Shape Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Extract Data & Metadata ---
    let a_shape = a_guard.shape.clone(); 
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;
    let a_requires_grad = a_guard.requires_grad;
    let b_requires_grad = b_guard.requires_grad;

    let a_buffer_data_arc = a_guard.buffer().try_get_cpu_f32()?.clone(); 
    let b_buffer_data_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    let a_node_arc = if a_requires_grad || b_requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if a_requires_grad || b_requires_grad { Some(b.data.clone()) } else { None };
    let b_tensor_clone = if b_requires_grad { Some(b.clone()) } else { None };

    drop(a_guard);
    drop(b_guard);

    // --- Calculation Logic (Manual Broadcasting) ---
    let numel_out = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_out);
    let a_data = a_buffer_data_arc.as_slice();
    let b_data = b_buffer_data_arc.as_slice();

    let mut a_indices = vec![0; a_shape.len()];
    let mut b_indices = vec![0; b_shape.len()];
    let mut current_indices = vec![0; output_shape.len()];
    let output_rank = output_shape.len();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

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
        
        let divisor = b_data[b_physical_offset];
        if divisor.is_zero() {
            return Err(NeuraRustError::DivisionByZero); 
        }
        result_data_vec.push(a_data[a_physical_offset] / divisor);
    }
    let result_buffer_arc = Arc::new(result_data_vec);

    // --- Create Output TensorData ---
    let output_td = TensorData::new(
        result_buffer_arc.as_ref().clone(),
        output_shape,
    )?;
    let result_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    // --- Autograd Setup ---
    // Determine if autograd is needed based on original inputs
    let autograd_needed = a_requires_grad || b_requires_grad;

    if autograd_needed {
        // We NEED the original Arcs and potentially the b clone if autograd is needed
        // Retrieve them from the Options created earlier.
        let a_arc = a_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing a_node_arc when autograd needed".to_string()))?;
        let b_arc = b_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing b_node_arc when autograd needed".to_string()))?;
        // Only need b_clone if b itself requires grad for the backward calculation
        let b_clone_for_backward = if b_requires_grad { 
            b_tensor_clone.ok_or_else(|| NeuraRustError::InternalError("Missing b_tensor_clone when b requires grad".to_string()))?
        } else {
            // If b doesn't require grad, we still need *a* tensor b for the backward pass of a.
            // Clone it here if it wasn't cloned earlier.
            b.clone() 
        };

        let mut output_guard = result_tensor.data.write().map_err(|_| NeuraRustError::InternalError("Failed to lock output tensor data for writing".to_string()))?;
        output_guard.requires_grad = true; // Set requires_grad on the output
        let backward_context = DivBackward { 
            a_node: a_arc, 
            b_node: b_arc, 
            b_tensor_clone: b_clone_for_backward, // Pass the correctly obtained clone
            a_requires_grad, 
            b_requires_grad 
        };
        output_guard.grad_fn = Some(Arc::new(backward_context));
        println!("DivBackward grad_fn set for div result."); // Debug print
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
#[path = "div_test.rs"]
mod tests;
