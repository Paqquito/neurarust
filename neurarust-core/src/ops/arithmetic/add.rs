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
// Import the iterators
use crate::tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64};

use std::fmt::Debug;

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise addition operation.
///
/// Stores references to the input tensors (`a_node`, `b_node`) and their original shapes
/// to handle gradient reduction correctly during backpropagation, especially when
/// broadcasting was involved in the forward pass.
#[derive(Debug)]
struct AddBackward {
    /// Reference counted pointer to the first input tensor's data (`a`).
    a_node: Arc<RwLock<TensorData>>,
    /// Reference counted pointer to the second input tensor's data (`b`).
    b_node: Arc<RwLock<TensorData>>,
    /// Original shape of the first input tensor (`a`).
    a_shape: Vec<usize>,
    /// Original shape of the second input tensor (`b`).
    b_shape: Vec<usize>,
    /// Flag indicating if the first input tensor (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the second input tensor (`b`) required gradients.
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for AddBackward {
    /// Computes gradients for the addition operation \( z = a + b \).
    ///
    /// The local gradients are \( \frac{dz}{da} = 1 \) and \( \frac{dz}{db} = 1 \).
    /// Using the chain rule, \( \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{dz}{da} = \frac{dL}{dz} \cdot 1 \)
    /// and \( \frac{dL}{db} = \frac{dL}{dz} \cdot \frac{dz}{db} = \frac{dL}{dz} \cdot 1 \).
    ///
    /// Therefore, the gradient flowing from the output (`grad_output`, representing \( \frac{dL}{dz} \))
    /// is passed back to both inputs.
    ///
    /// **Broadcasting Handling:** If the forward operation involved broadcasting (i.e., `a` and `b`
    /// had different shapes), the `grad_output` (which has the broadcasted shape) needs to be
    /// reduced back to the original shapes of `a` and `b` respectively before being assigned
    /// as their gradients. This reduction is typically done by summing the gradient along the
    /// broadcasted dimensions, which is handled by the [`reduce_gradient_to_shape`] helper function.
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

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` and `b` of the forward `add_op`.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        // Return IDs of inputs that required grad, in the order (a, b)
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

/// Reduces a gradient tensor to match a target shape, typically the shape of an input
/// tensor before broadcasting occurred in the forward pass.
///
/// When broadcasting is used in a forward operation (e.g., adding a `[3, 1]` tensor to a `[3, 5]` tensor),
/// the output gradient will have the broadcasted shape (e.g., `[3, 5]`). To compute the gradient
/// with respect to the original input (e.g., `[3, 1]`), we need to sum the output gradient
/// along the dimensions that were expanded during broadcasting.
///
/// This function identifies these broadcasted dimensions by comparing the `grad` shape
/// with the `target_shape` and performs a sum reduction along those axes.
/// It also handles cases where dimensions were added (e.g., broadcasting a `[3]` vector to `[5, 3]`).
///
/// # Arguments
/// * `grad`: The gradient tensor (having the broadcasted shape).
/// * `target_shape`: The original shape of the input tensor to which the gradient should be reduced.
///
/// # Returns
/// * `Ok(Tensor)`: A new tensor containing the gradient reduced to the `target_shape`.
/// * `Err(NeuraRustError)`: If the shapes are fundamentally incompatible for reduction or if an
///                        internal error occurs during sum or reshape operations.
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
     let final_shape: Vec<usize> = target_shape.to_vec();
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

/// Performs element-wise addition of two tensors (`a + b`), supporting broadcasting.
///
/// Computes the sum of two tensors, element by element. If the tensors have different
/// but compatible shapes, broadcasting rules are applied (similar to NumPy/PyTorch)
/// to make their shapes match before performing the addition.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `a`: The first input `Tensor`.
/// * `b`: The second input `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise sum, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - Tensors are not on the CPU (`DeviceMismatch`).
/// - Tensors have different `DType`s (`DataTypeMismatch`).
/// - Tensors have incompatible shapes for broadcasting (`BroadcastError`).
/// - An internal error occurs during computation or memory allocation.
///
/// # Broadcasting Example
/// ```text
/// a (shape [3, 1]): [[1], [2], [3]]
/// b (shape [  5]): [10, 20, 30, 40, 50]
/// broadcast_shapes(a, b) -> [3, 5]
/// a broadcasts to: [[1, 1, 1, 1, 1],
///                  [2, 2, 2, 2, 2],
///                  [3, 3, 3, 3, 3]]
/// b broadcasts to: [[10, 20, 30, 40, 50],
///                  [10, 20, 30, 40, 50],
///                  [10, 20, 30, 40, 50]]
/// result (shape [3, 5]): [[11, 21, 31, 41, 51],
///                      [12, 22, 32, 42, 52],
///                      [13, 23, 33, 43, 53]]
/// ```
pub fn add_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    // Lock data for reading
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device and DType Checks --- (Remain the same)
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU { /* ... */ }
    if a_guard.dtype != b_guard.dtype { /* ... */ }

    // --- Broadcasting --- (Remains the same)
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    // --- Prepare for Autograd --- (Remains the same)
    let requires_grad = a_guard.requires_grad || b_guard.requires_grad;
    let a_node_arc = if a_guard.requires_grad { Some(Arc::clone(&a.data)) } else { None };
    let b_node_arc = if b_guard.requires_grad { Some(Arc::clone(&b.data)) } else { None };
    let a_shape_clone = a_guard.shape.clone();
    let b_shape_clone = b_guard.shape.clone();
    let a_req_grad_clone = a_guard.requires_grad;
    let b_req_grad_clone = b_guard.requires_grad;

    // --- DType Dispatch for Computation using Iterators --- 
    let output_tensor = match a_guard.dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            
            // Use iterators
            let iter_a = NdArrayBroadcastingIter::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIter::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            
            let output_data_vec: Vec<f32> = iter_a.zip(iter_b).map(|(val_a, val_b)| val_a + val_b).collect();
            
            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!("add_op F32: Output vec len {} mismatch with expected numel {}", output_data_vec.len(), numel)));
            }
            
            drop(a_guard); // Drop guards before creating tensor
            drop(b_guard);
            Tensor::new(output_data_vec, output_shape)?
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;

            // Use iterators
            let iter_a = NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;

            let output_data_vec: Vec<f64> = iter_a.zip(iter_b).map(|(val_a, val_b)| val_a + val_b).collect();

            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!("add_op F64: Output vec len {} mismatch with expected numel {}", output_data_vec.len(), numel)));
            }

            drop(a_guard); // Drop guards before creating tensor
            drop(b_guard);
            Tensor::new_f64(output_data_vec, output_shape)?
        }
    };

    // --- Autograd Setup --- 
    if requires_grad {
         if let (Some(a_arc), Some(b_arc)) = (a_node_arc, b_node_arc) {
             // Create AddBackward with correct context (shapes, node arcs, flags)
             let backward_context = AddBackward {
                 a_node: a_arc,
                 b_node: b_arc,
                 a_shape: a_shape_clone, // Use cloned shapes
                 b_shape: b_shape_clone,
                 a_requires_grad: a_req_grad_clone,
                 b_requires_grad: b_req_grad_clone,
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
