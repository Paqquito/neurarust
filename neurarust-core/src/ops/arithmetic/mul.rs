use crate::{
    autograd::{backward_op::BackwardOp, graph::NodeId},
    error::NeuraRustError,
    ops::traits::NeuraNumeric,
    tensor::Tensor,
    tensor_data::TensorData,
};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Generic kernel for element-wise multiplication.
fn mul_kernel<T: NeuraNumeric>(a: T, b: T) -> T {
    a * b
}

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise multiplication operation.
///
/// Stores clones of the input tensors (`a`, `b`) needed for gradient calculation,
/// references to their nodes (`a_node`, `b_node`) for graph linkage, original shapes
/// (`a_shape`, `b_shape`) for gradient reduction, and flags indicating if inputs
/// required gradients (`a_requires_grad`, `b_requires_grad`).
#[derive(Debug)]
struct MulBackward {
    /// Clone of the first input tensor (`a`).
    a: Tensor,
    /// Clone of the second input tensor (`b`).
    b: Tensor,
    /// Optional reference to the first input node for graph traversal.
    a_node: Option<Arc<RwLock<TensorData>>>,
    /// Optional reference to the second input node for graph traversal.
    b_node: Option<Arc<RwLock<TensorData>>>,
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
impl BackwardOp for MulBackward {
    /// Computes gradients for the multiplication operation \( z = a \cdot b \).
    ///
    /// The gradients are:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \cdot b \quad \text{and} \quad \frac{dL}{db} = \frac{dL}{dz} \cdot a \\]
    /// Where \( \frac{dL}{dz} \) is `grad_output`.
    ///
    /// If broadcasting occurred, the computed gradients are reduced to the original
    /// input shapes using `reduce_gradient_to_shape`.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut result_grads = Vec::new();

        if self.a_requires_grad {
            let unreduced_grad_a = mul_op(grad_output, &self.b)?;
            let grad_a = unreduced_grad_a.reduce_to_shape(&self.a_shape)?;
            result_grads.push(grad_a);
        }

        if self.b_requires_grad {
            let unreduced_grad_b = mul_op(grad_output, &self.a)?;
            let grad_b = unreduced_grad_b.reduce_to_shape(&self.b_shape)?;
            result_grads.push(grad_b);
        }

        Ok(result_grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` and `b` of the forward `mul_op`.
    fn inputs(&self) -> Vec<NodeId> {
        let mut ids = Vec::new();
        if let Some(node) = &self.a_node {
            ids.push(Arc::as_ptr(node));
        }
        if let Some(node) = &self.b_node {
            ids.push(Arc::as_ptr(node));
        }
        ids
    }
}

// --- Forward Operation ---

/// Performs element-wise multiplication of two tensors (`a * b`), supporting broadcasting.
///
/// Computes the product of two tensors, element by element. If the tensors have different
/// but compatible shapes, broadcasting rules are applied.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `a`: The first input `Tensor`.
/// * `b`: The second input `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise product, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - Tensors are not on the CPU (`DeviceMismatch`).
/// - Tensors have different `DType`s (`DataTypeMismatch`).
/// - Tensors have incompatible shapes for broadcasting (`BroadcastError`).
/// - An internal error occurs during computation or memory allocation.
pub fn mul_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_clone = a.clone();
    let b_clone = b.clone();

    // Call the centralized helper
    crate::ops::arithmetic::apply_binary_op_broadcasted(
        a,
        b,
        // Closure for F32, calling the generic kernel
        |va, vb| mul_kernel::<f32>(va, vb),
        // Closure for F64, calling the generic kernel
        |va, vb| mul_kernel::<f64>(va, vb),
        // Closure captures and moves the clones
        move |a_node_opt, b_node_opt, a_shape, b_shape, a_req, b_req| {
            Arc::new(MulBackward {
                a: a_clone,
                b: b_clone,
                a_node: a_node_opt,
                b_node: b_node_opt,
                a_shape,
                b_shape,
                a_requires_grad: a_req,
                b_requires_grad: b_req,
            })
        },
        "mul_op", // Operation name for errors
    )
}

/// Performs element-wise multiplication of a tensor by a scalar (`tensor * scalar`).
///
/// This operation is not in-place; it returns a new tensor.
/// It currently supports F32 and F64 data types.
///
/// # Arguments
/// * `tensor`: The input `Tensor`.
/// * `scalar`: The scalar value (must match tensor's DType implicitly, or be convertible).
///
/// # Returns
/// A `Result` containing a new `Tensor` or a `NeuraRustError`.
pub fn mul_op_scalar<T: NeuraNumeric + Debug>(tensor: &Tensor, scalar: T) -> Result<Tensor, NeuraRustError> {
    let tensor_guard = tensor.read_data();

    if tensor_guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "mul_op_scalar".to_string(),
            expected: crate::device::StorageDevice::CPU,
            actual: tensor_guard.device,
        });
    }

    let output_shape = tensor_guard.shape.clone();
    let numel = tensor_guard.numel();
    let new_requires_grad = tensor_guard.requires_grad; // Scalar ops usually don't introduce new grad_fn directly this way
                                                     // unless the scalar itself is a Tensor, which is not the case here.

    let output_tensor = match tensor_guard.dtype {
        crate::types::DType::F32 => {
            let scalar_f32 = scalar.to_f32().ok_or_else(|| NeuraRustError::DataTypeMismatch {
                operation: "mul_op_scalar (scalar conversion to f32)".to_string(),
                expected: crate::types::DType::F32,
                actual: tensor_guard.dtype, // This actual is a bit misleading, it's about the scalar type
            })?;
            let data_buffer = tensor_guard.buffer.try_get_cpu_f32()?;
            let mut new_data = Vec::with_capacity(numel);
            if numel == 0 {
                // Handle empty tensor case
            } else if tensor_guard.is_contiguous() {
                let slice_data = &data_buffer[tensor_guard.offset..tensor_guard.offset + numel];
                for &val in slice_data {
                    new_data.push(val * scalar_f32);
                }
            } else {
                // Non-contiguous: iterate using logical to physical index
                for i in 0..numel {
                    let coords = crate::tensor::utils::index_to_coord(i, &tensor_guard.shape);
                    let physical_offset = tensor_guard.get_offset(&coords);
                    new_data.push(data_buffer[physical_offset] * scalar_f32);
                }
            }
            drop(tensor_guard); // Release lock before creating new tensor
            Tensor::new(new_data, output_shape)?
        }
        crate::types::DType::F64 => {
            let scalar_f64 = scalar.to_f64().ok_or_else(|| NeuraRustError::DataTypeMismatch {
                operation: "mul_op_scalar (scalar conversion to f64)".to_string(),
                expected: crate::types::DType::F64,
                actual: tensor_guard.dtype,
            })?;
            let data_buffer = tensor_guard.buffer.try_get_cpu_f64()?;
            let mut new_data = Vec::with_capacity(numel);
            if numel == 0 {
                // Handle empty tensor case
            } else if tensor_guard.is_contiguous() {
                let slice_data = &data_buffer[tensor_guard.offset..tensor_guard.offset + numel];
                for &val in slice_data {
                    new_data.push(val * scalar_f64);
                }
            } else {
                for i in 0..numel {
                    let coords = crate::tensor::utils::index_to_coord(i, &tensor_guard.shape);
                    let physical_offset = tensor_guard.get_offset(&coords);
                    new_data.push(data_buffer[physical_offset] * scalar_f64);
                }
            }
            drop(tensor_guard);
            Tensor::new_f64(new_data, output_shape)?
        }
        // Add other DTypes if supported by NeuraNumeric and Tensor creation
    };

    if new_requires_grad {
        // If the input tensor required grad, the output should also (conceptually).
        // However, a simple scalar multiplication usually doesn't add a new node to the graph
        // unless the scalar is also a tensor. For this op, we assume the scalar is a constant.
        // The backward pass of the *next* operation using this result would need to propagate
        // to the original tensor that `tensor` was derived from, if any.
        // For now, just propagate `requires_grad`.
        // A more sophisticated autograd would handle this. If `tensor` had a `grad_fn`,
        // this new tensor should not, as it's a result of an op not tracked by a BackwardOp here.
        // Or, mul_op_scalar should take part in autograd graph if tensor requires_grad.
        // For simplicity in an optimizer step (where grads are already computed):
        output_tensor.set_requires_grad(true)?; // Or false, depending on autograd design for such ops
    }

    Ok(output_tensor)
}

// --- Tests --- 
// Link the external test file
#[cfg(test)]
#[path = "mul_test.rs"]
mod tests;

/* --- REMOVED previous comments about removed internal module --- 
// --- REMOVED internal tests module --- 
// Tests for this module can be found in src/ops/arithmetic/mul_test.rs

#[cfg(test)]
mod tests {
    // ... (contenu de l'ancien module de tests) ...
}
*/
