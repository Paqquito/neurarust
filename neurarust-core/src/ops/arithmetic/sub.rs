use std::sync::{Arc, RwLock};
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::autograd::graph::NodeId;
use std::fmt::Debug;
use crate::ops::traits::NeuraNumeric;

// Importer les opérations nécessaires pour backward
use crate::ops::arithmetic::neg::neg_op;

/// Generic kernel for element-wise subtraction.
fn sub_kernel<T: NeuraNumeric>(a: T, b: T) -> T {
    a - b
}

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise subtraction operation.
///
/// Stores references to input tensor nodes (`a_node`, `b_node`) and flags
/// indicating if they required gradients (`a_requires_grad`, `b_requires_grad`).
/// Original shapes are implicitly handled by `Tensor::reduce_to_shape` in the backward pass.
#[derive(Debug)]
struct SubBackward {
    /// Reference counted pointer to the first input tensor's data (`a` in `a - b`).
    a_node: Option<Arc<RwLock<TensorData>>>,
    /// Reference counted pointer to the second input tensor's data (`b` in `a - b`).
    b_node: Option<Arc<RwLock<TensorData>>>,
    /// Flag indicating if the first input (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the second input (`b`) required gradients.
    b_requires_grad: bool,
    /// Shape of the first input tensor.
    a_shape: Vec<usize>,
    /// Shape of the second input tensor.
    b_shape: Vec<usize>,
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
        let mut grads = Vec::new();

        // Gradient pour a: dL/da = dL/dz * (1)
        if self.a_requires_grad {
            let grad_a = grad_output.reduce_to_shape(&self.a_shape)?;
            grads.push(grad_a);
        }

        // Gradient pour b: dL/db = dL/dz * (-1)
        if self.b_requires_grad {
            let grad_b_unreduced = neg_op(grad_output)?;
            let grad_b = grad_b_unreduced.reduce_to_shape(&self.b_shape)?;
            grads.push(grad_b);
        }

        Ok(grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` and `b` of the forward `sub_op`.
    fn inputs(&self) -> Vec<NodeId> {
        let mut ids = Vec::new();
        if self.a_requires_grad {
            // Use if let for safety, though unwrap() should be fine based on helper logic
            if let Some(ref node) = self.a_node {
                 ids.push(Arc::as_ptr(node));
            } else {
                 // This path indicates a logic error either here or in the helper
                 eprintln!("Error: a_node is None in SubBackward::inputs despite a_requires_grad being true");
                 // Consider returning an error or panicking depending on desired behavior
            }
        }
        if self.b_requires_grad {
             if let Some(ref node) = self.b_node {
                 ids.push(Arc::as_ptr(node));
            } else {
                 eprintln!("Error: b_node is None in SubBackward::inputs despite b_requires_grad being true");
            }
        }
        ids
    }
}

// --- Forward Operation ---

/// Performs element-wise subtraction (`a - b`) on two tensors with broadcasting.
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
pub(crate) fn sub_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    crate::ops::arithmetic::apply_binary_op_broadcasted(
        a,
        b,
        |va, vb| sub_kernel::<f32>(va, vb),
        |va, vb| sub_kernel::<f64>(va, vb),
        |va, vb| va - vb, // I32
        |va, vb| va - vb, // I64
        |a_node_opt, b_node_opt, a_shape, b_shape, a_req, b_req| {
            Arc::new(SubBackward {
                a_node: a_node_opt,
                b_node: b_node_opt,
                a_shape,
                b_shape,
                a_requires_grad: a_req,
                b_requires_grad: b_req,
            })
        },
        "sub_op",
    )
}

// --- Tests --- 
// Link the external test file
#[cfg(test)]
#[path = "sub_test.rs"]
mod tests;
