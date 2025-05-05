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
