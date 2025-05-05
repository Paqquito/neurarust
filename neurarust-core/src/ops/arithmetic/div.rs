use crate::{
    autograd::{backward_op::BackwardOp, graph::NodeId},
    error::NeuraRustError,
    ops::traits::NeuraNumeric,
    tensor::Tensor,
    tensor_data::TensorData,
};
use crate::ops::arithmetic::mul::mul_op;
use crate::ops::arithmetic::neg::neg_op;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Generic kernel for element-wise division.
fn div_kernel<T: NeuraNumeric>(a: T, b: T) -> T {
    // Note: Consider adding checks for division by zero if T can be integer
    // or if robust handling is needed for floats.
    // For Float types, this typically results in +/- infinity or NaN.
    a / b
}

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise division operation.
/// Stores references to input tensor nodes, clones of input tensors needed for gradient calculation,
/// original shapes, and flags indicating if gradients were required.
#[derive(Debug)]
struct DivBackward {
    /// Clone of the numerator tensor (`a`).
    a: Tensor,
    /// Clone of the denominator tensor (`b`).
    b: Tensor,
    /// Optional reference counted pointer to the numerator tensor's data (`a`).
    a_node: Option<Arc<RwLock<TensorData>>>,
    /// Optional reference counted pointer to the denominator tensor's data (`b`).
    b_node: Option<Arc<RwLock<TensorData>>>,
    /// Original shape of the numerator tensor (`a`).
    a_shape: Vec<usize>,
    /// Original shape of the denominator tensor (`b`).
    b_shape: Vec<usize>,
    /// Flag indicating if the numerator tensor (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the denominator tensor (`b`) required gradients.
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for DivBackward {
    /// Computes gradients for the division operation \( z = a / b \).
    /// \( \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{\partial z}{\partial a} = \frac{dL}{dz} \cdot \frac{1}{b} \)
    /// \( \frac{dL}{db} = \frac{dL}{dz} \cdot \frac{\partial z}{\partial b} = \frac{dL}{dz} \cdot \left(-\frac{a}{b^2}\right) \)
    /// Gradients need to be reduced to the original input shapes if broadcasting occurred.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut result_grads = Vec::new();

        // Gradient for a: dL/da = dL/dz * (1/b)
        if self.a_requires_grad {
            let grad_a_unreduced = div_op(grad_output, &self.b)?;
            let grad_a = grad_a_unreduced.reduce_to_shape(&self.a_shape)?;
            result_grads.push(grad_a);
        }

        // Gradient for b: dL/db = dL/dz * (-a / b^2)
        if self.b_requires_grad {
            let b_squared = mul_op(&self.b, &self.b)?;
            let a_div_b_squared = div_op(&self.a, &b_squared)?;
            let neg_a_div_b_squared = neg_op(&a_div_b_squared)?;
            let grad_b_unreduced = mul_op(grad_output, &neg_a_div_b_squared)?;
            let grad_b = grad_b_unreduced.reduce_to_shape(&self.b_shape)?;
            result_grads.push(grad_b);
        }

        Ok(result_grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` (numerator) and `b` (denominator).
    fn inputs(&self) -> Vec<NodeId> {
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node.as_ref().unwrap())); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node.as_ref().unwrap())); }
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
    let a_clone = a.clone();
    let b_clone = b.clone();

    // Call the centralized helper
    crate::ops::arithmetic::apply_binary_op_broadcasted(
        a,
        b,
        // Closure for F32, calling the generic kernel
        |va, vb| div_kernel::<f32>(va, vb),
        // Closure for F64, calling the generic kernel
        |va, vb| div_kernel::<f64>(va, vb),
        // Closure captures and moves the clones
        move |a_node_opt, b_node_opt, a_shape, b_shape, a_req, b_req| {
            Arc::new(DivBackward {
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
        "div_op", // Operation name for errors
    )
}

// --- Tests ---
#[cfg(test)]
#[path = "div_test.rs"]
mod tests;
