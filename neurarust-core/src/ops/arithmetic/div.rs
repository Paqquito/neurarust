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

/// Performs element-wise division between two tensors, supporting broadcasting and autograd.
///
/// # Supported Types
/// - `DType::F32`
/// - `DType::F64`
/// - `DType::I32`
/// - `DType::I64`
///
/// # Unsupported Types
/// - `DType::Bool` (returns an `UnsupportedOperation` error)
///
/// # Arguments
/// * `a` - Numerator (first input tensor).
/// * `b` - Denominator (second input tensor).
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise division, or a `NeuraRustError`.
///
/// # Errors
/// - `DeviceMismatch` if tensors are not on the CPU.
/// - `DataTypeMismatch` if the DTypes do not match.
/// - `BroadcastError` if the shapes are not broadcast-compatible.
/// - `DivisionByZero` or `ArithmeticError` for integer division by zero.
/// - `UnsupportedOperation` if the DType is not supported.
/// - `InternalError` for internal errors.
///
/// # Example
/// ```
/// use neurarust_core::{Tensor, DType};
/// use neurarust_core::ops::arithmetic::div_op;
/// let a = Tensor::new_i64(vec![10, 20, 30], vec![3]).unwrap();
/// let b = Tensor::new_i64(vec![2, 4, 5], vec![3]).unwrap();
/// let c = div_op(&a, &b).unwrap();
/// assert_eq!(c.get_i64_data().unwrap(), vec![5, 5, 6]);
/// assert_eq!(c.dtype(), DType::I64);
/// ```
pub fn div_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_clone = a.clone();
    let b_clone = b.clone();

    crate::ops::arithmetic::apply_binary_op_broadcasted(
        a,
        b,
        // Closure pour F32
        |va, vb| div_kernel::<f32>(va, vb),
        // Closure pour F64
        |va, vb| div_kernel::<f64>(va, vb),
        // I32 : closure simple, la gestion d'erreur sera dans apply_binary_op_broadcasted
        |va, vb| va / vb,
        // I64 : closure simple, la gestion d'erreur sera dans apply_binary_op_broadcasted
        |va, vb| va / vb,
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
        "div_op",
    )
}

// --- Tests ---
#[cfg(test)]
#[path = "div_test.rs"]
mod tests;
