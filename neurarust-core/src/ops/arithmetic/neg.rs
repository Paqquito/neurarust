use crate::autograd::backward_op::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use std::fmt::Debug;
// Add Add trait needed for potential acc_grad, Send/Sync for BackwardOp
use std::sync::{Arc, RwLock};
use crate::autograd::graph::NodeId;

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise negation operation.
///
/// Stores a reference to the input tensor node for graph linkage.
#[derive(Debug)]
struct NegBackward {
    /// Reference counted pointer to the input tensor's data for graph linkage.
    a_node: Option<Arc<RwLock<TensorData>>>,
}

// --- Backward Operation Implementation ---

impl BackwardOp for NegBackward {
    /// Computes the gradient for the negation operation \( z = -a \).
    ///
    /// Using the chain rule \( \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{dz}{da} \),
    /// where \( \frac{dz}{da} = -1 \), the gradient is:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \cdot (-1) = - \frac{dL}{dz} \\]
    ///
    /// This method simply negates the incoming gradient (`grad_output`).
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // dL/da = dL/dz * dz/da = grad_output * (-1)
        Ok(vec![neg_op(grad_output)?])
    }

    /// Returns the identifier of the input tensor node.
    fn inputs(&self) -> Vec<NodeId> {
        match &self.a_node {
            Some(node) => vec![Arc::as_ptr(node)],
            None => vec![], // Should not happen if requires_grad was true
        }
    }
}

// --- Forward Operation ---

/// Performs element-wise negation (`-input`) on a tensor.
///
/// Computes the negative of each element in the input tensor.
/// Supports `DType::F32` and `DType::F64` tensors on the CPU.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `input`: The input `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` with the negated values, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` if:
/// - The tensor is not on the CPU (`DeviceMismatch`).
/// - The tensor's `DType` is not F32 or F64 (`UnsupportedOperation`).
/// - An internal error occurs.
pub fn neg_op(a: &Tensor) -> Result<Tensor, NeuraRustError> {
    // Appelle le helper unaire de ops/mod.rs
    crate::ops::apply_unary_op(
        a,
        |x| -x, // Opération F32
        |x| -x, // Opération F64
        // Closure pour construire NegBackward
        |a_node_opt| Arc::new(NegBackward { a_node: a_node_opt }),
        "neg_op", // Nom de l'opération
    )
}

// --- std::ops::Neg implementation ---
// Implement the Neg trait for Tensor by calling neg_op
/* // Remove the generic implementation for now
impl<T> Neg for &Tensor<T>
where
    // Bounds must match neg_op requirements
    T: Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    type Output = Result<Tensor<T>, NeuraRustError>;

    fn neg(self) -> Self::Output {
        neg_op(self)
    }
}
*/

// --- Tests ---
// Link the external test file
#[cfg(test)]
#[path = "neg_test.rs"]
mod tests;
