// neurarust-core/src/ops/math_elem/ln.rs

use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
// Add imports for backward
use crate::ops::arithmetic::div_op;

// --- LnBackward Definition ---

/// Backward pass structure for the element-wise natural logarithm (`ln`) operation.
///
/// Stores a reference to the original input tensor node, as the input value is needed
/// to compute the gradient (1 / input).
#[derive(Debug)]
struct LnBackward {
    /// Reference counted pointer to the input tensor's data.
    a_node: Option<Arc<RwLock<TensorData>>>,
    a_shape: Vec<usize>,
}

// --- BackwardOp Implementation for LnBackward ---

impl BackwardOp for LnBackward {
    /// Computes the gradient for the natural logarithm operation \( z = \ln(a) \).
    ///
    /// Using the chain rule \( \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{dz}{da} \),
    /// where \( \frac{dz}{da} = \frac{1}{a} \), the gradient is:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{1}{a} \\]
    ///
    /// This method computes \( \frac{grad\_output}{input} \).
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let a_tensor = match &self.a_node {
            Some(node) => Tensor { data: node.clone() },
            None => return Err(NeuraRustError::InternalError("Missing input node in LnBackward".to_string())),
        };
        let grad_a_unreduced = div_op(grad_output, &a_tensor)?;
        let grad_a = grad_a_unreduced.reduce_to_shape(&self.a_shape)?;
        Ok(vec![grad_a])
    }

    /// Returns the identifier of the input tensor node.
    fn inputs(&self) -> Vec<NodeId> {
        match &self.a_node {
            Some(node) => vec![Arc::as_ptr(node)],
            None => vec![],
        }
    }
}

// --- ln_op Implementation (Public API + Autograd Setup) ---

/// Computes the element-wise natural logarithm (base \( e \)) of a tensor.
///
/// Calculates \( \ln(x) \) for each element \( x \) in the input tensor.
///
/// This operation supports automatic differentiation.
///
/// # Arguments
/// * `tensor`: The input `Tensor`.
///
/// # Returns
/// A `Result` containing a new `Tensor` with the natural logarithm applied, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError::UnsupportedOperation` if the input tensor is not a CPU tensor with `DType::F32`.
///
/// # Domain Considerations
/// The natural logarithm is only defined for strictly positive numbers.
/// This implementation currently returns `f32::NAN` for non-positive inputs.
/// The gradient \( 1/x \) is also undefined at \( x=0 \).
pub fn ln_op(a: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_shape = a.shape(); // Save shape before moving 'a' to the helper

    crate::ops::apply_unary_op(
        a,
        |x| x.ln(), // F32 op (returns NaN for x <= 0)
        |x| x.ln(), // F64 op
        // Closure to build LnBackward, capturing the shape
        move |a_node_opt| {
            Arc::new(LnBackward {
                a_node: a_node_opt,
                a_shape: a_shape.clone(), // Use captured shape
            })
        },
        "ln_op", // Operation name
    )
}

// --- Tests ---
#[cfg(test)]
#[path = "ln_test.rs"]
mod tests; // Link to the test file 