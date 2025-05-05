// neurarust-core/src/ops/arithmetic/add.rs

use crate::autograd::backward_op::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use std::sync::RwLock;
use std::sync::Arc;
use std::fmt::Debug;

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise addition operation.
///
/// Stores references to the input tensor nodes (`a_node`, `b_node`), their original
/// shapes (`a_shape`, `b_shape`), and flags indicating if they require gradients.
/// The original shapes are crucial for reducing the output gradient back to the
/// correct input shapes if broadcasting occurred during the forward pass.
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
    /// The gradient \( \frac{dL}{dz} \) received (`grad_output`) is passed back to both
    /// inputs `a` and `b` because the local derivatives \( \frac{dz}{da} = 1 \) and
    /// \( \frac{dz}{db} = 1 \).
    ///
    /// If broadcasting occurred, `grad_output` is reduced to the original shapes of `a` and `b`
    /// using [`reduce_gradient_to_shape`] before being returned.
    ///
    /// # Arguments
    /// * `grad_output`: The gradient tensor flowing back, corresponding to the output `z`.
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of gradient tensors corresponding to the inputs
    /// `a` and `b` (in that order) that required gradients. If an input did not require
    /// gradients, its corresponding gradient is not computed or returned.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::with_capacity(2);

        if self.a_requires_grad {
            let grad_a = grad_output.reduce_to_shape(&self.a_shape)?;
            grads.push(grad_a);
        }
        if self.b_requires_grad {
            let grad_b = grad_output.reduce_to_shape(&self.b_shape)?;
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
    // Appelle la fonction helper centralisée
    super::apply_binary_op_broadcasted(
        a,
        b,
        |va, vb| va + vb, // Opération pour F32
        |va, vb| va + vb, // Opération pour F64
        |a_node_opt, b_node_opt, a_shape, b_shape, a_req, b_req| { // Constructeur pour AddBackward
            // Vérifie que les noeuds existent si le gradient est requis
            let a_node = a_node_opt.expect("Missing a_node in add_op backward builder when grad required");
            let b_node = b_node_opt.expect("Missing b_node in add_op backward builder when grad required");
            // Crée l'Arc pour la structure Backward spécifique
            Arc::new(AddBackward {
                a_node, // Utilise les Arcs déballés
                b_node,
                a_shape,
                b_shape,
                a_requires_grad: a_req,
                b_requires_grad: b_req,
            })
        },
        "add_op", // Nom de l'opération pour les erreurs
    )
}

// Re-enable the test module link
#[cfg(test)]
#[path = "add_test.rs"]
mod tests;
