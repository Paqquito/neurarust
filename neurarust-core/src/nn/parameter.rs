use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};
use std::fmt::Debug;

/// A wrapper around a Tensor that indicates it is a trainable parameter.
/// Stores the tensor itself and potentially metadata in the future.
#[derive(Debug, Clone)]
pub struct Parameter(pub Tensor);

impl Parameter {
    /// Creates a new parameter wrapping the given tensor.
    pub fn new(tensor: Tensor) -> Self {
        // TODO: Devrait probablement vérifier que le tenseur n'a pas déjà un grad_fn
        // ou qu'il est bien une "feuille" du graphe.
        let _ = tensor.set_requires_grad(true); // Ensure gradients are tracked
        Parameter(tensor)
    }

    /// Consumes the Parameter and returns the underlying Tensor.
    pub fn into_inner(self) -> Tensor {
        self.0
    }
}

// Allow accessing the underlying Tensor immutably via Deref.
impl Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Allow accessing the underlying Tensor mutably via DerefMut.
impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[path = "parameter_test.rs"]
mod tests; 