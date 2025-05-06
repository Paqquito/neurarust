use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};
use std::fmt::Debug;

/// A wrapper around a Tensor that indicates it is a trainable parameter.
/// Stores the tensor itself and potentially metadata in the future.
#[derive(Debug, Clone)]
pub struct Parameter<T>(pub Tensor<T>);

impl<T> Parameter<T> {
    /// Creates a new parameter wrapping the given tensor.
    pub fn new(tensor: Tensor<T>) -> Self 
    where T: Debug
    {
        tensor.set_requires_grad(true); // Ensure gradients are tracked
        Parameter(tensor)
    }

    /// Consumes the Parameter and returns the underlying Tensor.
    pub fn into_inner(self) -> Tensor<T> {
        self.0
    }
}

// Allow accessing the underlying Tensor immutably via Deref.
impl<T> Deref for Parameter<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Allow accessing the underlying Tensor mutably via DerefMut.
impl<T> DerefMut for Parameter<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[path = "parameter_test.rs"]
mod tests; 