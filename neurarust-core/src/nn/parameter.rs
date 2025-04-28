use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};
use std::fmt::{self, Debug};

/// A wrapper around a Tensor indicating it is a learnable parameter of a Module.
/// Parameters automatically have `requires_grad` set to `true`.
#[derive(Clone)]
pub struct Parameter<T>(pub(crate) Tensor<T>);

impl<T: Debug> Parameter<T> {
    /// Creates a new parameter wrapping the given tensor.
    pub fn new(tensor: Tensor<T>) -> Self {
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

// Optional: Implement Debug, Clone if needed
impl<T: fmt::Debug> fmt::Debug for Parameter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate formatting to the inner Tensor, perhaps with a Parameter prefix
        write!(f, "Parameter({:?})", self.0)
    }
} 