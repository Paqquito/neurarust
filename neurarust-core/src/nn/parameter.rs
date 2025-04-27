use crate::tensor::Tensor;
use std::fmt;
use std::ops::Deref;

/// A wrapper around a Tensor indicating it is a learnable parameter of a Module.
/// Parameters automatically have `requires_grad` set to `true`.
pub struct Parameter<T>(Tensor<T>);

impl<T> Parameter<T> {
    /// Creates a new Parameter from a Tensor.
    /// Ensures that the underlying Tensor requires gradients.
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

// Optional: Implement Debug, Clone if needed
impl<T: fmt::Debug> fmt::Debug for Parameter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate formatting to the inner Tensor, perhaps with a Parameter prefix
        write!(f, "Parameter({:?})", self.0)
    }
}

impl<T> Clone for Parameter<T> {
    /// Cloning a Parameter clones the underlying Tensor (shallow clone via Rc).
    fn clone(&self) -> Self {
        Parameter(self.0.clone())
    }
} 