use crate::Tensor;

/// Trait representing an operation that can perform backpropagation.
/// Each operation (Add, Mul, Matmul, etc.) will have a corresponding struct
/// implementing this trait, storing the necessary context (inputs, shapes, etc.).
pub trait BackwardOp<T> {
    /// Performs the backward pass for this operation.
    ///
    /// Takes the gradient flowing from the *output* of this operation (`upstream_grad`)
    /// and computes/accumulates the gradients with respect to the *inputs* of this operation.
    fn backward(&self, upstream_grad: &Tensor<T>);
}

// Concrete BackwardOp implementations will live alongside their corresponding
// forward operations (e.g., in neurarust-core/src/ops/arithmetic.rs).
// Remove the placeholder AddBackward struct from here. 