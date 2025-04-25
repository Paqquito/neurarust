use crate::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

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

// Placeholder for a concrete operation context (example)
// We might move concrete ops to their own files later (e.g., autograd/add.rs)
struct AddBackward<T> {
    // We need references back to the input tensors' gradients.
    // Weak references are needed to break potential reference cycles.
    // These should point to the Rc<RefCell<TensorData<T>>> of the inputs.
    // Let's assume we store Weak refs to the RefCells containing Option<Tensor<T>> for gradients.
    // input_a_grad: Weak<RefCell<Option<Tensor<T>>>>, // This needs refinement
    // input_b_grad: Weak<RefCell<Option<Tensor<T>>>>,

    // Placeholder fields - we'll refine what needs storing.
    _phantom: std::marker::PhantomData<T>,
}

impl<T> BackwardOp<T> for AddBackward<T> {
     fn backward(&self, upstream_grad: &Tensor<T>) {
         println!("AddBackward: backward called (gradient accumulation pending)");
         // 1. Upgrade Weak refs to Rc.
         // 2. Check if upgrade successful.
         // 3. Borrow the gradient Option<Tensor<T>> mutably.
         // 4. If gradient exists, add upstream_grad to it.
         // 5. If gradient doesn't exist, create it by cloning upstream_grad.
         // Gradient of Add is 1, so dA = dC * 1, dB = dC * 1.
         // We just need to pass/accumulate the upstream_grad.
     }
} 