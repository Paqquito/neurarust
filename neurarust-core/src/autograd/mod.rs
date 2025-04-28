use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::cell::RefCell;
use std::rc::Weak;
use std::collections::HashMap;

pub mod graph;

/// Trait for operations that support backward pass (gradient calculation).
pub trait BackwardOp<T>: std::fmt::Debug {
    /// Performs the backward pass, calculating gradients for the inputs.
    /// Takes the upstream gradient and a mutable reference to the gradient map.
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>);

    /// Returns weak references to the input tensors.
    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>>;
}

// Concrete BackwardOp implementations will live alongside their corresponding
// forward operations (e.g., in neurarust-core/src/ops/arithmetic.rs).
// Remove the placeholder AddBackward struct from here. 