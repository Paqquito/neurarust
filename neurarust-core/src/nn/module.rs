use crate::tensor::Tensor;
use std::fmt::Debug;

/// The base trait for all neural network modules (layers, containers, etc.).
pub trait Module<T>: Debug {
    /// Performs the forward pass of the module.
    fn forward(&self, input: &Tensor<T>) -> Tensor<T>;

    /// Returns a list of all learnable parameters within the module.
    /// Typically returns clones of the underlying Tensors wrapped in Parameter.
    /// Note: We return Tensor<T> directly for now, as optimizers will likely operate on Tensors.
    /// If Parameter needs specific handling later, this could return Vec<Parameter<T>>.
    fn parameters(&self) -> Vec<Tensor<T>>;
}

// Example of how a simple container might implement it (not needed yet)
/*
use std::collections::BTreeMap;

pub struct Sequential<T> {
    modules: BTreeMap<String, Box<dyn Module<T>>>,
}

impl<T: 'static> Module<T> for Sequential<T> {
    fn forward(&self, input: &Tensor<T>) -> Tensor<T> {
        let mut current_input = input.clone(); // Start with the initial input
        for (_name, module) in &self.modules {
            current_input = module.forward(&current_input);
        }
        current_input
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        let mut params = Vec::new();
        for (_name, module) in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
}
*/ 