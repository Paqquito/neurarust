use crate::tensor::Tensor;
use std::fmt::Debug;
use crate::error::NeuraRustError;
use crate::nn::Parameter;

/// The base trait for all neural network modules (layers, containers, etc.).
pub trait Module<T>: Debug {
    /// Performs the forward pass of the module.
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>;

    /// Returns a list of all learnable parameters within the module.
    /// Should return owned Parameters for optimizer updates.
    fn parameters(&self) -> Vec<Parameter<T>>;
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