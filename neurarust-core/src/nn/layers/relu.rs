use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::ops::activation::relu_op; // Importer relu_op
use std::sync::{Arc, RwLock};

/// Layer that applies the Rectified Linear Unit (ReLU) activation function.
///
/// This layer does not have any learnable parameters.
#[derive(Debug, Default, Clone)]
pub struct ReLU {}

impl ReLU {
    /// Creates a new ReLU layer.
    pub fn new() -> Self {
        ReLU {}
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        relu_op(input)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        Vec::new() // ReLU has no parameters
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
        Vec::new() // ReLU has no named parameters
    }

    fn modules(&self) -> Vec<&dyn Module> {
        vec![self]
    }

    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
        f(self);
    }

    // Optional: Implement other Module methods if needed, like children(), modules(), apply()
    // For a simple layer like ReLU, the defaults or empty implementations are often sufficient.
} 