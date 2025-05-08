use crate::tensor::Tensor;
use std::collections::HashMap;

/// Represents the state of an optimizer.
///
/// This enum will be extended to hold specific state information
/// for different types of optimizers (e.g., momentum buffers for SGD,
/// first and second moment estimates for Adam).
/// The specific states might include Tensors, so `Tensor` might need to be in scope
/// when variants are fully defined. For now, it's kept simple.
#[derive(Debug, Clone)]
pub enum OptimizerState {
    /// State specific to the SGD optimizer.
    Sgd {
        /// Momentum buffers associated with parameters.
        /// The key could be a unique identifier for the parameter (e.g., its memory address or a generated ID).
        /// Using `usize` as a placeholder key type for now.
        momentum_buffers: HashMap<usize, Tensor>,
    },
    /// State specific to the Adam optimizer (Placeholder).
    Adam { 
        // m: HashMap<usize, Tensor>,
        // v: HashMap<usize, Tensor>,
        // step: usize 
    },
    /// A generic placeholder state for optimizers without specific state yet
    /// or for initialization.
    Placeholder,
}

impl Default for OptimizerState {
    fn default() -> Self {
        OptimizerState::Placeholder
    }
} 