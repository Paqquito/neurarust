/// Represents the state of an optimizer.
///
/// This enum will be extended to hold specific state information
/// for different types of optimizers (e.g., momentum buffers for SGD,
/// first and second moment estimates for Adam).
/// The specific states might include Tensors, so `Tensor` might need to be in scope
/// when variants are fully defined. For now, it's kept simple.
#[derive(Debug, Clone)]
pub enum OptimizerState {
    // Variants for specific optimizer states will be added here.
    // For example:
    // Sgd { momentum_buffers: std::collections::HashMap<String, crate::tensor::Tensor> },
    // Adam {
    //     m: std::collections::HashMap<String, crate::tensor::Tensor>,
    //     v: std::collections::HashMap<String, crate::tensor::Tensor>,
    //     step: usize
    // },
    /// A generic placeholder state, to be replaced or augmented
    /// by specific optimizer state variants.
    _Placeholder,
} 