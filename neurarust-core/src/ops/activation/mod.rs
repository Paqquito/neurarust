// src/ops/activation/mod.rs
// Module pour les fonctions d'activation (ReLU, Sigmoid, Tanh, etc.)

//! Activation functions like ReLU, Sigmoid, Tanh, etc.

pub mod relu;
// pub mod sigmoid;
// pub mod tanh; 

// Re-export key functions
pub use relu::relu_op; 