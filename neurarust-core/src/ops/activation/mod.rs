// src/ops/activation/mod.rs
// Module pour les fonctions d'activation (ReLU, Sigmoid, Tanh, etc.)

//! # Activation Functions
//!
//! This module implements various non-linear activation functions commonly used
//! in neural networks. Activation functions introduce non-linearity into the model,
//! allowing it to learn more complex patterns.
//!
//! ## Currently Implemented:
//! - [`ReLU`](relu/fn.relu_op.html): Rectified Linear Unit.
//!
//! ## Future Work:
//! - Sigmoid
//! - Tanh
//! - LeakyReLU, ELU, etc.

pub mod relu;
// pub mod sigmoid;
// pub mod tanh; 

// Re-export key functions
pub use relu::relu_op; 