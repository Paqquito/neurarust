// neurarust-core/src/optim/mod.rs

//! Optimizers for training neural networks.
//! 
//! This module provides the `Optimizer` trait, supporting structures like
//! `ParamGroup` and `OptimizerState`, and will later include implementations 
//! of common optimization algorithms such as SGD, Adam, etc.

// Declare modules within the optim crate
pub mod optimizer_state;
pub mod param_group;
pub mod optimizer_trait;

// Re-export key items for easier access
pub use optimizer_state::OptimizerState;
pub use param_group::ParamGroup;
pub use optimizer_trait::Optimizer; 