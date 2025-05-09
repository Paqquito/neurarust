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

// Declare the new sgd module
pub mod sgd;

// Declare the new adam module
pub mod adam;
#[cfg(test)]
mod adam_test;

// Add these lines for RMSprop
pub mod rmsprop;
#[cfg(test)]
mod rmsprop_test;

// Add this line for Adagrad
pub mod adagrad;

// Re-export key items for easier access
pub use optimizer_state::OptimizerState;
pub use param_group::ParamGroup;
pub use optimizer_trait::Optimizer;

// Re-export SgdOptimizer
pub use sgd::SgdOptimizer;

// Re-export AdamOptimizer
pub use adam::AdamOptimizer;

// Add this line for RMSprop
pub use rmsprop::RmsPropOptimizer;

// Add this line for Adagrad
pub use adagrad::AdagradOptimizer;

// Declare test module conditionally
#[cfg(test)]
mod sgd_test;
#[cfg(test)]
mod adagrad_test;

// Ajout du nouveau module
pub mod lr_scheduler;

// Exportation du trait LRScheduler
pub use lr_scheduler::LRScheduler;
// Décommenter si les traits placeholder doivent être accessibles globalement pour les tests initiaux
// pub use lr_scheduler::{OptimizerInterface, ParamGroupInterface}; 

pub mod grad_clipping;
pub use grad_clipping::*; 