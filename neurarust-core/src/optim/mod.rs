// src/optim/mod.rs
// Module pour les algorithmes d'optimisation (SGD, Adam, etc.)

pub mod optimizer;
pub mod sgd;

pub use optimizer::Optimizer;
pub use sgd::SGD;

// pub mod adam; 