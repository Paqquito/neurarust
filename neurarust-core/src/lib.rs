//!
//! # NeuraRust Core Library (`neurarust-core`)
//! 
//! This crate provides the foundational building blocks for the NeuraRust deep learning framework.
//! It focuses on tensor operations, automatic differentiation (autograd), and basic numerical utilities.
//! 
//! ## Key Modules:
//! - `tensor`: Defines the multi-dimensional `Tensor` struct and its core functionalities.
//! - `tensor_data`: Internal representation of tensor metadata and buffer management.
//! - `buffer`: Abstract representation of data buffers (CPU/GPU).
//! - `ops`: Contains various tensor operations (arithmetic, linear algebra, views, etc.).
//! - `autograd`: Implements the automatic differentiation engine.
//! - `types`: Defines core data types (`DType`) and devices (`StorageDevice`).
//! - `error`: Defines the custom error type `NeuraRustError` used throughout the crate.
//! - `device`: Handles device abstractions (currently CPU).
//! - `utils`: General utility functions.
//! 
//! The primary goal is to offer a flexible and performant tensor library inspired by PyTorch,
//! built entirely in Rust.

// Déclare les modules principaux de la crate
pub mod autograd;
// pub mod creation;
// pub mod indexing; // REMOVED: Declared within ops module now
pub mod ops;
pub mod tensor;
pub mod tensor_data;
// Add new modules for buffer management and device abstraction
pub mod buffer;
pub mod device;

// Declare new top-level modules
pub mod nn;
// pub mod optim; // REMOVED: Optimizers moved to neurarust-optim crate
pub mod utils;

// Declare new sub-modules within ops
// (ops/mod.rs needs to declare them too)
// pub mod activation; // Declaration should be in ops/mod.rs
// pub mod reduction; // Declaration should be in ops/mod.rs
// pub mod loss; // Declaration should be in ops/mod.rs

// Declare the new types module
pub mod types;

// Ré-exporte le type Tensor pour qu'il soit accessible directement via `neurarust_core::Tensor`
pub use tensor::Tensor;
// Re-export traits required by public functions/structs
pub use num_traits;

// Le reste du code (structs, impls, anciens tests) a été déplacé
// dans les modules correspondants.

pub mod error;
pub use error::NeuraRustError;

// Re-export key components for easier use
pub use autograd::BackwardOp;
pub use buffer::Buffer;
pub use device::StorageDevice;
pub use types::DType;

// Le module de test temporaire a été supprimé ici
