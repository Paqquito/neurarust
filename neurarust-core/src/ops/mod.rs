//! # Tensor Operations Module (`ops`)
//!
//! This module serves as the central hub for defining and organizing various tensor operations
//! within NeuraRust. Operations are categorized into submodules based on their functionality.
//!
//! ## Structure:
//!
//! - **Submodules:** Operations are grouped logically (e.g., `arithmetic`, `linalg`, `reduction`, `view`).
//! - **`_op` Functions:** Each operation typically has a core function (often named `xxx_op`)
//!   that performs the forward computation and sets up the backward pass for autograd.
//!   These functions are often marked `pub(crate)` as they are primarily intended for internal use,
//!   called by methods defined on the `Tensor` struct itself.
//! - **`Backward` Structs:** Each operation requiring gradient computation has a corresponding
//!   struct (e.g., `AddBackward`, `MatmulBackward`) that implements the
//!   [`BackwardOp`](../autograd/backward_op/trait.BackwardOp.html) trait. This struct stores the necessary
//!   context from the forward pass to compute gradients correctly during backpropagation.
//! - **Traits (`ops::traits`):** May define common traits for operations if needed (currently basic).
//!
//! ## Key Submodules:
//!
//! - [`arithmetic`]: Element-wise arithmetic operations (add, sub, mul, div, etc.).
//! - [`linalg`]: Linear algebra operations (matmul, etc.).
//! - [`nn`]: Operations commonly used in neural networks (activations, etc.).
//! - [`reduction`]: Operations that reduce tensor dimensions (sum, mean, max, etc.).
//! - [`view`]: Operations that create new views of tensors without copying data (reshape, slice, transpose, etc.).
//! - [`dtype`]: Operations related to data type conversion (cast).

// Declare operation submodules
pub mod activation; // Activation functions (formerly under nn)
pub mod arithmetic;
pub mod comparison;
pub mod linalg;
pub mod loss;       // Loss functions (currently empty)
pub mod math_elem;  // Element-wise math functions (ln, etc.)
pub mod reduction;
pub mod view;

// Re-exports: Make core operation functions easily accessible within the crate
// Using pub(crate) keeps them internal but usable by Tensor methods etc.


// Arithmetic ops are re-exported from ops/arithmetic/mod.rs
// pub(crate) use arithmetic::{add_op, div_op, mul_op, neg_op, sub_op, pow_op};


// math_elem ops are re-exported from ops/math_elem/mod.rs
// pub(crate) use math_elem::{ln_op, exp_op, sqrt_op}; // Assuming exp/sqrt exist later

 // Add min_op later if needed


// -- Removed old re-export that caused visibility errors --
// pub use arithmetic::{add_op, div_op, mul_op, neg_op, sub_op}; // Keep arithmetic ops

// Re-export the main BackwardOp trait for convenience within ops modules?
// Maybe not necessary, full path is clear.
// pub(crate) use crate::autograd::backward_op::BackwardOp;
