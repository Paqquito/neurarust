// src/ops/reduction/mod.rs

//! # Tensor Reduction Operations
//!
//! This module implements operations that reduce the dimensions of a tensor by applying
//! an aggregation function (like sum, mean, max, min) along specified axes.
//!
//! ## Structure:
//! Each submodule (`sum`, `mean`, `max`, etc.) typically contains:
//! - An `_op` function performing the forward reduction and autograd setup.
//! - A `Backward` struct implementing the `BackwardOp` trait for gradient calculation.
//!
//! ## Key Functions (Internal/Crate-Visible):
//! - [`sum_op`](sum/fn.sum_op.html): Computes the sum of elements along specified axes.
//! - `mean_op`: Computes the mean of elements along specified axes.
//! - `max_op`: Computes the maximum of elements along specified axes.
//!
//! ## Usage:
//! These `_op` functions are usually called internally by methods on the `Tensor` struct
//! (e.g., [`tensor.sum()`](../../tensor/struct.Tensor.html#method.sum), [`tensor.mean()`](../../tensor/struct.Tensor.html#method.mean), etc.), which provide a more user-friendly interface.

pub mod sum;
pub mod mean;
pub mod max;
pub mod utils;
pub mod all;
pub mod any;
pub mod bincount;

// Re-export the adapted reduction operations using pub(crate)
pub(crate) use sum::sum_op;
// pub(crate) use mean::mean_op;
// pub(crate) use max::max_op;

// Remove old/incorrect exports
// pub use sum::sum_axes;
// pub use mean::mean_axes;

#[cfg(test)]
mod all_test;
#[cfg(test)]
mod any_test;
#[cfg(test)]
mod bincount_test;

pub use bincount::bincount_op;
