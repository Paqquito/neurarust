//! # Tensor Operations (`ops`)
//!
//! This module serves as the central hub for all tensor operations within NeuraRust.
//! It organizes various mathematical, logical, neural network, and manipulation
//! operations into logical submodules.
//!
//! ## Structure:
//! Each submodule typically contains:
//! - **Forward Operation Functions:** Public functions (often ending in `_op`, e.g., `add_op`)
//!   that perform the core computation and handle autograd graph construction if necessary.
//! - **Backward Structures:** Implementations of the [`BackwardOp`](../../autograd/trait.BackwardOp.html)
//!   trait, defining how gradients are calculated for each operation.
//! - **Test Modules:** Unit and integration tests, including gradient checks (`check_grad`),
//!   to ensure the correctness of both forward and backward passes.
//!
//! ## Submodules:
//! - [`activation`](activation/index.html): Non-linear activation functions (e.g., ReLU).
//! - [`arithmetic`](arithmetic/index.html): Element-wise arithmetic operations (add, sub, mul, div, neg, etc.).
//! - [`comparison`](comparison/index.html): Element-wise comparison operations (eq, ne, gt, lt, etc.). (WIP)
//! - [`linalg`](linalg/index.html): Linear algebra operations (e.g., matrix multiplication).
//! - [`loss`](loss/index.html): Loss functions used for training models. (WIP)
//! - [`math_elem`](math_elem/index.html): Element-wise mathematical functions (exp, log, pow, sqrt, etc.).
//! - [`reduction`](reduction/index.html): Operations that reduce tensor dimensions (sum, mean, max, min).
//! - [`view`](view/index.html): Operations that change the tensor's shape/strides without copying data (reshape, transpose, slice, permute, expand).
//!
//! ## Usage:
//! While operations are defined here, they are often more conveniently accessed via methods
//! directly implemented on the [`Tensor`](../tensor/struct.Tensor.html) struct (e.g., `tensor.add(other)` might call `ops::arithmetic::add_op(tensor, other)`).
//! However, the `_op` functions can be called directly if needed.

// Déclare les sous-modules d'opérations
pub mod activation;
pub mod arithmetic;
pub mod comparison;
pub mod linalg;
pub mod loss;
pub mod math_elem;
pub mod reduction;
pub mod view;
// pub mod activation; // REMOVED
// pub mod indexing; // REMOVED
// pub mod stack; // REMOVED
// pub mod reshape; // REMOVED

// Declare new operation categories
// pub mod activation; // REMOVED
// pub mod loss; // REMOVED (part of nn, not ops base)
// pub mod indexing; // REMOVED
// pub mod stack; // REMOVED
// pub mod math_elem; // REMOVED

// Individual ops that might not fit cleanly into categories yet
// pub mod reshape; // REMOVED

// Potentially re-export specific operations if needed
// pub use arithmetic::add::add;

// Re-export core operations for easier access (Phase 0)
pub use arithmetic::{add_op, div_op, mul_op, neg_op, sub_op}; // Keep arithmetic ops
                                                              // pub use arithmetic::pow_op; // Pow might be Phase 1+
                                                              // Comment out problematic exports
                                                              // pub use linalg::{matmul, transpose}; // Keep existing linalg exports
// Replace sum_axes with sum_op
// pub use reduction::sum_op; // Export sum_op instead of sum_axes - Commented out as sum_op is pub(crate)

// Remove problematic exports causing linter errors
// pub use math_elem::{exp_op, ln_op, pow_op};
// pub use reduction::{mean_op, sum_axes_op, sum_op};
// pub use linalg::matmul_op;
// pub use activation::relu_op;

// Potentially export view ops later if needed publicly
// pub use view_ops::{slice_op, ...};
