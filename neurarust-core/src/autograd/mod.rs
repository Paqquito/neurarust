#![allow(clippy::module_inception)] // Allow module name = struct name

//! # Automatic Differentiation (`autograd`)
//!
//! This module provides the core components for automatic differentiation in NeuraRust,
//! enabling the computation of gradients for tensor operations.
//!
//! ## Key Concepts:
//! - **Computation Graph:** Tensor operations are implicitly recorded, forming a directed acyclic graph (DAG)
//!   where nodes represent tensors and edges represent the functions (`BackwardOp`) that produced them.
//! - **`BackwardOp` Trait:** Defines the interface for operations to compute their gradients with respect
//!   to their inputs, given the gradient of their output (chain rule).
//! - **Topological Sort:** The graph is traversed in reverse topological order during the `backward()` pass
//!   to ensure gradients are computed correctly.
//! - **Gradient Checking:** Utilities (`grad_check`) are provided to numerically verify the correctness
//!   of analytical gradients computed by `BackwardOp` implementations.

// Declare the modules within the autograd directory
pub mod backward_op;
pub mod graph;

// Declare the new module
pub mod grad_check;

// Re-export the core BackwardOp trait for easier access
pub use backward_op::BackwardOp;
