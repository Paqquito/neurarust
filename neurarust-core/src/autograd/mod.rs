#![allow(clippy::module_inception)] // Allow module name = struct name

// Declare the modules within the autograd directory
pub mod backward_op;
pub mod graph;

// Declare the new module
pub mod grad_check;

// Re-export the core BackwardOp trait for easier access
pub use backward_op::BackwardOp;
