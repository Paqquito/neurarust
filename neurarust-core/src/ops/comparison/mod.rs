// neurarust-core/src/ops/comparison/mod.rs

//! # Element-wise Comparison Operations
//!
//! This module implements element-wise comparison operations between tensors,
//! such as equality, inequality, greater than, less than, etc.
//!
//! These operations typically return a tensor of booleans (or potentially 0s and 1s)
//! with the same shape as the (broadcasted) inputs.
//! Autograd is generally not supported or meaningful for comparison operations themselves,
//! but they can be used within larger graphs.
//!
//! ## Currently Implemented:
//! - [`equal_op`](equal/fn.equal_op.html): Element-wise equality (`==`).
//!
//! ## Future Work:
//! - `not_equal_op`, `greater_op`, `less_op`, `greater_equal_op`, `less_equal_op`.

// Declare comparison operations
pub mod equal;
// Add others like greater, less, etc. later

// Re-export the operation functions
pub use equal::equal_op; 