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
//! - [`ne_op`](ne/fn.ne_op.html): Element-wise inequality (`!=`).
//! - [`gt_op`](gt/fn.gt_op.html): Element-wise greater than (`>`).
//! - [`lt_op`](lt/fn.lt_op.html): Element-wise less than (`<`).
//! - [`ge_op`](ge/fn.ge_op.html): Element-wise greater than or equal (`>=`).
//! - [`le_op`](le/fn.le_op.html): Element-wise less than or equal (`<=`).
//! - [`logical_and_op`](logical_and/fn.logical_and_op.html): Element-wise logical AND.
//! - [`logical_or_op`](logical_or/fn.logical_or_op.html): Element-wise logical OR.
//! - [`logical_xor_op`](logical_xor/fn.logical_xor_op.html): Element-wise logical XOR.
//! - [`logical_not_op`](logical_not/fn.logical_not_op.html): Element-wise logical NOT.

// Declare comparison operations
pub mod equal;
pub mod ge;
pub mod gt;
pub mod le;
pub mod logical_and;
pub mod logical_not;
pub mod logical_or;
pub mod logical_xor;
pub mod lt;
pub mod ne;

// Re-export the operation functions
pub use equal::equal_op;
pub use ge::ge_op;
pub use gt::gt_op;
pub use le::le_op;
pub use logical_and::logical_and_op;
pub use logical_not::logical_not_op;
pub use logical_or::logical_or_op;
pub use logical_xor::logical_xor_op;
pub use lt::lt_op;
pub use ne::ne_op;

#[cfg(test)]
mod gt_test;
#[cfg(test)]
mod le_test;
#[cfg(test)]
mod lt_test; 