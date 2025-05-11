// neurarust-core/src/ops/view/mod.rs

//! # Tensor View Operations
//!
//! This module provides operations that create new `Tensor` views without copying the
//! underlying data. These operations manipulate the tensor's metadata (shape, strides, offset)
//! to present a different perspective on the same data buffer.
//!
//! View operations are crucial for efficiency, especially in deep learning, as they avoid
//! unnecessary memory allocations and copies.
//!
//! ## Key Operations:
//! - **[`slice_op`](slice/fn.slice_op.html)**: Extracts a sub-tensor (slice).
//! - **[`transpose_op`](transpose/fn.transpose_op.html)**: Swaps two dimensions.
//! - **[`permute_op`](permute/fn.permute_op.html)**: Rearranges dimensions according to a given permutation.
//! - **[`reshape_op`](reshape/fn.reshape_op.html)**: Changes the shape of the tensor while preserving the number of elements.
//! - **[`expand_op`](fn.expand_op.html)**: Broadcasts singleton dimensions (size 1) to a larger size.
//!
//! ## Autograd Integration:
//! Each view operation (`_op` function) typically has a corresponding `Backward` struct
//! (e.g., `SliceBackward`, `TransposeBackward`) that implements the [`BackwardOp`](../../autograd/trait.BackwardOp.html)
//! trait. These structures store the necessary context (like original shapes or axes)
//! to correctly propagate gradients back through the view operation during the backward pass.
//!
//! For example, the backward pass of a `reshape` operation might involve reshaping the incoming
//! gradient back to the input tensor's original shape. The backward of `expand` requires summing
//! the gradient along the dimensions that were expanded.
//!
//! ## Usage:
//! These `_op` functions are usually called internally by methods on the `Tensor` struct
//! (e.g., [`tensor.slice()`](../../tensor/struct.Tensor.html#method.slice), [`tensor.transpose()`](../../tensor/struct.Tensor.html#method.transpose), etc.),
//! which provide a more user-friendly interface.

pub mod contiguous;
pub mod expand;
pub mod permute;
pub mod reshape;
pub mod slice;
pub mod squeeze_unsqueeze;
pub mod transpose;
pub mod utils;
pub mod index_select;
pub mod masked_select;

// Re-exports for easier access
pub use expand::expand_op;
pub use permute::permute_op;
pub use reshape::reshape_op;
pub use slice::SliceArg; // slice_op and SliceRange are kept crate-public for now
pub use squeeze_unsqueeze::{unsqueeze_op, squeeze_op};
pub use transpose::transpose_op;
pub use masked_select::masked_select_op;
pub use index_select::index_select_op;
// Note: contiguous_op is part of Tensor::contiguous directly, not a separate op here.

// The rest of the file used to contain definition for ExpandBackward etc.
// These should remain in their respective files (e.g. expand.rs)
// This file should primarily be for module declarations and re-exports.

// Example of what might have been here before refactoring into submodules:
// use crate::autograd::BackwardOp;
// use crate::autograd::graph::NodeId;
// use crate::error::NeuraRustError;
// use crate::tensor::Tensor;
// use crate::tensor_data::TensorData;
// use std::sync::{Arc, RwLock};
// use std::fmt::Debug;

