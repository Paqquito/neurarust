#![allow(clippy::needless_borrow)]
// src/tensor/mod.rs

use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::marker::Copy;
use std::sync::{Arc, RwLock};

// --- Define existing implementation modules ---
// These should correspond to actual files like `autograd_methods.rs`
mod autograd_methods;
mod traits; // Keep traits module declared
// Remove module declaration, implementation will be below
// mod traits;
// Add other existing modules here if they exist (e.g., based on previous structure or file listing)
mod accessors;
mod create;
mod view_methods;

// --- Declare utility modules ---
pub mod utils; // Declare public utils
pub mod broadcast_utils; // Declare the new public broadcast utils module

/// Represents a multi-dimensional array (tensor).
///
/// `Tensor` uses `Arc<RwLock<TensorData<T>>>` internally to allow for:
/// 1.  **Shared Ownership:** Multiple `Tensor` instances can point to the same
///     underlying data without cloning the data itself (cheap clones).
/// 2.  **Interior Mutability:** Metadata (like `requires_grad` or `grad`) within
///     `TensorData` can be modified even through an immutable `Tensor` reference,
///     using the `RwLock`. Read/write locks ensure thread safety.
///
/// `T` is the data type of the tensor elements (e.g., `f32`, `i64`).
/// It must be `Debug` and `Copy`. `Copy` is crucial because operations often
/// involve reading/writing individual elements. `'static` is needed due to
/// the way `Arc` and threading interact.
pub struct Tensor<T: 'static + Debug + Copy> {
    /// Arc for shared ownership, RwLock for interior mutability of TensorData.
    pub(crate) data: Arc<RwLock<TensorData<T>>>,
}

// --- Public Exports ---
// Remove this re-export as it causes name collision
// pub use crate::tensor_data::TensorData;

// --- Test Module ---
// Removed: #[cfg(test)] mod tests;
