#![allow(clippy::needless_borrow)]
// src/tensor/mod.rs

// Keep necessary imports for the main struct definition
// Remove this redundant import
// use crate::tensor_data::TensorData;
use std::fmt::Debug; // Keep Debug and Copy for the struct bound
use std::marker::Copy;
use std::sync::{Arc, RwLock};

// Define modules within the tensor directory
// These files contain the implementations of methods previously in this file.
mod accessors;
mod autograd_methods;
mod create;
mod traits;
mod view_methods;

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
// Remove Clone derive here
// Remove Debug derive here
pub struct Tensor<T: 'static + Debug + Copy> {
    /// Arc for shared ownership, RwLock for interior mutability of TensorData.
    pub(crate) data: Arc<RwLock<TensorData<T>>>,
}

// Declare the modules
pub mod utils; // Keep utils module declaration

// Re-export standalone creation functions for convenience
// pub use create::{zeros, ones, full, scalar};

// The test module reference remains here
#[cfg(test)]
mod tests;

// --- Public Exports ---
// Export the main Tensor struct
pub use crate::tensor_data::TensorData; // Re-export TensorData if needed externally
                                        // Remove these re-exports, users will access methods via Tensor::method
                                        // pub use crate::tensor::accessors::*; // Export methods related to accessing tensor properties
                                        // pub use crate::tensor::autograd_methods::*; // Export methods related to autograd
                                        // pub use crate::tensor::create::*; // Export methods related to tensor creation
                                        // pub use crate::tensor::traits::*; // Export trait implementations
                                        // pub use crate::tensor::view_methods::*; // Export view/reshape/etc methods

// Remove the re-export of specific creation functions
// Users will call Tensor::zeros, Tensor::ones, etc.
// pub use create::{zeros, ones, full, scalar};

// --- Potentially keep standalone functions if they have a different API ---
// Example: Maybe a version of zeros that infers device differently?
// If not, these can be removed. For now, rely on Tensor::zeros etc.

// pub fn zeros<T: Zero + 'static + Debug + Copy>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
//     Tensor::zeros(shape)
// }
// pub fn ones<T: One + 'static + Debug + Copy>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
//     Tensor::ones(shape)
// }
// pub fn full<T: 'static + Debug + Copy>(shape: Vec<usize>, fill_value: T) -> Result<Tensor<T>, NeuraRustError> {
//     Tensor::full(shape, fill_value)
// }
