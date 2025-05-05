#![allow(clippy::needless_borrow)]
// src/tensor/mod.rs

use crate::tensor_data::TensorData;
use std::sync::{Arc, RwLock};
use crate::error::NeuraRustError;

// --- Define existing implementation modules ---
// These should correspond to actual files like `autograd_methods.rs`
mod autograd_methods;
mod traits; // Keep traits module declared
mod accessors; // Ensure accessors module is declared
mod reduction_methods; // Declare the new module
mod view_methods; // Declare view methods

pub mod create; // Make the create module public

// --- Declare utility modules ---
pub mod utils; // Declare public utils
pub mod broadcast_utils; // Declare the new public broadcast utils module
pub mod iter_utils; // Declare the new iterator utils module

// Re-export creation functions to make them public
pub use create::{zeros, ones, full, zeros_like, ones_like,
                 zeros_f64, ones_f64, full_f64,
                 from_vec_f32, from_vec_f64};

/// Represents a multi-dimensional array (tensor).
///
/// `Tensor` uses `Arc<RwLock<TensorData>>` internally to allow for:
/// 1.  **Shared Ownership:** Multiple `Tensor` instances can point to the same
///     underlying data without cloning the data itself (cheap clones).
/// 2.  **Interior Mutability:** Metadata (like `requires_grad` or `grad`) within
///     `TensorData` can be modified even through an immutable `Tensor` reference,
///     using the `RwLock`. Read/write locks ensure thread safety.
///
/// Tensor now holds non-generic `TensorData` which internally manages
/// the data type (`DType`) and the typed buffer (`Buffer`).
pub struct Tensor {
    /// Arc for shared ownership, RwLock for interior mutability of TensorData.
    pub(crate) data: Arc<RwLock<TensorData>>,
}

// --- Test Module ---
// Removed: #[cfg(test)] mod tests;

// --- Re-exports ---
// pub use tensor_data::TensorData; // Commented out
// pub use storage::{Buffer, Storage, StorageDevice}; // Commented out
// pub use create::{zeros, ones, full}; // Commented out

// pub use methods::{Tensor, TensorMethods}; // Commented out

// pub use crate::error::NeuraRustError; // Commented out

// --- Tensor Implementation --- 
impl Tensor {
    /// Creates a new Tensor with the given f32 data and shape on the CPU.
    ///
    /// This is the primary constructor for creating tensors from raw data.
    /// It calculates contiguous strides automatically.
    pub fn new(data_vec: Vec<f32>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        // Call the adapted TensorData::new which handles f32/CPU
        let tensor_data = TensorData::new(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Creates a new Tensor with the given f64 data and shape on the CPU.
    ///
    /// This constructor creates F64 tensors.
    /// It calculates contiguous strides automatically.
    pub fn new_f64(data_vec: Vec<f64>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        // Call the adapted TensorData::new_f64 which handles it
        let tensor_data = TensorData::new_f64(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    // Accessors like shape(), dtype(), device(), strides(), is_contiguous(), numel() are removed
    // as they are now definitively in accessors.rs

    // Methods read_data(), write_data(), get_f32_data(), get_f64_data() are moved to accessors.rs

    // Remaining methods to be moved:
    // clear_grad(), mean(), max()

    // All methods previously here have been moved to:
    // - accessors.rs (dtype, device, shape, strides, numel, is_contiguous, read_data, write_data, get_f32/f64_data)
    // - create.rs (from_vec_f32, from_vec_f64 - as free functions)
    // - autograd_methods.rs (clear_grad)
    // - reduction_methods.rs (mean, max)
}
