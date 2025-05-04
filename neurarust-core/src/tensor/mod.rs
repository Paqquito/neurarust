#![allow(clippy::needless_borrow)]
// src/tensor/mod.rs

use crate::tensor_data::TensorData;
use std::sync::{Arc, RwLock};
use crate::error::NeuraRustError;
use crate::types::DType;
use crate::device::StorageDevice;

// --- Define existing implementation modules ---
// These should correspond to actual files like `autograd_methods.rs`
mod autograd_methods;
mod traits; // Keep traits module declared
// Remove module declaration, implementation will be below
// mod traits;
// Add other existing modules here if they exist (e.g., based on previous structure or file listing)
// mod accessors;
pub mod create; // Make the create module public
mod view_methods;

// --- Declare utility modules ---
pub mod utils; // Declare public utils
pub mod broadcast_utils; // Declare the new public broadcast utils module

// Re-export creation functions to make them public
pub use create::{zeros, ones, full, zeros_like, ones_like};

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

// --- Tensor Implementation --- (Assuming basic methods might be here)
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

    // Add basic accessors here (or confirm they are in accessors.rs)
    // These methods typically just acquire a read lock and return the corresponding field.

    /// Returns the data type (`DType`) of the tensor elements.
    pub fn dtype(&self) -> DType {
        self.data.read().unwrap().dtype
    }

    /// Returns the device (`StorageDevice`) where the tensor's data resides.
    pub fn device(&self) -> StorageDevice {
        self.data.read().unwrap().device
    }

    /// Returns a clone of the tensor's shape (`Vec<usize>`).
    pub fn shape(&self) -> Vec<usize> {
        self.data.read().unwrap().shape.clone()
    }

    /// Returns a clone of the tensor's strides (`Vec<usize>`).
    pub fn strides(&self) -> Vec<usize> {
        self.data.read().unwrap().strides.clone()
    }

    /// Checks if the tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.data.read().unwrap().is_contiguous()
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.data.read().unwrap().numel()
    }

    /// Acquires a read lock on the tensor's data.
    ///
    /// This allows reading the `TensorData` fields immutably.
    /// The lock is automatically released when the guard goes out of scope.
    /// Panics if the RwLock is poisoned.
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<'_, TensorData> {
        self.data.read().expect("RwLock poisoned")
    }

    /// Acquires a write lock on the tensor's data.
    ///
    /// This allows modifying the `TensorData` fields mutably.
    /// The lock is automatically released when the guard goes out of scope.
    /// Panics if the RwLock is poisoned.
    pub fn write_data(&self) -> std::sync::RwLockWriteGuard<'_, TensorData> {
        self.data.write().expect("RwLock poisoned")
    }

    // --- New Helper Methods for GradCheck ---

    /// Attempts to get the tensor data as a `Vec<f32>`.
    /// Returns an error if the tensor is not on the CPU or not F32.
    pub fn get_f32_data(&self) -> Result<Vec<f32>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_f32_data".to_string(),
            });
        }
        if guard.dtype != DType::F32 {
            // Use UnsupportedOperation for type mismatch
            return Err(NeuraRustError::UnsupportedOperation(
                format!("get_f32_data requires DType::F32, got {:?}", guard.dtype)
            ));
        }
        // TODO: Add support for non-contiguous tensors if necessary.
        // Currently assumes underlying buffer can be directly cloned.
        // If non-contiguous, would need to create a new contiguous Vec.
        if !guard.is_contiguous() {
             return Err(NeuraRustError::UnsupportedOperation(
                 "get_f32_data on non-contiguous tensor not implemented.".to_string()
             ));
        }

        // Use buffer().try_get_cpu_f32()?
        let buffer_arc = guard.buffer().try_get_cpu_f32()?;
        // Clone the Vec itself to return owned data
        Ok(buffer_arc.as_ref().clone())
    }

    /// Creates a new CPU F32 Tensor from a Vec<f32> and shape.
    /// Calculates contiguous strides. Requires grad defaults to false.
    pub fn from_vec_f32(data_vec: Vec<f32>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        // Call the adapted TensorData::new which handles f32/CPU
        let tensor_data = TensorData::new(data_vec, shape)?; // TensorData::new defaults requires_grad=false
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Clears the gradient tensor associated with this tensor.
    pub fn clear_grad(&self) {
        // Only makes sense if requires_grad is true, but harmless otherwise
        if self.requires_grad() {
             let mut guard = self.write_data();
             guard.grad = None;
        }
    }
}
