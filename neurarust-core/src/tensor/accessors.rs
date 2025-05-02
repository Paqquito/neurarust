// src/tensor/accessors.rs

use super::Tensor;
use crate::tensor_data::TensorData;
use crate::error::NeuraRustError;
use crate::device::StorageDevice;
use crate::buffer::Buffer; // Needed for borrow_data_buffer
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::fmt::Debug;
use std::marker::Copy;
use crate::autograd::graph::NodeId;

// Note: T bounds for the impl block cover all methods inside
impl<T: 'static + Debug + Copy> Tensor<T> {
    /// Returns the shape of the tensor. Acquires a read lock.
    pub fn shape(&self) -> Vec<usize> {
        self.read_data().shape.clone()
    }

    /// Returns the strides of the tensor. Acquires a read lock.
    pub fn strides(&self) -> Vec<usize> {
        self.read_data().strides.clone()
    }

    /// Returns the device where the tensor's data is stored. Acquires a read lock.
    pub fn device(&self) -> StorageDevice {
        self.read_data().device
    }

    /// Returns the number of dimensions (rank) of the tensor. Acquires a read lock.
    pub fn ndim(&self) -> usize {
        self.read_data().shape.len()
    }

    /// Returns the total number of elements in the tensor. Acquires a read lock.
    pub fn numel(&self) -> usize {
        self.read_data().numel()
    }

    /// Returns a clone of the thread-safe reference-counted pointer (Arc)
    /// to the underlying shared data buffer (Buffer<T>).
    /// Acquires a read lock temporarily to access the Arc.
    pub fn borrow_data_buffer(&self) -> Arc<Buffer<T>> {
        self.read_data().data.clone()
    }

    /// Retrieves a single element from the tensor using multi-dimensional indices.
    /// Requires the tensor to be on the CPU.
    /// Acquires a read lock.
    pub fn get(&self, indices: &[usize]) -> Result<T, NeuraRustError> {
        let td = self.read_data();
        let cpu_data_arc = td.data.cpu_data()?;

        if indices.len() != td.shape.len() {
            return Err(NeuraRustError::DimensionMismatch {
                expected: td.shape.len(),
                actual: indices.len(),
            });
        }
        for i in 0..td.shape.len() {
            if indices[i] >= td.shape[i] {
                return Err(NeuraRustError::IndexOutOfBounds {
                    index: indices.to_vec(),
                    shape: td.shape.clone(),
                });
            }
        }
        let offset = td.get_offset(indices);
        Ok(cpu_data_arc[offset])
    }

    // --- Accessing internal data ---

    /// Provides immutable access (read guard) to the underlying TensorData.
    /// Panics if the lock is poisoned.
    // Make this pub(super) so other modules within tensor can access it?
    // Or keep it public for now.
    pub fn read_data(&self) -> RwLockReadGuard<'_, TensorData<T>> {
        self.data.read().expect("RwLock poisoned")
    }

    /// Provides mutable access (write guard) to the underlying TensorData.
    /// Panics if the lock is poisoned.
    pub fn write_data(&self) -> RwLockWriteGuard<'_, TensorData<T>> {
        self.data.write().expect("RwLock poisoned")
    }

    // --- ID Methods ---

    /// Returns the raw pointer to the RwLock<TensorData>. Used as a unique ID.
    pub fn id_ptr(&self) -> *const RwLock<TensorData<T>> {
        Arc::as_ptr(&self.data)
    }

    /// Returns a type-erased pointer, useful as a unique identifier.
    pub fn id(&self) -> *const () {
        Arc::as_ptr(&self.data) as *const ()
    }

    /// Helper function to get the NodeId (*const RwLock<TensorData<T>>) for this Tensor.
    /// Used internally by autograd operations.
    pub(crate) fn get_node_id(&self) -> NodeId<T> {
        // `Arc::as_ptr` gives a pointer to the inner value (RwLock<TensorData<T>>).
        // This pointer is stable as long as the Arc lives.
        // This is inherently unsafe if not managed carefully, but within the autograd
        // system, we ensure the corresponding Tensors (and thus Arcs) are kept alive.
        std::sync::Arc::as_ptr(&self.data)
    }
} 