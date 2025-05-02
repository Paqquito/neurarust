// src/tensor/mod.rs
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
// Use Arc and RwLock for thread-safe sharing and interior mutability
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use num_traits::{Zero, One};

// Import new types
use crate::device::StorageDevice;
use crate::tensor_data::TensorData;
use crate::error::NeuraRustError;

pub mod utils;

/// Represents a multi-dimensional array (Tensor).
/// Uses Arc<RwLock<TensorData<T>>> for thread-safe interior mutability
/// and shared ownership. Allows multiple Tensors (views) to share the same
/// underlying data buffer while having potentially different shapes/strides/offsets.
pub struct Tensor<T> {
    // The core data and metadata, protected by a read-write lock and
    // shared across potentially multiple Tensor instances via an Arc.
    pub(crate) data: Arc<RwLock<TensorData<T>>>,
}

// --- Combine all inherent methods into one block ---
impl<T> Tensor<T> {
    // --- Constructors and Basic Properties ---
    /// Creates a new tensor from a vector of data and a shape.
    /// The tensor will be allocated on the CPU by default.
    pub fn new(data_vec: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: Clone {
        let tensor_data = TensorData::new(data_vec, shape)?;
        Ok(Tensor {
            // Wrap TensorData in RwLock and Arc
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }
    
    // TODO: Add constructors for other devices, e.g., `new_gpu(...)`

    /// Creates a tensor of zeros with the specified shape on the CPU.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: Zero + Clone {
        let numel = shape.iter().product::<usize>();
        let data_vec = vec![T::zero(); numel];
        Tensor::new(data_vec, shape)
    }

    /// Creates a tensor of zeros with the same shape and device as another tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Result<Self, NeuraRustError> where T: Zero + Clone {
        // Acquire read lock to get shape and device
        let other_guard = other.read_data();
        let shape = other_guard.shape.clone();
        let device = other_guard.device;
        // Explicitly drop guard before creating new tensor to avoid deadlock
        drop(other_guard);

        match device {
            StorageDevice::CPU => {
                let numel = shape.iter().product::<usize>();
                let data_vec = vec![T::zero(); numel];
                Tensor::new(data_vec, shape)
            }
            StorageDevice::GPU => {
                // TODO: Implement GPU tensor creation
                Err(NeuraRustError::UnsupportedOperation("GPU zero tensor creation not yet implemented".to_string()))
            }
        }
    }

    /// Returns the shape of the tensor. Acquires a read lock.
    /// Panics if the lock is poisoned.
    pub fn shape(&self) -> Vec<usize> {
        // Acquire read lock, unwrap (panic on poison), clone shape
        self.data.read().expect("RwLock poisoned").shape.clone()
    }

    /// Returns the strides of the tensor. Acquires a read lock.
    /// Panics if the lock is poisoned.
    pub fn strides(&self) -> Vec<usize> {
        // Acquire read lock, unwrap, clone strides
        self.data.read().expect("RwLock poisoned").strides.clone()
    }

    /// Returns the device where the tensor's data is stored. Acquires a read lock.
    /// Panics if the lock is poisoned.
    pub fn device(&self) -> StorageDevice {
        self.data.read().expect("RwLock poisoned").device
    }

    /// Returns the number of dimensions (rank) of the tensor. Acquires a read lock.
    /// Panics if the lock is poisoned.
    pub fn ndim(&self) -> usize {
        // Acquire read lock, unwrap, get shape len
        self.data.read().expect("RwLock poisoned").shape.len()
    }

    /// Returns the total number of elements in the tensor. Acquires a read lock.
    /// Panics if the lock is poisoned.
    pub fn numel(&self) -> usize {
        // Acquire read lock, unwrap, calculate numel from shape
        self.data.read().expect("RwLock poisoned").shape.iter().product()
    }

    /// Returns a clone of the thread-safe reference-counted pointer (Arc)
    /// to the underlying shared data buffer (Buffer<T>).
    /// Acquires a read lock temporarily to access the Arc.
    /// Panics if the lock is poisoned.
    pub fn borrow_data_buffer(&self) -> Arc<crate::buffer::Buffer<T>> { // Return type changed
        // Acquire read lock, unwrap, clone Arc<Buffer<T>>
        self.data.read().expect("RwLock poisoned").data.clone()
    }

    /// Retrieves a single element from the tensor using multi-dimensional indices.
    /// Requires the tensor to be on the CPU.
    /// Acquires a read lock. Panics if lock is poisoned.
    /// Returns DataNotAvailableError if tensor is not on CPU.
    pub fn get(&self, indices: &[usize]) -> Result<T, NeuraRustError> where T: Clone {
        // Acquire read lock
        let td = self.data.read().expect("RwLock poisoned");
        
        // Ensure data is on CPU before trying to access vec
        let cpu_data_arc = td.data.cpu_data()?; // Returns error if not CPU

        // Check bounds using locked data
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
        // Calculate offset using locked data
        let offset = td.get_offset(indices);
        // Access data via Arc<Vec<T>> and clone the element
        Ok(cpu_data_arc[offset].clone())
        // Read lock is released here when 'td' goes out of scope
    }

    // --- Accessing internal data ---
    // It's generally safer to provide methods like shape(), strides(), etc.
    // instead of exposing the lock guards directly, but these can be
    // useful for operations needing multiple fields consistently.

    /// Provides immutable access (read guard) to the underlying TensorData.
    /// The caller is responsible for handling the lock guard.
    /// Panics if the lock is poisoned.
    pub fn read_data(&self) -> RwLockReadGuard<'_, TensorData<T>> {
        self.data.read().expect("RwLock poisoned")
    }

    /// Provides mutable access (write guard) to the underlying TensorData.
    /// Use with caution, only when metadata needs mutation (e.g., for autograd).
    /// Panics if the lock is poisoned.
    pub fn write_data(&self) -> RwLockWriteGuard<'_, TensorData<T>> {
        self.data.write().expect("RwLock poisoned")
    }

    // --- ID Methods ---

    /// Returns the raw pointer to the RwLock<TensorData>. Used as a unique ID.
    /// This pointer itself is stable even if the lock is held.
    pub fn id_ptr(&self) -> *const RwLock<TensorData<T>> {
        // Get pointer to the Arc's inner value (the RwLock)
        Arc::as_ptr(&self.data)
    }

    /// Returns a type-erased pointer, useful as a unique identifier.
    pub fn id(&self) -> *const () {
        Arc::as_ptr(&self.data) as *const ()
    }

    // --- Scalar Creation ---

    /// Creates a scalar tensor (0-dimensional) on the CPU.
    pub fn scalar(value: T) -> Self where T: Clone {
        Tensor::new(vec![value], vec![]).expect("Scalar creation failed")
    }

    // --- View Operations ---

    /// Checks if the tensor is contiguous in memory.
    /// Acquires a read lock.
    /// Panics if the lock is poisoned.
    /// See `TensorData::is_contiguous` for details.
    pub fn is_contiguous(&self) -> bool {
        self.read_data().is_contiguous()
    }

    /// Creates a view of the tensor by slicing along specified dimensions.
    /// This operation does not copy data; the new tensor shares the same underlying buffer.
    ///
    /// # Arguments
    /// * `ranges`: A slice of tuples `(start, end)` defining the slice for each dimension.
    ///             The length must match the tensor's rank. `end` is exclusive.
    ///
    /// # Returns
    /// A new Tensor representing the view, or an error if the ranges are invalid.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Assuming tensor is 2x3
    /// // Slice to get the first row: tensor.slice(&[(0, 1), (0, 3)])?;
    /// // Slice to get the sub-matrix [[elem_11, elem_12]]: tensor.slice(&[(1, 2), (1, 3)])?;
    /// ```
    pub fn slice(&self, ranges: &[crate::ops::view_ops::SliceArg]) -> Result<Self, NeuraRustError>
    where
        T: Clone + Debug + Default + Send + Sync + 'static, // Match trait bounds of slice_op
    {
        // Call the internal slice_op function from the view_ops module
        crate::ops::view_ops::slice_op(self, ranges)
    }

    /// Returns a view of the tensor with dimensions `dim1` and `dim2` swapped.
    /// This is a view operation and does not copy data.
    ///
    /// # Arguments
    /// * `dim1`: The first dimension to transpose.
    /// * `dim2`: The second dimension to transpose.
    ///
    /// # Returns
    /// A new Tensor representing the transposed view, or an error if dimensions are invalid.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Assuming tensor is 2x3
    /// let transposed_tensor = tensor.transpose(0, 1)?; // Shape becomes 3x2
    /// ```
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, NeuraRustError>
    where
        T: Clone + Debug + Default + Send + Sync + 'static, // Match trait bounds of transpose_op
    {
        crate::ops::view_ops::transpose_op(self, dim1, dim2)
    }

    /// Returns a view of the tensor with its dimensions permuted according to `dims`.
    /// This is a view operation and does not copy data.
    ///
    /// `dims` must be a permutation of `0..self.ndim()`.
    ///
    /// # Arguments
    /// * `dims`: The desired ordering of dimensions.
    ///
    /// # Returns
    /// A new Tensor representing the permuted view, or an error if `dims` is invalid.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Assuming tensor is 2x3x4
    /// let permuted_tensor = tensor.permute(&[2, 0, 1])?; // Shape becomes 4x2x3
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Result<Self, NeuraRustError>
    where
        T: Clone + Debug + Default + Send + Sync + 'static, // Match trait bounds of permute_op
    {
        crate::ops::view_ops::permute_op(self, dims)
    }

    /// Returns a view of the tensor with the specified shape.
    ///
    /// This operation attempts to return a view without copying data. Currently,
    /// this is only possible if the original tensor is contiguous.
    /// If the tensor is not contiguous, this method will return an error.
    /// Call `.contiguous().reshape(...)` to ensure the operation succeeds by potentially copying data first.
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    /// * `new_shape`: The desired new shape.
    ///
    /// # Returns
    /// A new Tensor representing the reshaped view, or an error.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Assuming tensor is contiguous 2x6
    /// let reshaped_tensor = tensor.reshape(vec![3, 4])?; // Shape becomes 3x4
    /// // Assuming tensor is non-contiguous
    /// // tensor.reshape(vec![3, 4]); -> Returns Error
    /// // tensor.contiguous()?.reshape(vec![3, 4])?; -> Ok
    /// ```
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where
        T: Clone + Debug + Default + Send + Sync + 'static, // Match trait bounds of reshape_op
    {
        crate::ops::view_ops::reshape_op(self, new_shape)
    }

    // Optional: Alias or stricter view-only version
    // pub fn view(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError> where ... { self.reshape(new_shape) }
}

// --- Traits Implementations ---
impl<T> Clone for Tensor<T> {
    /// Clones the Tensor. This creates a new Tensor instance that shares the
    /// same underlying data and metadata via the Arc<RwLock<...>>.
    /// It does NOT perform a deep copy of the data.
    fn clone(&self) -> Self {
        // Clone the Arc, not the data inside
        Tensor { data: Arc::clone(&self.data) }
    }
}

impl<T: Debug> Debug for Tensor<T> where T: Clone { // Added Clone bound for get()
    /// Formats the Tensor for debugging. Acquires a read lock.
    /// Panics if the lock is poisoned. Displays data only if on CPU.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Acquire read lock
        let td = self.data.read().expect("RwLock poisoned");
        // Try to format based on device
        match td.device {
            StorageDevice::CPU => {
                // Safely get CPU data for display
                match td.data.cpu_data() {
                    Ok(cpu_data_arc) => {
                        // TODO: Implement a more sophisticated display logic that respects shape/strides
                        // For now, just show metadata and the first few elements of the raw buffer
                        write!(f, "Tensor(CPU, shape={:?}, strides={:?}, offset={}, data=[{:?}...])",
                               td.shape,
                               td.strides,
                               td.offset,
                               cpu_data_arc.iter().take(10).collect::<Vec<_>>() // Show up to 10 elements
                        )
                    }
                    Err(_) => write!(f, "Tensor(CPU, shape={:?}, strides={:?}, offset={}, data=<Error getting CPU data>)", td.shape, td.strides, td.offset)
                }
            },
            StorageDevice::GPU => {
                write!(f, "Tensor(GPU, shape={:?}, strides={:?}, offset={}, buffer_len={})",
                       td.shape, td.strides, td.offset, td.data.len())
            }
        }
        // Read lock is dropped here
    }
}

impl<T: PartialEq + Clone> PartialEq for Tensor<T> {
    /// Compares two tensors for equality.
    /// Tensors are considered equal if they point to the same underlying locked data
    /// (checked via Arc::ptr_eq) OR if their underlying TensorData (shape, strides, offset,
    /// device, and data Arc<Buffer> pointer) are equal when compared field-by-field.
    /// Note: This does NOT compare the numerical content of the data buffers deeply.
    fn eq(&self, other: &Self) -> bool {
        // Fast path: check if they point to the exact same Arc<RwLock>
        if Arc::ptr_eq(&self.data, &other.data) {
            return true;
        }
        // Otherwise, acquire read locks and compare the TensorData inside.
        // The derived PartialEq on TensorData compares fields, including the Buffer (which uses Arc::ptr_eq for Cpu).
        *self.data.read().expect("RwLock poisoned") == *other.data.read().expect("RwLock poisoned")
    }
}

impl<T: Eq + Clone> Eq for Tensor<T> {} // Eq follows from PartialEq

impl<T> Hash for Tensor<T> {
    /// Hashes the Tensor based on the pointer address of the Arc wrapping the RwLock.
    /// Tensors sharing the same allocation (clones) will have the same hash.
    /// Tensors with different allocations, even if holding identical views or data,
    /// will likely have different hashes.
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the pointer address of the Arc itself
        Arc::as_ptr(&self.data).hash(state);
    }
}

// --- Standalone Creation Functions (Assume CPU for now) ---
/// Creates a tensor of zeros with the specified shape on the CPU.
pub fn zeros<T: Zero + Clone>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
    Tensor::<T>::zeros(shape)
}

/// Creates a tensor of ones with the specified shape on the CPU.
pub fn ones<T: One + Clone>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data_vec = vec![T::one(); numel];
    Tensor::new(data_vec, shape)
}

/// Creates a tensor filled with a specific value on the CPU.
pub fn full<T: Clone>(shape: Vec<usize>, fill_value: T) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data_vec = vec![fill_value; numel];
    Tensor::new(data_vec, shape)
}

// Declare the tests module, indicating it's in a separate file (tests.rs or tests/mod.rs)
#[cfg(test)]
mod tests;
