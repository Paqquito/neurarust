#![allow(clippy::needless_borrow)] // Needed because of trait bounds propagation in ops
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
use std::iter::Product;

pub mod utils;

/// Represents a multi-dimensional array (Tensor).
/// Uses Arc<RwLock<TensorData<T>>> for thread-safe interior mutability
/// and shared ownership. Allows multiple Tensors (views) to share the same
/// underlying data buffer while having potentially different shapes/strides/offsets.
pub struct Tensor<T: 'static + Debug + Copy> {
    // The core data and metadata, protected by a read-write lock and
    // shared across potentially multiple Tensor instances via an Arc.
    pub(crate) data: Arc<RwLock<TensorData<T>>>,
}

// Helper function for recursive multidimensional iteration used by contiguous()
// This function iterates through the dimensions and calls itself recursively.
// It builds up the current index set and, at the deepest level, calculates
// the source offset, reads the data, and pushes it to the new contiguous buffer.
fn copy_non_contiguous_recursive<T: Clone + Debug + Copy + 'static>(
    original_guard: &RwLockReadGuard<'_, TensorData<T>>,
    original_cpu_data: &Arc<Vec<T>>, // Pass Arc reference
    new_buffer: &mut Vec<T>,
    current_indices: &mut Vec<usize>,
    current_dim: usize,
) {
    if current_dim == original_guard.shape.len() {
        // Base case: We have a full index set
        let original_offset = original_guard.get_offset(current_indices);
        new_buffer.push(original_cpu_data[original_offset].clone());
    } else {
        // Recursive step: Iterate through the current dimension
        for i in 0..original_guard.shape[current_dim] {
            current_indices[current_dim] = i;
            copy_non_contiguous_recursive(
                original_guard,
                original_cpu_data,
                new_buffer,
                current_indices,
                current_dim + 1,
            );
        }
    }
}

// --- Combine all inherent methods into one block ---
impl<T: 'static + Debug + Copy> Tensor<T> {
    // --- Constructors and Basic Properties ---
    /// Creates a new tensor from a vector of data and a shape.
    /// The tensor will be allocated on the CPU by default.
    pub fn new(data_vec: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new(data_vec, shape)?;
        Ok(Tensor {
            // Wrap TensorData in RwLock and Arc
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }
    
    // TODO: Add constructors for other devices, e.g., `new_gpu(...)`

    /// Creates a tensor of zeros with the specified shape on the CPU.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: Zero {
        let numel = shape.iter().product::<usize>();
        let data_vec = vec![T::zero(); numel];
        Tensor::new(data_vec, shape)
    }

    /// Creates a tensor of zeros with the same shape and device as another tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Result<Self, NeuraRustError> where T: Zero {
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
        self.data.read().expect("RwLock poisoned").numel()
    }

    /// Returns a clone of the thread-safe reference-counted pointer (Arc)
    /// to the underlying shared data buffer (Buffer<T>).
    /// Acquires a read lock temporarily to access the Arc.
    /// Panics if the lock is poisoned.
    pub fn borrow_data_buffer(&self) -> Arc<crate::buffer::Buffer<T>> {
        // Acquire read lock, unwrap, clone Arc<Buffer<T>>
        self.data.read().expect("RwLock poisoned").data.clone()
    }

    /// Retrieves a single element from the tensor using multi-dimensional indices.
    /// Requires the tensor to be on the CPU.
    /// Acquires a read lock. Panics if lock is poisoned.
    /// Returns DataNotAvailableError if tensor is not on CPU.
    pub fn get(&self, indices: &[usize]) -> Result<T, NeuraRustError> {
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
        Ok(cpu_data_arc[offset])
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
    pub fn scalar(value: T) -> Self {
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
        T: Default + Send + Sync,
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
        T: Default + Send + Sync,
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
        T: Default + Send + Sync,
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
        T: Default + Send + Sync,
    {
        crate::ops::view_ops::reshape_op(self, new_shape)
    }

    /// Returns a contiguous tensor containing the same data.
    ///
    /// If the tensor is already contiguous, this method clones the tensor (cheaply, via Arc).
    /// If the tensor is not contiguous, it performs a data copy to create a new,
    /// contiguous tensor on the same device.
    ///
    /// # Returns
    /// A Result containing the contiguous tensor, or an error if data access fails (e.g., non-CPU).
    pub fn contiguous(&self) -> Result<Self, NeuraRustError>
    where
        T: Default + Send + Sync + Product,
    {
        if self.is_contiguous() {
            Ok(self.clone()) // Cheap clone if already contiguous
        } else {
            // Need to copy data
            let guard = self.read_data();
            let device = guard.device;
            let shape = guard.shape.clone();
            let numel = guard.numel();

            // --- Data Copy Logic (CPU only for now) ---
            match device {
                StorageDevice::CPU => {
                    // Get access to the original CPU buffer data
                    let original_cpu_data = guard.data.cpu_data()?;
                    let mut new_buffer = Vec::with_capacity(numel);

                    // Perform multidimensional iteration and copy
                    let mut current_indices = vec![0; shape.len()];
                    copy_non_contiguous_recursive(
                        &guard,
                        &original_cpu_data, // Pass Arc reference
                        &mut new_buffer,
                        &mut current_indices,
                        0, // Start recursion at dimension 0
                    );

                    // Drop the read guard before creating the new tensor
                    drop(guard);

                    // Create a new tensor from the copied, contiguous buffer
                    // Tensor::new will calculate the correct contiguous strides.
                    Tensor::new(new_buffer, shape)
                }
                StorageDevice::GPU => {
                    // TODO: Implement contiguous copy for GPU tensors
                    Err(NeuraRustError::UnsupportedOperation(
                        "Contiguous copy for GPU tensors not yet implemented".to_string(),
                    ))
                }
            }
            // --- End Data Copy Logic ---
        }
    }

    // Optional: Alias or stricter view-only version
    // pub fn view(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError> where ... { self.reshape(new_shape) }
}

// --- Traits Implementations ---
impl<T: 'static + Debug + Copy> Clone for Tensor<T> {
    /// Clones the Tensor. This creates a new Tensor instance that shares the
    /// same underlying data and metadata via the Arc<RwLock<...>>.
    /// It does NOT perform a deep copy of the data.
    fn clone(&self) -> Self {
        // Clone the Arc, not the data inside
        Tensor { data: Arc::clone(&self.data) }
    }
}

impl<T: 'static + Debug + Copy> Debug for Tensor<T> {
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

impl<T: PartialEq + 'static + Debug + Copy> PartialEq for Tensor<T> {
    /// Checks for tensor equality.
    /// Two tensors are equal if they have the same shape, device, offset, strides,
    /// AND the corresponding elements in their *effective* data views are equal.
    /// This means it compares the data accessible via shape/strides/offset,
    /// not necessarily the entire underlying buffers if they are views.
    /// **Does not consider autograd fields.**
    /// **Currently, data comparison only works reliably for CPU tensors.**
    fn eq(&self, other: &Self) -> bool {
        // Fast path: check if they point to the exact same Arc<RwLock<TensorData>>
        if Arc::ptr_eq(&self.data, &other.data) {
            return true;
        }

        // Acquire read locks
        let self_guard = self.read_data();
        let other_guard = other.read_data();

        // Compare metadata first
        if self_guard.shape != other_guard.shape ||
           self_guard.device != other_guard.device ||
           self_guard.offset != other_guard.offset || // Include offset comparison
           self_guard.strides != other_guard.strides // Include strides comparison
        {
            return false;
        }

        // If metadata matches, attempt to compare data content.
        // Only implement comparison for CPU tensors for now.
        match (self_guard.device, other_guard.device) {
            (StorageDevice::CPU, StorageDevice::CPU) => {
                match (self_guard.data.cpu_data(), other_guard.data.cpu_data()) {
                    (Ok(self_cpu_data), Ok(other_cpu_data)) => {
                        // If they share the same buffer Arc, metadata match implies equality.
                        if Arc::ptr_eq(&self_guard.data, &other_guard.data) {
                            return true;
                        }

                        // If buffers are different, we MUST compare element-wise respecting views.
                        // Initial simplified version: Only compare content if both are CONTIGUOUS
                        // and start at the same offset (covers test_tensor_equality t1 vs t2 case).
                        // This is NOT a complete view equality check.
                        // TODO: Implement full element-wise comparison respecting strides/offsets for views.
                        if self_guard.is_contiguous() && other_guard.is_contiguous() && self_guard.offset == other_guard.offset {
                            // Compare the relevant slices of the underlying Vec<T>
                            let numel = self_guard.numel();
                            let self_slice = &self_cpu_data[self_guard.offset..self_guard.offset + numel];
                            let other_slice = &other_cpu_data[other_guard.offset..other_guard.offset + numel];
                            self_slice == other_slice
                        } else {
                            // Cannot reliably compare non-contiguous views or views with different offsets
                            // without proper iteration logic.
                            eprintln!(
                                "Warning: PartialEq comparing non-contiguous CPU Tensors or views with different offsets. Returning false. Implement element-wise comparison."
                            );
                            false
                        }
                    }
                    _ => false, // Failed to get CPU data, consider unequal
                }
            }
            (StorageDevice::GPU, StorageDevice::GPU) => {
                // For GPU tensors, consider them equal only if they share the same buffer Arc
                // and metadata already matched. Content comparison requires GPU kernels or H2D copies.
                 Arc::ptr_eq(&self_guard.data, &other_guard.data)
            }
            _ => false, // Tensors on different devices are not equal
        }
    }
}

impl<T: Eq + 'static + Debug + Copy> Eq for Tensor<T> {} // Eq follows from PartialEq

impl<T: Hash + 'static + Debug + Copy> Hash for Tensor<T> {
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
pub fn zeros<T: Zero + 'static + Debug + Copy>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
    Tensor::zeros(shape)
}

/// Creates a tensor of ones with the specified shape on the CPU.
pub fn ones<T: One + 'static + Debug + Copy>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data_vec = vec![T::one(); numel];
    Tensor::new(data_vec, shape)
}

/// Creates a tensor filled with a specific value on the CPU.
pub fn full<T: 'static + Debug + Copy>(shape: Vec<usize>, fill_value: T) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data_vec = vec![fill_value; numel];
    Tensor::new(data_vec, shape)
}

// Declare the tests module, indicating it's in a separate file (tests.rs or tests/mod.rs)
#[cfg(test)]
mod tests;
