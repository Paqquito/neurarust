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

// --- Tests (Simplified for Phase 0) ---
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_traits::{Zero, One};
    use std::ops::AddAssign; // Keep AddAssign for helper
    use std::fmt::Debug;
    use std::iter::Sum; // Keep Sum for helper
    use std::cmp::PartialEq; // Keep PartialEq for helper
    use std::default::Default; // Keep Default for helper
    use std::sync::Arc; // For Arc::ptr_eq

    // Helper needs Copy trait because test_tensor_creation uses .clone() on f32 data
    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where T: Clone + Debug + PartialEq + Zero + One + Copy + AddAssign + Sum + Default
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    // Tests remain largely the same, accessing methods like shape(), get()
    // which now handle the locking internally.
    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let t = create_test_tensor(data.clone(), shape.clone());
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.strides(), vec![2, 1]);
        assert_relative_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_relative_eq!(t.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_tensor_creation_error() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![2, 2];
        let result = Tensor::<f32>::new(data, shape);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::TensorCreationError { data_len, shape: err_shape } => {
                assert_eq!(data_len, 3);
                assert_eq!(err_shape, vec![2, 2]);
            }
            _ => panic!("Expected TensorCreationError"),
        }
    }

    #[test]
    fn test_tensor_equality() {
        let data1 = vec![1.0_f32, 2.0];
        let shape1 = vec![2];
        let t1 = create_test_tensor(data1.clone(), shape1.clone());
        let t2 = create_test_tensor(data1.clone(), shape1.clone());
        let t3 = t1.clone(); // Clones Arc<RwLock>, points to same allocation
        let t4 = create_test_tensor(vec![3.0, 4.0], shape1.clone());
        let t5 = create_test_tensor(data1.clone(), vec![1, 2]);

        assert_eq!(t1, t1); // Equal to self
        assert_eq!(t1, t3); // Equal to Arc clone
        assert_ne!(t1, t2, "t1 and t2 should have different Arc<RwLock> pointers");

        assert_ne!(t1, t4); // Different data Arc pointer
        assert_ne!(t1, t5); // Different shape
    }

    #[test]
    fn test_get_element() {
        let t = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1);
        assert_eq!(t.get(&[0, 2]).unwrap(), 3);
        assert_eq!(t.get(&[1, 0]).unwrap(), 4);
        assert_eq!(t.get(&[1, 2]).unwrap(), 6);
    }

    #[test]
    fn test_get_element_out_of_bounds() {
        let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0, 2]).is_err());
        match t.get(&[0, 2]).err().unwrap() {
            NeuraRustError::IndexOutOfBounds { index, shape } => {
                assert_eq!(index, vec![0, 2]);
                assert_eq!(shape, vec![2, 2]);
            }
            _ => panic!("Expected IndexOutOfBounds"),
        }
    }

    #[test]
    fn test_get_element_wrong_ndim() {
        let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(t.get(&[0]).is_err());
        assert!(t.get(&[0, 0, 0]).is_err());
        match t.get(&[0]).err().unwrap() {
            NeuraRustError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected DimensionMismatch"),
        }
    }

    #[test]
    fn test_zeros_creation() {
        let shape = vec![2, 3];
        let t = zeros::<f64>(shape.clone()).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 6);
        for i in 0..2 { for j in 0..3 { assert_relative_eq!(t.get(&[i, j]).unwrap(), 0.0); } }
    }

    #[test]
    fn test_ones_creation() {
        let shape = vec![1, 4];
        let t = ones::<i32>(shape.clone()).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 4);
        for j in 0..4 { assert_eq!(t.get(&[0, j]).unwrap(), 1); }
    }

    #[test]
    fn test_full_creation() {
        let shape = vec![3, 1, 2];
        let fill_val = 42.5_f32;
        let t = full(shape.clone(), fill_val).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 6);
        for i in 0..3 { for j in 0..1 { for k in 0..2 { assert_relative_eq!(t.get(&[i, j, k]).unwrap(), fill_val); } } }
    }

    #[test]
    fn test_simple_slice() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = create_test_tensor(data, vec![2, 3, 4]);
        let ranges = vec![(0, 1), (0, 3), (0, 4)];
        let view = tensor.slice(&ranges).expect("Simple slice failed");

        assert_eq!(view.shape(), vec![1, 3, 4], "View shape mismatch");
        assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0, "Value mismatch at [0,0,0]");
        assert_eq!(view.get(&[0, 2, 3]).unwrap(), 11.0, "Value mismatch at [0,2,3]");
    }

    #[test]
    fn test_slice_shares_data() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = create_test_tensor(data, vec![2, 3, 4]);
        let ranges = vec![(1, 2), (1, 3), (0, 2)];
        let view = tensor.slice(&ranges).expect("Slice for data sharing test failed");

        let original_buffer_ptr = Arc::as_ptr(&tensor.borrow_data_buffer());
        let view_buffer_ptr = Arc::as_ptr(&view.borrow_data_buffer());

        assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "View does not share the same data buffer Arc");
        assert_eq!(original_buffer_ptr, view_buffer_ptr, "View does not point to the same buffer allocation");
    }

    #[test]
    fn test_slice_metadata() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = create_test_tensor(data, vec![2, 3, 4]); // Shape [2, 3, 4], Strides [12, 4, 1], Offset 0
        let ranges = vec![(1, 2), (1, 3), (0, 2)]; // Slice: [1, 1:3, 0:2] -> Shape [1, 2, 2]
        let view = tensor.slice(&ranges).expect("Slice for metadata test failed");

        let view_data = view.read_data();

        assert_eq!(view_data.shape, vec![1, 2, 2], "View shape mismatch");
        assert_eq!(view_data.strides, vec![12, 4, 1], "View strides should be inherited");
        assert_eq!(view_data.offset, 16, "View offset calculation incorrect");

        // Drop guard before calling view.get() which also locks
        drop(view_data);
        assert_eq!(view.get(&[0, 0, 0]).unwrap(), 16.0, "First element value mismatch");
    }

    #[test]
    fn test_slice_invalid_range() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let tensor = create_test_tensor(data, vec![2, 3, 4]);

        // Range end > dimension size
        let ranges_invalid_end = vec![(0, 1), (0, 4), (0, 4)]; // Dim 1 size is 3, end is 4
        let result_invalid_end = tensor.slice(&ranges_invalid_end);
        assert!(matches!(result_invalid_end, Err(NeuraRustError::SliceError { .. })), "Expected SliceError for end > dim size");

        // Range start >= end
        let ranges_invalid_start = vec![(0, 1), (2, 2), (0, 4)]; // Dim 1 start == end
        let result_invalid_start = tensor.slice(&ranges_invalid_start);
        assert!(matches!(result_invalid_start, Err(NeuraRustError::SliceError { .. })), "Expected SliceError for start >= end");

        // Incorrect number of ranges
        let ranges_wrong_ndim = vec![(0, 1), (0, 1)]; // Only 2 ranges for 3 dims
        let result_wrong_ndim = tensor.slice(&ranges_wrong_ndim);
        // Use DimensionMismatch as per the updated slice_op
        assert!(matches!(result_wrong_ndim, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for wrong number of ranges");
    }
}
