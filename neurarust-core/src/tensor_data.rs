// src/tensor_data.rs
use std::fmt::{Debug}; // Removed Formatter, Result as FmtResult
// Use Arc for the buffer for thread-safe sharing
use std::sync::Arc;

// Import new types
use crate::buffer::Buffer;
use crate::device::StorageDevice;
use crate::error::NeuraRustError; // Import error type

/// Holds the actual data buffer reference and metadata for a tensor.
/// Uses Arc<Buffer<T>> for shared ownership of the data buffer.
/// The Tensor struct itself will wrap this in an Arc<RwLock<...>>
/// for thread-safe interior mutability of metadata (like shape, strides for views).
#[derive(Debug, PartialEq, Eq)] // Keep derive (PartialEq/Eq work due to Buffer impl)
pub struct TensorData<T> {
    // Data buffer. Shared via Arc. Views will clone this Arc.
    pub data: Arc<Buffer<T>>,
    // The device where the data buffer resides.
    pub device: StorageDevice,
    // The starting offset of this tensor's view within the shared data buffer.
    pub offset: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> TensorData<T> {
    // Public constructor - Takes ownership of data vec, wraps in CPU Buffer and Arc,
    // calculates contiguous strides. Assumes CPU for now.
    pub fn new(data_vec: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let numel: usize = shape.iter().product();
        let data_len = data_vec.len();
        if data_len != numel {
            return Err(NeuraRustError::TensorCreationError {
                 data_len,
                 shape,
             });
        }
        let strides = Self::calculate_contiguous_strides(&shape);
        // Create a CPU buffer and wrap it in Arc
        let buffer = Buffer::new_cpu(data_vec);
        Ok(TensorData {
            // Wrap buffer in Arc
            data: Arc::new(buffer),
            device: StorageDevice::CPU, // Default to CPU
            offset: 0, // New tensors always start at offset 0
            shape,
            strides,
        })
    }

    // Constructor for creating views (internal/advanced usage later)
    // Takes an existing Arc<Buffer>, offset, shape, strides, device
    // Note: Needs careful validation externally that shape/strides/offset are valid for the buffer
    #[allow(dead_code)] // Might be used later for views
    pub(crate) fn new_view(
        buffer_arc: Arc<Buffer<T>>,
        device: StorageDevice,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        // Assert that the provided device matches the buffer's device
        // This prevents creating inconsistent TensorData
        assert_eq!(buffer_arc.device(), device, "Device mismatch between provided buffer and device parameter");
        TensorData {
            data: buffer_arc,
            device,
            offset,
            shape,
            strides,
        }
    }

    // Calculates strides for a contiguous tensor
    pub fn calculate_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        if shape.is_empty() { return strides; } // Handle empty shape (scalar)

        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // Helper to get number of elements, used internally
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculates the linear offset into the *underlying shared data buffer* for given multi-dimensional indices.
    /// Handles strides and the tensor's own offset correctly.
    /// Panics if the number of indices doesn't match the tensor rank or if any index is out of bounds.
    /// NOTE: This returns the logical offset. Accessing data requires checking the buffer type (CPU/GPU).
    pub fn get_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(),
                   "Number of indices ({}) does not match tensor rank ({}) for shape {:?}",
                   indices.len(), self.shape.len(), self.shape);

        let mut relative_offset = 0;
        for i in 0..self.shape.len() {
            assert!(indices[i] < self.shape[i],
                    "Index {} is out of bounds for dimension {} with size {} (shape: {:?})",
                    indices[i], i, self.shape[i], self.shape);
            relative_offset += indices[i] * self.strides[i];
        }
        // Add the base offset of this tensor view
        self.offset + relative_offset
    }
} 