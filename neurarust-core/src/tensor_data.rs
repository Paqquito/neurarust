// src/tensor_data.rs
use std::fmt::Debug; // Removed unnecessary braces
// Use Arc for the buffer for thread-safe sharing
use std::sync::Arc; // Removed unused RwLock import

// Import new types
use crate::autograd::BackwardOp; // Import BackwardOp
use crate::buffer::Buffer;
use crate::device::StorageDevice;
use crate::error::NeuraRustError; // Import error type
use crate::tensor::Tensor; // Import Tensor for Option<Tensor<T>>

/// Holds the actual data buffer reference and metadata for a tensor.
/// Uses Arc<Buffer<T>> for shared ownership of the data buffer.
/// The Tensor struct itself will wrap this in an Arc<RwLock<...>>
/// for thread-safe interior mutability of metadata (like shape, strides, grad info).
// Note: Removed PartialEq/Eq derive. Comparing grad_fn (trait object) is complex and not usually needed.
// Equality might be redefined later if specific comparison logic is required.
#[derive(Debug)]
pub struct TensorData<T: 'static + Debug + Copy> { // Add T bounds needed by BackwardOp
    // Data buffer. Shared via Arc. Views will clone this Arc.
    pub data: Arc<Buffer<T>>,
    // The device where the data buffer resides.
    pub device: StorageDevice,
    // The starting offset of this tensor's view within the shared data buffer.
    pub offset: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,

    // --- Autograd Fields ---
    /// Does this tensor require gradient computation?
    pub requires_grad: bool,
    /// Stores the computed gradient for this tensor after backward().
    /// Must reside on the same device as the tensor data.
    pub grad: Option<Tensor<T>>,
    /// Reference to the backward operation that produced this tensor.
    /// This forms the edge in the computation graph.
    /// Uses Arc for shared ownership across potential multiple outputs or graph references.
    pub grad_fn: Option<Arc<dyn BackwardOp<T> + Send + Sync>>,
}

// Note: Added T bounds here as well
impl<T: 'static + Debug + Copy> TensorData<T> {
    // Public constructor - Takes ownership of data vec, wraps in CPU Buffer and Arc,
    // calculates contiguous strides. Assumes CPU for now.
    // Initializes autograd fields to default (non-differentiable).
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
            // Initialize autograd fields
            requires_grad: false,
            grad: None,
            grad_fn: None,
        })
    }

    // Constructor for creating views (internal/advanced usage later)
    // Takes an existing Arc<Buffer>, offset, shape, strides, device
    // Views created this way DO NOT track gradients initially.
    // If a view needs to be part of the graph, the operation creating it
    // (e.g., slice_op) will set requires_grad and grad_fn later.
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
            // Initialize autograd fields (views don't require grad by default)
            requires_grad: false,
            grad: None,
            grad_fn: None,
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

    /// Checks if the tensor is contiguous in memory.
    /// A tensor is contiguous if its elements are laid out in the standard
    /// row-major order (C order) without gaps, considering its strides.
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() { return true; } // Scalar is contiguous

        // Check if the strides match the standard C-contiguous strides
        // Comment out unused variable
        // let expected_strides = Self::calculate_contiguous_strides(&self.shape);

        // More robust check considering dimensions of size 1:
        let mut current_stride = 1;
        for i in (0..self.shape.len()).rev() {
            let shape_i = self.shape[i];
            if shape_i == 0 { return true; } // Tensor with 0 elements is contiguous
            if shape_i != 1 {
                if self.strides[i] != current_stride {
                    return false;
                }
                current_stride *= shape_i;
            }
            // If shape_i is 1, its stride doesn't break contiguity, just continue.
        }
        true
    }
} 