// src/tensor_data.rs
use std::fmt::Debug; // Removed unnecessary braces
                     // Use Arc for the buffer for thread-safe sharing
use std::sync::Arc; // Removed unused RwLock import

// Import new types
use crate::autograd::BackwardOp; // Import BackwardOp
use crate::buffer::{Buffer, CpuBuffer}; // Adjusted import
use crate::device::StorageDevice;
use crate::error::NeuraRustError; // Import error type
use crate::tensor::Tensor; // Import Tensor for Option<Tensor<T>>
use crate::types::DType; // Import DType
use crate::tensor::utils::calculate_strides; // Keep this utility import

/// Internal storage and metadata for a Tensor.
///
/// This struct holds the actual data buffer, shape, strides, device,
/// data type, and autograd-related information.
/// It is typically wrapped in `Arc<RwLock<TensorData>>` by the `Tensor` struct
/// to allow shared ownership and interior mutability.
#[derive(Debug)] // Keep Debug, remove Clone as Buffer might not be easily cloneable later
pub struct TensorData {
    /// The underlying data buffer (CPU, GPU, etc.) holding typed data.
    /// Wrapped in Arc for cheap cloning (sharing the buffer itself, e.g., for views).
    pub(crate) buffer: Arc<Buffer>,
    /// The device where the buffer resides.
    pub(crate) device: StorageDevice,
    /// The data type of the elements in the buffer.
    pub(crate) dtype: DType,

    // --- Metadata --- Keep these fields ---
    /// The shape (dimensions) of the tensor.
    pub shape: Vec<usize>,
    /// The strides for each dimension.
    pub strides: Vec<usize>,
    /// The offset into the buffer for the first element (used for views).
    pub offset: usize,

    // --- Autograd Metadata --- Keep these fields ---
    /// Flag indicating if the tensor requires gradient computation.
    pub requires_grad: bool,
    /// Option holding the gradient tensor, if computed.
    /// Must be on the same device as the data buffer.
    pub grad: Option<Tensor>,
    /// Option holding the backward operation function node in the computation graph.
    pub grad_fn: Option<Arc<dyn BackwardOp + Send + Sync>>,
}

impl TensorData {
    /// Public constructor for creating a new TensorData instance with f32 data on CPU.
    ///
    /// Takes ownership of the data vector, calculates contiguous strides,
    /// and initializes metadata.
    pub fn new(data_vec: Vec<f32>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let numel: usize = shape.iter().product();
        let data_len = data_vec.len();
        if data_len != numel {
            return Err(NeuraRustError::TensorCreationError { data_len, shape });
        }

        // Calculate strides
        let strides = calculate_strides(&shape); // Use the utility function

        // --- Create the Buffer --- Updated logic
        // 1. Wrap the data Vec in an Arc for potential sharing
        let data_arc = Arc::new(data_vec);
        // 2. Create the specific CPU buffer variant
        let cpu_buffer = CpuBuffer::F32(data_arc);
        // 3. Create the general Buffer enum variant
        let buffer = Buffer::Cpu(cpu_buffer);
        // 4. Wrap the Buffer enum in an Arc for sharing the Buffer structure (e.g., by views)
        let buffer_arc = Arc::new(buffer);
        // --- End Buffer Creation ---

        Ok(TensorData {
            buffer: buffer_arc,
            device: StorageDevice::CPU, // Hardcode CPU for now
            dtype: DType::F32,          // Hardcode F32 for now
            offset: 0,
            shape,
            strides,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        })
    }

    /// Constructor for creating views (internal usage).
    ///
    /// Takes an existing shared buffer (`Arc<Buffer>`), offset, shape, strides, and device.
    /// Assumes the dtype is the same as the source buffer (F32 for now).
    /// Views do not require gradients by default.
    #[allow(dead_code)]
    pub(crate) fn new_view(
        buffer_arc: Arc<Buffer>, // Changed from Arc<Buffer<T>>
        device: StorageDevice,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        // TODO: Later, assert that the provided device matches the actual device
        //       stored within the buffer_arc, once we implement accessors for it.
        // assert_eq!(buffer_arc.device(), device, ...);

        TensorData {
            buffer: buffer_arc, // Share the buffer Arc
            device,             // Use the provided device
            // TODO: Later, get dtype from the buffer_arc instead of hardcoding.
            dtype: DType::F32,  // Assume F32 for views for now
            offset,
            shape,
            strides,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        }
    }

    /// Provides immutable access to the underlying shared data buffer.
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    // Keep calculate_contiguous_strides as a static/associated function
    // It doesn't depend on T or the buffer type
    pub fn calculate_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        if shape.is_empty() {
            return strides;
        }
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // Keep numel, it only depends on shape
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculates the linear offset into the *underlying shared data buffer* for given multi-dimensional indices.
    /// Handles strides and the tensor's own offset correctly.
    /// Panics if the number of indices doesn't match the tensor rank or if any index is out of bounds.
    /// NOTE: This returns the logical offset. Accessing data requires checking the buffer type (CPU/GPU).
    pub fn get_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Number of indices ({}) does not match tensor rank ({}) for shape {:?}",
            indices.len(),
            self.shape.len(),
            self.shape
        );

        let mut relative_offset = 0;
        for i in 0..self.shape.len() {
            assert!(
                indices[i] < self.shape[i],
                "Index {} is out of bounds for dimension {} with size {} (shape: {:?})",
                indices[i],
                i,
                self.shape[i],
                self.shape
            );
            relative_offset += indices[i] * self.strides[i];
        }
        self.offset + relative_offset
    }

    /// Checks if the tensor is contiguous in memory.
    /// A tensor is contiguous if its elements are laid out in the standard
    /// row-major order (C order) without gaps, considering its strides.
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut current_stride = 1;
        for i in (0..self.shape.len()).rev() {
            let shape_i = self.shape[i];
            if shape_i == 0 {
                return true;
            }
            if shape_i != 1 {
                if self.strides[i] != current_stride {
                    return false;
                }
                current_stride *= shape_i;
            }
        }
        true
    }
}
