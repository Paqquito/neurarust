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
    pub(crate) shape: Vec<usize>,
    /// The strides for each dimension.
    /// Strides define the jump in memory required to move one step along a given dimension.
    pub(crate) strides: Vec<usize>,
    /// The offset into the buffer for the first element (used for views).
    /// This indicates the starting position of this tensor's data within the shared buffer.
    pub(crate) offset: usize,

    // --- Autograd Metadata --- Keep these fields ---
    /// Flag indicating if the tensor requires gradient computation.
    /// If true, operations involving this tensor will be tracked in the computation graph.
    pub(crate) requires_grad: bool,
    /// Option holding the gradient tensor, if computed.
    /// The gradient tensor has the same shape and device as this tensor.
    /// It's typically populated during the backward pass.
    pub(crate) grad: Option<Tensor>,
    /// Option holding the backward operation function node in the computation graph.
    /// This links the tensor to the operation that produced it, enabling backpropagation.
    /// Leaf tensors (created directly by the user) have `grad_fn = None`.
    pub(crate) grad_fn: Option<Arc<dyn BackwardOp + Send + Sync>>,
}

impl TensorData {
    /// Creates a new `TensorData` instance with the given f32 data and shape on the CPU.
    ///
    /// This is the primary constructor for creating tensors from raw f32 data.
    /// It takes ownership of the data vector, calculates contiguous strides automatically,
    /// and initializes metadata (offset=0, requires_grad=false, etc.).
    ///
    /// # Arguments
    /// * `data_vec`: A `Vec<f32>` containing the tensor data in a flattened, row-major order.
    /// * `shape`: A `Vec<usize>` defining the desired tensor shape.
    ///
    /// # Errors
    /// Returns `NeuraRustError::TensorCreationError` if the length of `data_vec` does not match
    /// the total number of elements specified by `shape`.
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

    /// Creates a new `TensorData` instance with the given f64 data and shape on the CPU.
    ///
    /// Similar to `new`, but for `f64` data.
    ///
    /// # Arguments
    /// * `data_vec`: A `Vec<f64>` containing the tensor data.
    /// * `shape`: A `Vec<usize>` defining the desired tensor shape.
    ///
    /// # Errors
    /// Returns `NeuraRustError::TensorCreationError` if data length mismatches shape numel.
    pub fn new_f64(data_vec: Vec<f64>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let numel: usize = shape.iter().product();
        let data_len = data_vec.len();
        if data_len != numel {
            return Err(NeuraRustError::TensorCreationError { data_len, shape });
        }

        // Calculate strides
        let strides = calculate_strides(&shape); // Use the utility function

        // --- Create the Buffer --- Updated logic for F64
        let data_arc = Arc::new(data_vec);
        let cpu_buffer = CpuBuffer::F64(data_arc);
        let buffer = Buffer::Cpu(cpu_buffer);
        let buffer_arc = Arc::new(buffer);
        // --- End Buffer Creation ---

        Ok(TensorData {
            buffer: buffer_arc,
            device: StorageDevice::CPU, // Hardcode CPU for now
            dtype: DType::F64,          // Set DType to F64
            offset: 0,
            shape,
            strides,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        })
    }

    /// Creates a new `TensorData` representing a view of an existing buffer.
    /// (Used internally by view operations like slice, transpose, etc.)
    ///
    /// This constructor does **not** allocate new memory for the data but shares the
    /// provided `buffer_arc`. It sets new metadata (offset, shape, strides).
    /// Views created this way do not require gradients by default and have no `grad_fn`.
    ///
    /// # Arguments
    /// * `buffer_arc`: An `Arc<Buffer>` pointing to the shared data storage.
    /// * `device`: The `StorageDevice` where the view resides (must match the buffer's device).
    /// * `offset`: The starting offset within the shared buffer for this view.
    /// * `shape`: The shape of this view.
    /// * `strides`: The strides for this view.
    ///
    /// # Errors
    /// Returns `NeuraRustError::DeviceMismatch` if the provided `device` does not match the
    /// actual device of the `buffer_arc`.
    /// Returns `NeuraRustError::UnsupportedOperation` if the buffer type is unsupported (e.g., GPU currently).
    #[allow(dead_code)] // May not be used directly outside the crate yet
    pub(crate) fn new_view(
        buffer_arc: Arc<Buffer>,
        device: StorageDevice,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Result<Self, NeuraRustError> {
        // Infer DType from the buffer
        let dtype = match &*buffer_arc {
            Buffer::Cpu(CpuBuffer::F32(_)) => DType::F32,
            Buffer::Cpu(CpuBuffer::F64(_)) => DType::F64,
            Buffer::Gpu { .. } => { // Assuming GPU buffers will have associated DType later
                // For now, return an error or a default/placeholder DType if GPU supported
                return Err(NeuraRustError::UnsupportedOperation(
                    "Cannot determine DType for GPU buffer in new_view yet.".to_string()
                ));
            }
        };

        // Verify device consistency (optional but good practice)
        match (&*buffer_arc, device) {
            (Buffer::Cpu(_), StorageDevice::CPU) => { /* Ok */ }
            (Buffer::Gpu { device: buffer_device, .. }, StorageDevice::GPU) => {
                if buffer_device != &device {
                    return Err(NeuraRustError::DeviceMismatch {
                        expected: *buffer_device,
                        actual: device,
                        operation: "new_view device consistency check".to_string(),
                    });
                }
                /* Ok */
            }
            (buffer, view_device) => {
                // Mismatch between buffer location and specified device for the view
                let actual_device = match buffer {
                    Buffer::Cpu(_) => StorageDevice::CPU,
                    Buffer::Gpu { device, .. } => *device,
                };
                 return Err(NeuraRustError::DeviceMismatch {
                    expected: actual_device, // The buffer's actual device
                    actual: view_device, // The device specified for the view
                    operation: "new_view buffer/device mismatch".to_string(),
                 });
            }
        }

        Ok(TensorData {
            buffer: buffer_arc, // Share the buffer Arc
            device,             // Use the provided device
            dtype,              // Use inferred dtype
            offset,
            shape,
            strides,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        })
    }

    /// Provides immutable access to the underlying shared data buffer (`Arc<Buffer>`).
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Calculates the strides required for a contiguous tensor of the given shape.
    /// This is a static utility function.
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
