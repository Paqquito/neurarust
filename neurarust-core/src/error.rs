use crate::device::StorageDevice;
use crate::types::DType; // Import DType
use thiserror::Error; // Import StorageDevice

/// Custom error type for the NeuraRust framework.
///
/// This enum encompasses all potential errors that can occur during tensor creation,
/// manipulation, autograd computation, and other framework operations.
#[derive(Error, Debug, PartialEq, Clone)] // PartialEq for easier testing, Clone added
pub enum NeuraRustError {
    /// Error indicating a mismatch in tensor ranks (number of dimensions).
    #[error("Rank mismatch: expected {expected}, got {actual}")]
    RankMismatch {
        /// Expected rank.
        expected: usize,
        /// Actual rank found.
        actual: usize,
    },

    /// Error indicating a mismatch in tensor shapes for a specific operation.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}, operation: {operation}")]
    ShapeMismatch {
        /// Expected shape description (can be specific like `[2, 3]` or general like `matching`).
        expected: String,
        /// Actual shape found.
        actual: String,
        /// The name of the operation where the mismatch occurred.
        operation: String,
    },

    /// Error indicating a mismatch in the size of a specific dimension.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { 
        /// Expected dimension size.
        expected: usize, 
        /// Actual dimension size found.
        actual: usize 
    },

    /// Error indicating that two shapes cannot be made compatible for an operation (e.g., element-wise ops before broadcasting).
    #[error("Incompatible shapes for operation: {shape1:?} and {shape2:?}")]
    IncompatibleShapes {
        /// Shape of the first tensor.
        shape1: Vec<usize>,
        /// Shape of the second tensor.
        shape2: Vec<usize>,
    },

    /// Error indicating that two shapes cannot be broadcast together according to broadcasting rules.
    #[error("Cannot broadcast shapes: {shape1:?} and {shape2:?}")]
    BroadcastError {
        /// Shape of the first tensor.
        shape1: Vec<usize>,
        /// Shape of the second tensor.
        shape2: Vec<usize>,
    },

    /// Error indicating that an index provided is outside the valid bounds of a tensor's shape.
    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        /// The index that caused the error.
        index: Vec<usize>,
        /// The shape of the tensor being indexed.
        shape: Vec<usize>,
    },

    /// Generic error related to tensor slicing operations.
    #[error("Slice error: {message}")]
    SliceError { 
        /// Detailed message about the slicing error.
        message: String 
    },

    /// Error indicating an invalid permutation of axes for a given tensor rank.
    #[error("Invalid permutation: dims {dims:?} are not a valid permutation for rank {rank}")]
    InvalidPermutation { 
        /// The invalid dimension permutation provided.
        dims: Vec<usize>, 
        /// The rank of the tensor.
        rank: usize 
    },

    /// Error during tensor creation due to mismatch between data length and shape volume.
    #[error("Tensor creation error: data length {data_len} does not match shape {shape:?}")]
    TensorCreationError { 
        /// Length of the provided data buffer.
        data_len: usize, 
        /// The target shape for the tensor.
        shape: Vec<usize> 
    },

    /// Error indicating an operation requiring gradient tracking was called on a tensor where it's disabled.
    #[error("Operation requires tensor to require grad, but it doesn't.")]
    RequiresGradNotMet,

    /// Error when `backward()` is called on a non-scalar tensor without providing an initial gradient.
    #[error("Backward called on non-scalar tensor without explicit gradient.")]
    BackwardNonScalar,

    /// Error during backward pass when accumulating gradients into a tensor with an incompatible shape.
    #[error("Shape mismatch during gradient accumulation: expected {expected:?}, got {actual:?}")]
    GradientAccumulationShapeMismatch {
        /// Expected shape for the gradient.
        expected: Vec<usize>,
        /// Actual shape of the gradient being accumulated.
        actual: Vec<usize>,
    },

    /// Error for operations that are not supported under the current configuration or data types.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Generic internal error, indicating a potential bug or unexpected state within the framework.
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Error indicating an attempt to divide by zero.
    #[error("Division by zero error")]
    DivisionByZero,

    /// Error when trying to stack an empty list of tensors.
    #[error("Cannot stack an empty list of tensors")]
    EmptyTensorList,

    /// Error indicating that tensor data is not available on the expected storage device.
    #[error(
        "Data is not available on the expected device: expected {expected:?}, actual {actual:?}"
    )]
    DataNotAvailableError {
        /// The device where the data was expected.
        expected: StorageDevice,
        /// The device where the data actually resides.
        actual: StorageDevice,
    },

    /// Error when an operation requires tensors to be on the same device, but they are not.
    #[error("Device mismatch for operation '{operation}': expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        /// The expected device (e.g., the device of the first tensor).
        expected: StorageDevice,
        /// The actual device of the other tensor.
        actual: StorageDevice,
        /// Name of the operation where the mismatch occurred.
        operation: String,
    },

    /// Error indicating a cycle was detected in the computation graph, preventing backward pass.
    #[error("Cycle detected in the computation graph during backward pass.")]
    CycleDetected,

    /// Generic error occurring during the backward pass of autograd.
    #[error("Backward pass error: {0}")]
    BackwardError(String),

    /// Error indicating an invalid axis was provided for an operation (e.g., reduction, transpose).
    #[error("Invalid axis specified: Axis {axis} is out of bounds for rank {rank}")]
    InvalidAxis {
        /// The invalid axis index.
        axis: usize,
        /// The rank of the tensor.
        rank: usize,
    },
    
    /// Error indicating that the dimensions provided for a transpose operation are invalid.
    /// E.g. out of bounds or the same dimension specified twice.
    #[error("Invalid dimension specified for transpose: dim {dim} is invalid for rank {rank}")]
    InvalidDimension {
        /// The invalid dimension index.
        dim: usize,
        /// The rank of the tensor.
        rank: usize
    },

    /// Error indicating an invalid slice range for a given dimension size.
    #[error("Invalid slice specified: Slice {slice_start}..{slice_end} (step {step}) is invalid for dimension {dimension} with size {size}")]
    InvalidSlice {
        /// Start index of the slice.
        slice_start: usize,
        /// End index (exclusive) of the slice.
        slice_end: usize,
        /// Step of the slice.
        step: usize,
        /// Dimension index being sliced.
        dimension: usize,
        /// Size of the dimension being sliced.
        size: usize,
    },

    /// Error related to accessing the underlying data buffer of a tensor.
    #[error("Buffer access error: Could not access {buffer_type} buffer. Details: {details}")]
    BufferAccessError {
        /// Type of buffer being accessed (e.g., "CPU", "GPU").
        buffer_type: String,
        /// Specific details about the access error.
        details: String,
    },

    /// Error indicating a failure to acquire a read or write lock on tensor data.
    #[error("Locking error: Failed to acquire {lock_type} lock. Reason: {reason}")]
    LockError {
        /// Type of lock attempted ("read" or "write").
        lock_type: String,
        /// Reason for the lock failure (e.g., "poisoned").
        reason: String,
    },

    /// Error indicating a cycle was detected specifically during graph traversal for backward pass.
    /// (Note: Potentially redundant with `CycleDetected`, could be merged or refined).
    #[error("Backward pass error: A cycle was detected in the computation graph.")]
    BackwardGraphCycle,

    /// Error indicating a mismatch in expected vs actual data types (`DType`) for an operation.
    #[error("Data type mismatch for operation '{operation}': expected {expected:?}, got {actual:?}")]
    DataTypeMismatch {
        /// The expected data type.
        expected: DType,
        /// The actual data type found.
        actual: DType,
        /// Name of the operation where the mismatch occurred.
        operation: String,
    },

    /// Error indicating an attempt to perform an in-place operation that is not allowed,
    /// typically because the tensor requires gradients.
    #[error("In-place modification error during {operation}: {reason}")]
    InplaceModificationError {
        operation: String,
        reason: String,
    },

    /// Error indicating that an operation is attempted on an unsupported or unavailable device.
    #[error("Operation {operation} unsupported or unavailable on device {device:?}")]
    UnsupportedDevice {
        device: StorageDevice,
        operation: String,
    },

    /// Error indicating an operation (like in-place modification) cannot be performed
    /// because the underlying data buffer is shared by multiple tensors (views).
    #[error("Operation {operation} failed because the buffer is shared (e.g., by views).")]
    BufferSharedError {
        operation: String,
    },

    /// Error for arithmetic errors like division by zero.
    #[error("{0}")]
    ArithmeticError(String),

    // Add more specific errors as needed
}
