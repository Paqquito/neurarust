use crate::device::StorageDevice;
use thiserror::Error; // Import StorageDevice

/// Custom error type for the NeuraRust framework.
#[derive(Error, Debug, PartialEq, Clone)] // PartialEq for easier testing, Clone added
pub enum NeuraRustError {
    #[error("Rank mismatch: expected {expected}, got {actual}")]
    RankMismatch {
        expected: usize,
        actual: usize,
    },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}, operation: {operation}")]
    ShapeMismatch {
        expected: String,
        actual: String,
        operation: String,
    },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Incompatible shapes for operation: {shape1:?} and {shape2:?}")]
    IncompatibleShapes {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    #[error("Cannot broadcast shapes: {shape1:?} and {shape2:?}")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    #[error("Slice error: {message}")]
    SliceError { message: String },

    #[error("Invalid permutation: dims {dims:?} are not a valid permutation for rank {rank}")]
    InvalidPermutation { dims: Vec<usize>, rank: usize },

    #[error("Tensor creation error: data length {data_len} does not match shape {shape:?}")]
    TensorCreationError { data_len: usize, shape: Vec<usize> },

    #[error("Operation requires tensor to require grad, but it doesn't.")]
    RequiresGradNotMet,

    #[error("Backward called on non-scalar tensor without explicit gradient.")]
    BackwardNonScalar,

    #[error("Shape mismatch during gradient accumulation: expected {expected:?}, got {actual:?}")]
    GradientAccumulationShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Division by zero error")]
    DivisionByZero,

    #[error("Cannot stack an empty list of tensors")]
    EmptyTensorList,

    #[error(
        "Data is not available on the expected device: expected {expected:?}, actual {actual:?}"
    )]
    DataNotAvailableError {
        expected: StorageDevice,
        actual: StorageDevice,
    },

    #[error("Device mismatch for operation '{operation}': expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        expected: StorageDevice,
        actual: StorageDevice,
        operation: String,
    },

    #[error("Cycle detected in the computation graph during backward pass.")]
    CycleDetected,

    #[error("Backward pass error: {0}")]
    BackwardError(String),

    #[error("Invalid axis specified: Axis {axis} is out of bounds for rank {rank}")]
    InvalidAxis {
        axis: usize,
        rank: usize,
    },

    #[error("Invalid slice specified: Slice {slice_start}..{slice_end} (step {step}) is invalid for dimension {dimension} with size {size}")]
    InvalidSlice {
        slice_start: usize,
        slice_end: usize,
        step: usize,
        dimension: usize,
        size: usize,
    },

    #[error("Buffer access error: Could not access {buffer_type} buffer. Details: {details}")]
    BufferAccessError {
        buffer_type: String, // e.g., "CPU", "GPU"
        details: String,
    },

    #[error("Locking error: Failed to acquire {lock_type} lock. Reason: {reason}")]
    LockError {
        lock_type: String, // e.g., "read", "write"
        reason: String,
    },

    #[error("Backward pass error: A cycle was detected in the computation graph.")]
    BackwardGraphCycle,

    // Add more specific errors as needed
}
