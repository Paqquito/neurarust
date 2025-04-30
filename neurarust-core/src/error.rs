use thiserror::Error;

/// Custom error type for the NeuraRust framework.
#[derive(Error, Debug, PartialEq, Clone)] // PartialEq for easier testing, Clone added
pub enum NeuraRustError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },

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

    #[error("Tensor creation error: data length {data_len} does not match shape {shape:?}")]
    TensorCreationError {
        data_len: usize,
        shape: Vec<usize>,
    },

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

    // Add more specific errors as needed
} 