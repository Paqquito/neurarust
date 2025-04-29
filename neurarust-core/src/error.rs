use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum NeuraRustError {
    #[error("Shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },

    #[error("Incompatible shapes for broadcasting: {shape1:?} and {shape2:?}")]
    BroadcastError { shape1: Vec<usize>, shape2: Vec<usize> },

    #[error("Index out of bounds: index {index:?} is out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds { index: usize, dim: usize, size: usize },

    #[error("Slice out of bounds: range {start}..{end} is out of bounds for dimension {dim} with size {size}")]
    SliceOutOfBounds { start: usize, end: usize, dim: usize, size: usize },

    #[error("Invalid number of dimensions for operation: expected {expected}, found {found}")]
    InvalidDimensions { expected: usize, found: usize },

    #[error("Operation requires tensor to require grad, but it does not")]
    RequiresGradError,

    #[error("Cannot perform backward on non-scalar tensor without providing gradient")]
    BackwardNonScalarError,

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Negative dimension size is not allowed")]
    NegativeDimension,

    #[error("Feature not implemented yet: {feature}")]
    NotImplemented { feature: String },

    #[error("Internal error: {message}")] // For unexpected logic errors
    InternalError { message: String },
} 