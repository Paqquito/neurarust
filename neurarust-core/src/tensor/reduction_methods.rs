use crate::{tensor::Tensor, error::NeuraRustError};

impl Tensor {
    /// Computes the mean of the tensor elements over given axes.
    /// Delegates to `ops::reduction::mean::mean_op`.
    pub fn mean(&self, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
        // Call the mean_op function from the ops module
        crate::ops::reduction::mean::mean_op(self, axes, keep_dims)
    }

    /// Computes the maximum of the tensor elements over given axes.
    /// Delegates to `ops::reduction::max::max_op`.
    pub fn max(&self, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
        // Call the max_op function from the ops module
        crate::ops::reduction::max::max_op(self, axes, keep_dims)
    }
    
    // TODO: Add sum, min methods here similarly?
} 