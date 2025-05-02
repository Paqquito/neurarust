// neurarust-core/src/ops/view_ops.rs

use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData; // Needed for TensorData::new_view
// use crate::buffer::Buffer;          // Not directly used in this file
// use crate::device::StorageDevice;  // Not directly used in this file
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

// Define a type alias or struct for slice arguments for clarity.
// Using a tuple (start, end) for each dimension for now.
// Excludes the 'step' for simplicity initially.
pub type SliceArg = (usize, usize);

/// Performs the slicing operation, creating a view.
///
/// # Arguments
/// * `tensor`: The input tensor to slice.
/// * `ranges`: A slice of tuples `(start, end)` defining the slice for each dimension.
///             The length must match the tensor's rank. `end` is exclusive.
///
/// # Returns
/// A new Tensor representing the view, or an error.
pub(crate) fn slice_op<T: Clone + Debug + Default + Send + Sync + 'static>(
    tensor: &Tensor<T>,
    ranges: &[SliceArg],
) -> Result<Tensor<T>, NeuraRustError> {
    // Access the pub(crate) field directly
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| NeuraRustError::InternalError("Failed to acquire read lock on TensorData for slicing".to_string()))?;

    if ranges.len() != guard.shape.len() {
        return Err(NeuraRustError::DimensionMismatch {
            expected: guard.shape.len(),
            actual: ranges.len(),
        });
    }

    let mut new_shape = Vec::with_capacity(guard.shape.len());
    let mut new_offset = guard.offset;

    for (i, &(start, end)) in ranges.iter().enumerate() {
        let dim_size = guard.shape[i];
        // Use SliceError for invalid ranges within a dimension
        if start >= end {
            return Err(NeuraRustError::SliceError {
                message: format!(
                    "Invalid slice range start >= end ({}:{}) for dimension {} with size {}",
                    start, end, i, dim_size
                )
            });
        }
        if end > dim_size {
             return Err(NeuraRustError::SliceError {
                 message: format!(
                    "Invalid slice range end > size ({}:{}) for dimension {} with size {}",
                    start, end, i, dim_size
                 )
            });
        }
        new_shape.push(end - start);
        new_offset += start * guard.strides[i];
    }

    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let strides = guard.strides.clone();

    // Drop the read guard before creating the new RwLock to avoid potential deadlocks
    // if TensorData::new_view were to lock something itself (unlikely here, but good practice).
    drop(guard);

    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        new_offset,
        new_shape,
        strides,
    );

    // Construct the Tensor directly using its field
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td))
    };

    Ok(new_tensor)
}

// Placeholder for Debug trait implementation if needed 