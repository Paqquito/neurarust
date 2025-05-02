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
pub(crate) fn slice_op<T: Default + Send + Sync + 'static + Debug + Copy>(
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

        // Adjust validation for slice ranges:
        // - start must be <= end
        // - end must be <= dim_size
        // - Allow start == end == 0 if dim_size == 0

        if start > end { // Only check start > end
            return Err(NeuraRustError::SliceError {
                message: format!(
                    "Invalid slice range start > end ({}:{}) for dimension {} with size {}",
                    start, end, i, dim_size
                )
            });
        }
        if end > dim_size { // Check end <= dim_size
             return Err(NeuraRustError::SliceError {
                 message: format!(
                    "Invalid slice range end > size ({}:{}) for dimension {} with size {}",
                    start, end, i, dim_size
                 )
            });
        }
        // The case start == end is allowed (results in dim size 0)
        // Special case start == end == 0 is also allowed if dim_size == 0.

        new_shape.push(end - start);
        // Only add offset if the dimension is not empty, otherwise stride is irrelevant
        if dim_size > 0 {
             new_offset += start * guard.strides[i];
        }
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

/// Performs the transpose operation between two dimensions, creating a view.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `dim1`: The first dimension to transpose.
/// * `dim2`: The second dimension to transpose.
///
/// # Returns
/// A new Tensor representing the transposed view, or an error.
pub(crate) fn transpose_op<T: Default + Send + Sync + 'static + Debug + Copy>(
    tensor: &Tensor<T>,
    dim1: usize,
    dim2: usize,
) -> Result<Tensor<T>, NeuraRustError> {
    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| NeuraRustError::InternalError("Failed to acquire read lock on TensorData for transpose".to_string()))?;

    let rank = guard.shape.len();

    // 2. Validate dimensions
    if dim1 >= rank || dim2 >= rank {
        return Err(NeuraRustError::DimensionMismatch {
            expected: rank, // Or maybe rank - 1 as max index? Let's stick to rank for now.
            actual: std::cmp::max(dim1, dim2), // The invalid dimension
        });
        // TODO: Consider a more specific error like InvalidDimensionError?
        // Sticking with DimensionMismatch for now as it conveys rank issue.
    }

    // 3. Calculate new shape and strides
    let mut new_shape = guard.shape.clone();
    let mut new_strides = guard.strides.clone();

    // Swap shape and strides at dim1 and dim2
    new_shape.swap(dim1, dim2);
    new_strides.swap(dim1, dim2);

    // 4. Get other necessary info
    let buffer_arc = Arc::clone(&guard.data); // Clone the Arc to the buffer
    let device = guard.device;
    let offset = guard.offset; // Transpose doesn't change the offset

    // Drop the read guard before creating the new RwLock
    drop(guard);

    // 5. Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape,
        new_strides, // Use the *new* swapped strides
    );

    // 6. Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td))
    };

    Ok(new_tensor)
}

/// Performs the permute operation, creating a view with reordered dimensions.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `dims`: A slice representing the desired permutation of dimensions.
///           Must contain each dimension index from 0 to rank-1 exactly once.
///
/// # Returns
/// A new Tensor representing the permuted view, or an error.
pub(crate) fn permute_op<T: Default + Send + Sync + 'static + Debug + Copy>(
    tensor: &Tensor<T>,
    dims: &[usize],
) -> Result<Tensor<T>, NeuraRustError> {
    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| NeuraRustError::InternalError("Failed to acquire read lock on TensorData for permute".to_string()))?;

    let rank = guard.shape.len();

    // Add check for scalar tensor (rank 0)
    if rank == 0 {
        // Permutation is invalid for a scalar. The number of dimensions provided (dims.len())
        // must also be 0 to match the rank. If not, it's a DimensionMismatch.
        if !dims.is_empty() {
            return Err(NeuraRustError::DimensionMismatch {
                 expected: 0, // rank
                 actual: dims.len(),
            });
        } else {
            // If dims is also empty, it's still a mismatch conceptually.
            // Trying to permute 0 dimensions with 0 dimensions doesn't fit the operation.
            // Return DimensionMismatch as the test expects.
            return Err(NeuraRustError::DimensionMismatch {
                 expected: 0, // Indicate rank mismatch conceptually
                 actual: 0,
            });
            // Previously returned UnsupportedOperation, but test expects DimensionMismatch.
        }
    }

    // 2. Validate permutation dimensions (original check, now only for rank > 0)
    if dims.len() != rank {
        return Err(NeuraRustError::DimensionMismatch {
            expected: rank,
            actual: dims.len(),
        });
    }
    let mut seen = vec![false; rank];
    for &dim in dims {
        if dim >= rank || seen[dim] {
            return Err(NeuraRustError::InvalidPermutation {
                dims: dims.to_vec(),
                rank,
            });
        }
        seen[dim] = true;
    }

    // 3. Calculate new shape and strides
    let mut new_shape = Vec::with_capacity(rank);
    let mut new_strides = Vec::with_capacity(rank);
    for &new_dim_index in dims {
        new_shape.push(guard.shape[new_dim_index]);
        new_strides.push(guard.strides[new_dim_index]);
    }

    // 4. Get other necessary info
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    drop(guard);

    // 5. Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape,
        new_strides,
    );

    // 6. Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td))
    };

    Ok(new_tensor)
}

/// Performs the reshape operation. Currently only supports creating a view
/// for contiguous tensors. For non-contiguous tensors, call `.contiguous()` first.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `new_shape_vec`: The desired new shape.
///
/// # Returns
/// A new Tensor representing the reshaped view, or an error.
pub(crate) fn reshape_op<T: Default + Send + Sync + 'static + Debug + Copy>(
    tensor: &Tensor<T>,
    new_shape_vec: Vec<usize>,
) -> Result<Tensor<T>, NeuraRustError> {
    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| NeuraRustError::InternalError("Failed to acquire read lock on TensorData for reshape".to_string()))?;

    // 2. Validate number of elements
    let original_numel: usize = guard.shape.iter().product();
    let new_numel: usize = new_shape_vec.iter().product();

    if original_numel != new_numel {
        return Err(NeuraRustError::ShapeMismatch {
            expected: guard.shape.clone(), // Keep returning shapes
            actual: new_shape_vec,
        });
    }

    // 3. Check for contiguity (current limitation)
    if !guard.is_contiguous() {
        // Offer suggestion to call .contiguous()
        return Err(NeuraRustError::UnsupportedOperation(
            "Reshape currently only supports contiguous tensors. Call .contiguous() first.".to_string()
        ));
    }

    // 4. If contiguous and numel matches, calculate new strides and create view
    let new_strides = TensorData::<T>::calculate_contiguous_strides(&new_shape_vec);

    // Get necessary info from locked data
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset; // Reshape of contiguous tensor starts at the same offset

    drop(guard);

    // Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape_vec.clone(), // Use the validated new shape
        new_strides,
    );

    // Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td))
    };

    Ok(new_tensor)
}

// Placeholder for Debug trait implementation if needed 