use crate::error::NeuraRustError;

/// Validates the dimensions provided for a transpose operation.
///
/// Checks if `dim1` and `dim2` are within the valid range `[0, rank)`. Allows
/// transposing a dimension with itself (which results in a no-op view).
///
/// # Arguments
/// * `rank` - The number of dimensions of the tensor.
/// * `dim1` - The first dimension index.
/// * `dim2` - The second dimension index.
///
/// # Returns
/// `Ok(())` if dimensions are valid, `Err(NeuraRustError)` otherwise.
pub fn validate_transpose_dims(rank: usize, dim1: usize, dim2: usize) -> Result<(), NeuraRustError> {
    if dim1 >= rank || dim2 >= rank {
        Err(NeuraRustError::IndexOutOfBounds {
            // Indicate the problematic dimension index
            index: vec![dim1.max(dim2)],
            shape: vec![rank], // Represent shape as just the rank for this context
        })
    } else if dim1 == dim2 {
         // Allow transposing the same dimension (no-op view)
         Ok(())
         // Or return error:
         // Err(NeuraRustError::InvalidAxis { axis: dim1, rank }) // If transposing same dim is invalid
    } else {
        Ok(())
    }
}

/// Validates if the given slice `dims` represents a valid permutation of axes `0..rank`.
///
/// Checks if `dims` has length `rank` and contains each number from `0` to `rank-1`
/// exactly once.
///
/// # Arguments
/// * `rank` - The expected rank (number of dimensions).
/// * `dims` - The slice representing the permutation.
///
/// # Returns
/// `Ok(())` if `dims` is a valid permutation, `Err(NeuraRustError)` otherwise.
pub fn validate_permutation(rank: usize, dims: &[usize]) -> Result<(), NeuraRustError> {
    if dims.len() != rank {
        return Err(NeuraRustError::RankMismatch {
            expected: rank,
            actual: dims.len(),
        });
    }
    let mut seen = vec![false; rank];
    for &axis in dims {
        if axis >= rank {
            return Err(NeuraRustError::IndexOutOfBounds {
                index: vec![axis],
                shape: vec![rank], // Represent shape as rank
            });
        }
        if seen[axis] {
            return Err(NeuraRustError::InvalidPermutation {
                dims: dims.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

/// Calculates the new shape of a tensor after applying a permutation.
///
/// # Arguments
/// * `shape` - The original shape of the tensor.
/// * `dims` - The permutation slice (validated by `validate_permutation`). `dims[i]`
///   indicates the original dimension that moves to the new dimension `i`.
///
/// # Returns
/// A `Vec<usize>` representing the new shape.
pub fn permute_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    dims.iter().map(|&axis| shape[axis]).collect()
}

/// Calculates the new strides of a tensor after applying a permutation.
///
/// # Arguments
/// * `strides` - The original strides of the tensor.
/// * `dims` - The permutation slice (validated by `validate_permutation`). `dims[i]`
///   indicates the original dimension that moves to the new dimension `i`.
///
/// # Returns
/// A `Vec<usize>` representing the new strides.
pub fn permute_strides(strides: &[usize], dims: &[usize]) -> Vec<usize> {
    dims.iter().map(|&axis| strides[axis]).collect()
}

/// Normalizes a potentially negative index relative to a dimension size.
///
/// Converts an `isize` index (which can be negative to count from the end)
/// into a `usize` index within the valid range `[0, dim_size]` (inclusive end).
/// Indices out of bounds are clamped to `0` or `dim_size`.
///
/// # Arguments
/// * `idx` - The potentially negative index.
/// * `dim_size` - The size of the dimension.
///
/// # Returns
/// The normalized `usize` index, clamped to `[0, dim_size]`.
fn normalize_idx(idx: isize, dim_size: usize) -> usize {
    if idx >= 0 {
        std::cmp::min(idx as usize, dim_size) // Clamp positive index to dim_size
    } else {
        // Negative index: calculate from the end
        let abs_idx = idx.abs() as usize;
        if abs_idx > dim_size {
            0 // Clamp out-of-bounds negative index to 0
        } else {
            dim_size - abs_idx
        }
    }
}

// Import SliceArg and SliceRange from the parent module
use super::slice::{SliceArg, SliceRange};

/// Converts a `SliceArg::Slice` into a validated internal `SliceRange`.
///
/// This function takes the `start`, `end`, and `step` from a `SliceArg::Slice`,
/// normalizes negative indices using `normalize_idx`, validates the step,
/// and calculates the resulting size of the sliced dimension.
///
/// **Note:** This function currently only supports positive steps and returns errors
/// for `SliceArg::Index`, `Ellipsis`, and `NewAxis`, reflecting the limitations
/// in the main `slice_op` implementation.
///
/// # Arguments
/// * `slice` - The `SliceArg` to normalize (must be `SliceArg::Slice`).
/// * `dim_size` - The size of the dimension being sliced.
///
/// # Returns
/// A `Result` containing the normalized `SliceRange` or a `NeuraRustError` if
/// the step is invalid or the `SliceArg` variant is unsupported.
pub(crate) fn normalize_slice(
    slice: SliceArg,
    dim_size: usize,
) -> Result<SliceRange, NeuraRustError> {
    match slice {
        SliceArg::Slice(start, end, step) => {
            if step == 0 {
                return Err(NeuraRustError::SliceError {
                    message: "Step cannot be zero".to_string(),
                });
            }
            if step < 0 {
                // TODO: Implement negative step support later if needed
                return Err(NeuraRustError::UnsupportedOperation(
                    "Negative step in slice not yet supported".to_string(),
                ));
            }

            let step_usize = step as usize;
            let start_norm = normalize_idx(start, dim_size);
            let end_norm = normalize_idx(end, dim_size);

            // Calculate size based on normalized indices *before* potential swap
            let size = if start_norm >= end_norm {
                0
            } else {
                 // Use normalized end directly, clamped to dim_size
                 let clamped_end = std::cmp::min(end_norm, dim_size);
                 (clamped_end - start_norm + step_usize - 1) / step_usize
            };

            Ok(SliceRange {
                start: start_norm, // Use the original normalized start
                step: step_usize,
                size, // Use the size calculated before swap logic
            })
        }
        SliceArg::Index(_) => Err(NeuraRustError::UnsupportedOperation(
            "SliceArg::Index not yet supported by slice_op".to_string(),
        )),
        SliceArg::Ellipsis => Err(NeuraRustError::UnsupportedOperation(
            "SliceArg::Ellipsis not yet supported by slice_op".to_string(),
        )),
        SliceArg::NewAxis => Err(NeuraRustError::UnsupportedOperation(
            "SliceArg::NewAxis not yet supported by slice_op".to_string(),
        )),
    }
} 