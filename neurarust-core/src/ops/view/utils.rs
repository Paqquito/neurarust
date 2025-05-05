use crate::error::NeuraRustError;

/// Validates dimensions for transpose operation.
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

/// Validates permutation axes.
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

/// Calculates the new shape after permutation.
pub fn permute_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    dims.iter().map(|&axis| shape[axis]).collect()
}

/// Calculates the new strides after permutation.
pub fn permute_strides(strides: &[usize], dims: &[usize]) -> Vec<usize> {
    dims.iter().map(|&axis| strides[axis]).collect()
}

/// Normalizes a potentially negative index relative to a dimension size.
/// Clamps the result to the valid range [0, dim_size].
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

/// Takes a SliceArg and dimension size, returns a validated SliceRange.
/// Handles negative indices and clamps ranges.
/// Currently only supports positive steps.
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