use crate::error::NeuraRustError;
use std::cmp::max;

/// Calculates the strides for a given shape.
/// Strides represent the number of elements to skip in the flattened data array
/// to move one step along each dimension.
///
/// Example:
/// shape = [2, 3] -> strides = [3, 1]
/// shape = [2, 2, 2] -> strides = [4, 2, 1]
pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Determines the output shape resulting from broadcasting two input shapes.
///
/// Follows NumPy/PyTorch broadcasting rules:
/// 1. If the shapes have different numbers of dimensions, prepend 1s to the shorter shape.
/// 2. Compare dimensions element-wise from right to left.
/// 3. Dimensions are compatible if they are equal, or one of them is 1.
/// 4. The resulting dimension size is the maximum of the two compared dimensions (if one is 1, it's the other dimension).
///
/// Returns `Ok(broadcasted_shape)` if the shapes are compatible, `Err(NeuraRustError::BroadcastError)` otherwise.
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>, NeuraRustError> {
    let rank_a = shape_a.len();
    let rank_b = shape_b.len();
    let max_rank = max(rank_a, rank_b);
    let mut result_shape = vec![0; max_rank];

    for i in 0..max_rank {
        let idx_a = rank_a.checked_sub(1 + i);
        let idx_b = rank_b.checked_sub(1 + i);

        let dim_a = idx_a.map(|idx| shape_a[idx]).unwrap_or(1);
        let dim_b = idx_b.map(|idx| shape_b[idx]).unwrap_or(1);

        if dim_a == dim_b {
            result_shape[max_rank - 1 - i] = dim_a;
        } else if dim_a == 1 {
            result_shape[max_rank - 1 - i] = dim_b;
        } else if dim_b == 1 {
            result_shape[max_rank - 1 - i] = dim_a;
        } else if dim_a == 0 || dim_b == 0 {
            // If one dimension is 0, the resulting dimension is 0.
            // Compatibility with the other dimension (if non-zero and non-one)
            // would have been checked in previous iterations or will be checked in the final else.
            result_shape[max_rank - 1 - i] = 0;
        } else {
            // If we reach here, dimensions differ and neither is 1 or 0.
            return Err(NeuraRustError::BroadcastError {
                shape1: shape_a.to_vec(),
                shape2: shape_b.to_vec(),
            });
        }
    }
    Ok(result_shape)
}

/// Converts a linear index into multi-dimensional coordinates based on shape.
/// Panics if the index is out of bounds for the given shape.
pub fn index_to_coord(index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        // Handle scalar tensor case
        assert_eq!(index, 0, "Index must be 0 for scalar tensor");
        return vec![];
    }
    let rank = shape.len();
    let mut coord = vec![0; rank];
    let mut current_index = index;

    // Check if index is out of bounds
    // Calculate numel carefully to handle shapes with 0
    let numel: usize = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim)).unwrap_or(0);
    if numel == 0 {
         if index == 0 { // Allow index 0 for empty tensors
             return coord; // Return vec![0, 0, ...] for shape like [2, 0, 3]
         } else {
             panic!("Index {} out of bounds for empty tensor with shape {:?}", index, shape);
         }
    } else if index >= numel {
         panic!("Index {} out of bounds for shape {:?} with numel {}", index, shape, numel);
    }

    for i in (0..rank).rev() {
        let dim_size = shape[i];
        if dim_size == 0 {
            // For dim size 0, coordinate must be 0. current_index remains unchanged.
            coord[i] = 0;
        } else {
             // Calculate coordinate for this dimension
             coord[i] = current_index % dim_size;
             // Update the index for the next (higher) dimension
             current_index /= dim_size;
        }
    }
    // After the loop, current_index should be 0 if the original index was valid.
    // assert_eq!(current_index, 0, "Error in index_to_coord logic");
    coord
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_strides_simple() {
        assert_eq!(calculate_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(calculate_strides(&[4, 5, 6]), vec![30, 6, 1]);
        assert_eq!(calculate_strides(&[5]), vec![1]);
        assert_eq!(calculate_strides(&[1, 5]), vec![5, 1]);
        assert_eq!(calculate_strides(&[5, 1]), vec![1, 1]);
    }

    #[test]
    fn test_calculate_strides_empty() {
        assert_eq!(calculate_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_calculate_strides_single_zero_dim() {
        assert_eq!(calculate_strides(&[0]), vec![1]);
    }

    #[test]
    fn test_calculate_strides_includes_zero_dim() {
        assert_eq!(calculate_strides(&[2, 0, 3]), vec![0, 3, 1]);
        assert_eq!(calculate_strides(&[2, 3, 0]), vec![0, 0, 1]);
    }

    #[test]
    fn test_broadcast_shapes_equal() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]), Ok(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[5], &[5]), Ok(vec![5]));
        assert_eq!(broadcast_shapes(&[], &[]), Ok(vec![]));
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        assert_eq!(broadcast_shapes(&[2, 3], &[]), Ok(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[], &[2, 3]), Ok(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[1], &[]), Ok(vec![1]));
    }

    #[test]
    fn test_broadcast_shapes_one_dimension() {
        assert_eq!(broadcast_shapes(&[4, 1], &[4, 5]), Ok(vec![4, 5]));
        assert_eq!(broadcast_shapes(&[4, 5], &[1, 5]), Ok(vec![4, 5]));
        assert_eq!(broadcast_shapes(&[4, 5], &[4, 1]), Ok(vec![4, 5]));
        assert_eq!(broadcast_shapes(&[1, 5], &[4, 5]), Ok(vec![4, 5]));
    }

    #[test]
    fn test_broadcast_shapes_prepend_ones() {
        assert_eq!(broadcast_shapes(&[4, 5], &[5]), Ok(vec![4, 5]));
        assert_eq!(broadcast_shapes(&[5], &[4, 5]), Ok(vec![4, 5]));
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[3, 1]), Ok(vec![2, 3, 4]));
        assert_eq!(broadcast_shapes(&[3, 4], &[2, 1, 4]), Ok(vec![2, 3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_complex() {
        assert_eq!(
            broadcast_shapes(&[5, 1, 4, 1], &[4, 5]),
            Ok(vec![5, 1, 4, 5])
        );
        assert_eq!(
            broadcast_shapes(&[4, 5], &[5, 1, 4, 1]),
            Ok(vec![5, 1, 4, 5])
        );
    }

    #[test]
    fn test_broadcast_shapes_with_zero() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 0]), Ok(vec![2, 0]));
        assert_eq!(broadcast_shapes(&[2, 0], &[2, 3]), Ok(vec![2, 0]));
        assert_eq!(broadcast_shapes(&[1, 0], &[5, 1]), Ok(vec![5, 0]));
        assert_eq!(broadcast_shapes(&[5, 1], &[1, 0]), Ok(vec![5, 0]));
        assert_eq!(broadcast_shapes(&[5, 0], &[5, 0]), Ok(vec![5, 0]));
        assert_eq!(broadcast_shapes(&[0], &[5]), Ok(vec![0]));
        assert_eq!(broadcast_shapes(&[5], &[0]), Ok(vec![0]));
        assert_eq!(broadcast_shapes(&[], &[0]), Ok(vec![0]));
        assert_eq!(broadcast_shapes(&[1, 0], &[5, 3]), Ok(vec![5, 0]));
        let res1 = broadcast_shapes(&[2, 3], &[2, 4]);
        assert!(matches!(res1, Err(NeuraRustError::BroadcastError { .. })));
        let res2 = broadcast_shapes(&[3, 0], &[2, 0]);
        assert!(matches!(res2, Err(NeuraRustError::BroadcastError { .. })));
        let res3 = broadcast_shapes(&[2, 0], &[5, 3]);
        assert!(matches!(res3, Err(NeuraRustError::BroadcastError { .. })));
    }
}
