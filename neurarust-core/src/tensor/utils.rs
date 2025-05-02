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
/// Returns `Ok(broadcasted_shape)` if the shapes are compatible, `Err(String)` otherwise.
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>, String> {
    let rank_a = shape_a.len();
    let rank_b = shape_b.len();
    let max_rank = max(rank_a, rank_b);
    let mut result_shape = vec![0; max_rank];

    for i in 0..max_rank {
        let dim_a = shape_a
            .get(rank_a.wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        let dim_b = shape_b
            .get(rank_b.wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);

        if dim_a == dim_b {
            result_shape[max_rank - 1 - i] = dim_a;
        } else if dim_a == 1 {
            result_shape[max_rank - 1 - i] = dim_b;
        } else if dim_b == 1 {
            result_shape[max_rank - 1 - i] = dim_a;
        } else if dim_a == 0 {
            result_shape[max_rank - 1 - i] = 0;
        } else if dim_b == 0 {
            result_shape[max_rank - 1 - i] = 0;
        } else {
            return Err(format!(
                "Shapes {:?} and {:?} are not broadcastable: dimension size mismatch at index {} ({} vs {})",
                shape_a, shape_b, max_rank - 1 - i, dim_a, dim_b
            ));
        }
    }
    Ok(result_shape)
}

// Helper to convert a linear index to multi-dimensional coordinates
// TODO: Handle strides/shape containing 0 more robustly?
pub fn index_to_coord(index: usize, strides: &[usize], shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let rank = shape.len();
    let mut coord = vec![0; rank];
    let mut current_index = index;
    for i in 0..rank {
        if strides[i] == 0 {
            if shape[i] > 0 {
                // Avoid division by zero if stride is 0 but dim is not
                // This happens in shapes like [2,0,3] where stride[0] is 0
                // If index > 0, this coordinate must be 0 anyway?
                coord[i] = 0;
            } else {
                // If shape[i] is 0, coord must be 0
                coord[i] = 0;
            }
        } else {
            coord[i] = current_index / strides[i];
            current_index %= strides[i];
        }
    }
    coord
}

// Helper to get the original data index from broadcasted coordinates
pub fn coord_to_index_broadcasted(
    target_coord: &[usize],
    original_shape: &[usize],
    original_strides: &[usize],
) -> usize {
    if original_shape.is_empty() {
        return 0;
    } // Scalar
    let rank_diff = target_coord.len().saturating_sub(original_shape.len());
    let mut index = 0;
    for i in 0..original_shape.len() {
        let coord_idx = rank_diff + i;
        let dim_size = original_shape[i];
        let stride = original_strides[i];
        // If the original dimension was 1, use coord 0, otherwise use the target coord
        let effective_coord = if dim_size == 1 {
            0
        } else {
            target_coord[coord_idx]
        };
        index += effective_coord * stride;
    }
    index
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
        assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
        assert!(broadcast_shapes(&[3, 0], &[2, 0]).is_err());
    }
}
