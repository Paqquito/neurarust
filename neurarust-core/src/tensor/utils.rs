use std::cmp::max;
use crate::tensor::Tensor;
use std::ops::{AddAssign, Add, Neg};
use num_traits::{Zero, One};
use std::iter::Sum;
use std::fmt::Debug;
use std::cell::RefCell;
use std::rc::Weak;

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
        let dim_a = shape_a.get(rank_a.wrapping_sub(1 + i)).copied().unwrap_or(1);
        let dim_b = shape_b.get(rank_b.wrapping_sub(1 + i)).copied().unwrap_or(1);

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
    if shape.is_empty() { return vec![]; }
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
pub fn coord_to_index_broadcasted(target_coord: &[usize], original_shape: &[usize], original_strides: &[usize]) -> usize {
    if original_shape.is_empty() { return 0; } // Scalar
    let rank_diff = target_coord.len().saturating_sub(original_shape.len());
    let mut index = 0;
    for i in 0..original_shape.len() {
        let coord_idx = rank_diff + i;
        let dim_size = original_shape[i];
        let stride = original_strides[i];
        // If the original dimension was 1, use coord 0, otherwise use the target coord
        let effective_coord = if dim_size == 1 { 0 } else { target_coord[coord_idx] };
        index += effective_coord * stride;
    }
    index
}

// Helper to reduce (sum) the gradient to match the original input shape before broadcasting
// Used in backward passes of broadcastable binary ops.
pub fn reduce_gradient<T>(grad: &Tensor<T>, target_shape: &[usize]) -> Tensor<T>
where
    T: AddAssign + Copy + Clone + Default + Debug + 'static + Add<Output = T> + Zero + One + Sum<T>,
{
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone(); // Simple case: no reduction needed
    }

    let rank_diff = grad_shape.len().saturating_sub(target_shape.len());
    let mut axes_to_sum = Vec::new();

    // 1. Identify dimensions added by broadcasting (prepended 1s)
    for i in 0..rank_diff {
        axes_to_sum.push(i);
    }

    // 2. Identify dimensions that were 1 in the target shape but >1 in the grad shape
    for i in 0..target_shape.len() {
        let grad_dim_index = rank_diff + i;
        if grad_dim_index < grad_shape.len() && target_shape[i] == 1 && grad_shape[grad_dim_index] != 1 {
             if !axes_to_sum.contains(&grad_dim_index) {
                axes_to_sum.push(grad_dim_index);
             }
        }
    }

    // 3. Special case: if target is scalar ([]), sum all gradient axes
    if target_shape.is_empty() && !grad_shape.is_empty() {
        axes_to_sum = (0..grad_shape.len()).collect();
    }

    // Perform summation if needed
    if !axes_to_sum.is_empty() {
        // Use the Tensor's sum_axes method (requires T: Sum)
        grad.sum_axes(&axes_to_sum, true) // Keep dims = true!
    } else {
        // Should not happen if shapes differ and target is not scalar, indicates potential logic error elsewhere?
        eprintln!(
            "Warning: reduce_gradient logic anomaly. Shapes {:?} and {:?} differ, but no axes to sum found.",
            grad_shape, target_shape
        );
        grad.clone() // Return original gradient as a safe fallback
    }
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
        assert_eq!(broadcast_shapes(&[5, 1, 4, 1], &[4, 5]), Ok(vec![5, 1, 4, 5]));
        assert_eq!(broadcast_shapes(&[4, 5], &[5, 1, 4, 1]), Ok(vec![5, 1, 4, 5]));
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