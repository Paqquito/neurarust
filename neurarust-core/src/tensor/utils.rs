use std::cmp::max;
use crate::tensor::Tensor;
use std::ops::{AddAssign, Add};
use num_traits::{Zero, One};
use std::iter::Sum as IterSum;
use std::fmt::Debug;

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

/// Reduces a gradient tensor to match the shape of an original input tensor 
/// that was involved in a broadcasting operation.
///
/// When broadcasting occurs during a forward pass (e.g., A[2,3] + B[3] -> C[2,3]),
/// the gradient flowing back to an input must have the shape of that input.
/// If the upstream gradient (dL/dC) has the broadcasted shape (e.g., [2,3]), 
/// the gradient w.r.t. the smaller input (dL/dB) needs to be summed across 
/// the dimensions that were broadcasted.
///
/// This function identifies the dimensions that were broadcasted and sums the 
/// `grad` tensor along those dimensions.
///
/// # Arguments
/// * `grad` - The gradient tensor, typically having the broadcasted shape.
/// * `target_shape` - The shape of the original input tensor to which the 
///   gradient should be reduced.
///
/// # Returns
/// A new Tensor representing the gradient summed over the appropriate dimensions 
/// to match `target_shape`. If no broadcasting occurred (i.e., `grad.shape() == target_shape`),
/// a clone of the original `grad` is returned.
pub fn reduce_gradient<T>(grad: &Tensor<T>, target_shape: &[usize]) -> Tensor<T>
where
    T: AddAssign + Copy + Clone + Default + Debug + 'static + Add<Output = T> + Zero + One + IterSum,
{
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone(); // No reduction needed if shapes already match
    }

    // Handle reduction to scalar explicitly for clarity and efficiency.
    // A scalar target shape means we need to sum the gradient over all its dimensions.
    if target_shape.is_empty() {
        let axes_to_sum = (0..grad_shape.len()).collect::<Vec<_>>();
        // Use sum_axes with keep_dims=false to produce a scalar tensor (shape []).
        // Note: Tensor::new expects shape `[]` for scalar, not `[1]`.
        // If Tensor::sum_axes returns shape [1] for scalar, we might need reshape.
        // Assuming Tensor::sum_axes(..., false) correctly produces shape [].
        return grad.sum_axes(&axes_to_sum, false);
    }

    let rank_diff = grad_shape.len().saturating_sub(target_shape.len());
    let mut axes_to_sum = Vec::new();

    // Identify axes to sum:
    // 1. Dimensions that were added (prepended) to the target shape during broadcasting.
    // These correspond to the first `rank_diff` dimensions of the gradient tensor.
    for i in 0..rank_diff {
        axes_to_sum.push(i);
    }

    // 2. Dimensions that were originally 1 in the target shape but are > 1 in the gradient's shape.
    // These dimensions were expanded during broadcasting.
    for i in 0..target_shape.len() {
        let grad_dim_index = rank_diff + i;
        // Ensure we don't go out of bounds for grad_shape (can happen if target_shape is longer? Unlikely)
        if grad_dim_index < grad_shape.len() 
           && target_shape[i] == 1         // Dimension was 1 in the original input
           && grad_shape[grad_dim_index] != 1 // Dimension is > 1 in the gradient (result of broadcast)
        {
             // Avoid adding the same axis twice if it was already added in step 1 (rank_diff)
             if !axes_to_sum.contains(&grad_dim_index) {
                axes_to_sum.push(grad_dim_index);
             }
        }
    }
    
    // Perform the summation along the identified axes.
    if !axes_to_sum.is_empty() {
        // We use `keep_dims=true` here. This ensures the resulting tensor still has the same
        // number of dimensions as the input `grad`, but with size 1 in the summed dimensions.
        // This is often the desired behavior for gradient reduction in autograd frameworks,
        // as it simplifies further operations if needed. The final shape might not exactly
        // match `target_shape` if target_shape had fewer dimensions, but it represents the
        // correctly summed gradient values projected back into the higher-rank space.
        // Example: grad [2,3], target [3] -> axes_to_sum=[0] -> result [1,3]
        // Example: grad [2,3], target [1] -> axes_to_sum=[0,1] -> result [1,1]
        grad.sum_axes(&axes_to_sum, true) 
    } else {
        // If shapes differ but no axes to sum were identified, it might indicate an issue
        // in the broadcasting logic or an unexpected input.
        // Warn and return a clone as a safe fallback, though this case shouldn't ideally happen.
        eprintln!(
            "Warning: reduce_gradient inconsistency. Shapes {:?} and {:?} differ, but no axes to sum found.",
            grad_shape, target_shape
        );
        grad.clone()
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