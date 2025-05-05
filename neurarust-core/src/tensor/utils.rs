use crate::error::NeuraRustError;
use std::cmp::max;

/// Calculates the strides for a given shape.
///
/// Strides represent the number of elements to skip in the flattened data array
/// to move one step along each dimension. For a contiguous tensor (like one
/// created directly from a flat vector), the strides are calculated based on
/// the dimensions from right to left.
///
/// # Arguments
/// * `shape`: A slice representing the dimensions of the tensor.
///
/// # Returns
/// A `Vec<usize>` containing the strides corresponding to each dimension.
/// Returns an empty vector if the shape is empty (scalar case).
///
/// # Example
/// ```
/// use neurarust_core::tensor::utils::calculate_strides;
///
/// assert_eq!(calculate_strides(&[2, 3]), vec![3, 1]); // Stride for dim 0 is 3, for dim 1 is 1
/// assert_eq!(calculate_strides(&[2, 2, 2]), vec![4, 2, 1]);
/// assert_eq!(calculate_strides(&[5]), vec![1]); // 1D tensor
/// assert_eq!(calculate_strides(&[]), vec![]); // Scalar tensor
/// ```
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
/// Broadcasting allows operations between tensors of different shapes if their shapes
/// are compatible according to certain rules. This function implements the standard
/// broadcasting logic used in libraries like NumPy and PyTorch.
///
/// # Rules for Broadcasting:
/// 1. If the tensors have different numbers of dimensions, the shape of the one with fewer
///    dimensions is padded with ones on its leading (left) side.
/// 2. The function then compares the shapes element-wise, starting from the trailing dimensions.
/// 3. Two dimensions are compatible if:
///    a. They are equal.
///    b. One of them is 1.
///    c. One of them is 0 (results in a 0-sized dimension, compatible with any size including 1 or 0).
/// 4. If dimensions are compatible, the resulting dimension size is the maximum of the two.
///    If one dimension is 0, the resulting dimension is 0.
///
/// # Arguments
/// * `shape_a`: The shape of the first tensor.
/// * `shape_b`: The shape of the second tensor.
///
/// # Returns
/// * `Ok(Vec<usize>)`: The broadcasted shape if the input shapes are compatible.
/// * `Err(NeuraRustError::BroadcastError)`: If the shapes are incompatible according to the rules.
///
/// # Example
/// ```
/// use neurarust_core::tensor::utils::broadcast_shapes;
///
/// // Standard broadcasting
/// assert_eq!(broadcast_shapes(&[5, 3], &[3]), Ok(vec![5, 3]));
/// assert_eq!(broadcast_shapes(&[5, 1], &[3]), Ok(vec![5, 3]));
/// assert_eq!(broadcast_shapes(&[1, 3], &[5, 1]), Ok(vec![5, 3]));
///
/// // Broadcasting with a scalar (empty shape)
/// assert_eq!(broadcast_shapes(&[2, 3], &[]), Ok(vec![2, 3]));
/// assert_eq!(broadcast_shapes(&[], &[2, 3]), Ok(vec![2, 3]));
///
/// // Broadcasting involving zero dimensions
/// assert_eq!(broadcast_shapes(&[5, 0], &[3]), Ok(vec![5, 0])); // [5, 0] vs [1, 3] -> dim 0 mismatch is ok, result dim 0
/// assert_eq!(broadcast_shapes(&[5, 3], &[0]), Ok(vec![5, 0])); // Error? No, [5,3] vs [1,0] -> 3 vs 0 -> 0. 5 vs 1 -> 5. Result [5,0]
/// assert_eq!(broadcast_shapes(&[0], &[3]), Ok(vec![0])); // [0] vs [3] -> 0
/// assert_eq!(broadcast_shapes(&[0, 1], &[0, 5]), Ok(vec![0, 5])); // [0,1] vs [0,5] -> 1 vs 5 -> 5. 0 vs 0 -> 0.
///
/// // Incompatible shapes
/// assert!(broadcast_shapes(&[5, 3], &[4, 1]).is_err());
/// assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
/// ```
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
///
/// Given a flat index into the tensor's underlying 1D data buffer, this function
/// calculates the corresponding coordinates in the multi-dimensional tensor space.
/// It assumes a standard row-major layout (C-style).
///
/// # Arguments
/// * `index`: The linear index (0-based) into the flattened data array.
/// * `shape`: A slice representing the dimensions of the tensor.
///
/// # Returns
/// A `Vec<usize>` containing the multi-dimensional coordinates corresponding to the linear index.
/// Returns an empty vector if the shape is empty (scalar case).
///
/// # Panics
/// Panics if the `index` is out of bounds for the total number of elements
/// defined by the `shape`. An index of 0 is considered valid even for shapes
/// containing a zero dimension (representing an empty tensor), in which case
/// the coordinates returned will contain zeros for the zero-sized dimensions.
///
/// # Example
/// ```
/// use neurarust_core::tensor::utils::index_to_coord;
///
/// let shape = &[2, 3, 4]; // 2*3*4 = 24 elements
/// assert_eq!(index_to_coord(0, shape), vec![0, 0, 0]);
/// assert_eq!(index_to_coord(4, shape), vec![0, 1, 0]); // Start of the second row in the first plane
/// assert_eq!(index_to_coord(11, shape), vec![0, 2, 3]);// Last element of the first plane
/// assert_eq!(index_to_coord(12, shape), vec![1, 0, 0]);// First element of the second plane
/// assert_eq!(index_to_coord(23, shape), vec![1, 2, 3]);// Last element
///
/// // Scalar case
/// assert_eq!(index_to_coord(0, &[]), vec![]);
///
/// // Empty tensor case
/// let empty_shape = &[2, 0, 3];
/// assert_eq!(index_to_coord(0, empty_shape), vec![0, 0, 0]); // Index 0 is valid
/// ```
/// ```should_panic
/// use neurarust_core::tensor::utils::index_to_coord;
///
/// let shape = &[2, 3];
/// index_to_coord(6, shape); // Index out of bounds (0-5 are valid)
/// ```
/// ```should_panic
/// use neurarust_core::tensor::utils::index_to_coord;
///
/// let empty_shape = &[2, 0, 3];
/// index_to_coord(1, empty_shape); // Index > 0 is invalid for empty tensor
/// ```
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

// Link the external tests file
#[cfg(test)]
#[path = "utils_test.rs"] mod tests;
