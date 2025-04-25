// Ce module contiendra les implémentations pour l'indexation des Tenseurs.

use crate::tensor::Tensor;
use std::ops::Index;

/// Implements 2D indexing (read-only) for Tensors.
///
/// Allows accessing elements using `tensor[[row, col]]` syntax for 2D tensors.
/// Requires the tensor to have exactly 2 dimensions.
impl<T> Index<[usize; 2]> for Tensor<T> {
    type Output = T;

    /// Accesses the element at the specified 2D index (row, column).
    ///
    /// Assumes row-major storage layout.
    ///
    /// # Panics
    /// - Panics if the tensor is not 2-dimensional.
    /// - Panics if the provided row or column index is out of bounds.
    #[inline] // Suggest inlining for performance critical access
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        // Ensure the tensor is 2D
        assert_eq!(self.shape.len(), 2, "Indexing with [row, col] requires a 2D tensor.");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let row_idx = index[0];
        let col_idx = index[1];

        // Bounds checking
        assert!(row_idx < rows, "Row index {} out of bounds for shape {:?}", row_idx, self.shape);
        assert!(col_idx < cols, "Column index {} out of bounds for shape {:?}", col_idx, self.shape);

        // Calculate the flat index for row-major layout
        let flat_index = row_idx * cols + col_idx;

        // Access the data using the flat index
        // This index is guaranteed to be in bounds due to the checks above
        // and the invariant that data.len() == rows * cols established in Tensor::new
        &self.data[flat_index]
    }
}

// TODO: Implement IndexMut for mutable access
// TODO: Implement Index for other dimensionalities (e.g., Index<usize> for 1D, Index<&[usize]> for N-D)

#[cfg(test)]
mod tests {
    // Note: We need to import Tensor from the crate root because `tensor.rs` is a sibling module.
    use crate::Tensor;

    #[test]
    fn test_index_2d_ok() {
        // 1 2 3
        // 4 5 6
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        assert_eq!(t[[0, 0]], 1);
        assert_eq!(t[[0, 1]], 2);
        assert_eq!(t[[0, 2]], 3);
        assert_eq!(t[[1, 0]], 4);
        assert_eq!(t[[1, 1]], 5);
        assert_eq!(t[[1, 2]], 6);
    }

    #[test]
    #[should_panic = "Row index 2 out of bounds for shape [2, 3]"]
    fn test_index_2d_row_out_of_bounds() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let _ = t[[2, 0]]; // Access row 2 (out of bounds)
    }

    #[test]
    #[should_panic = "Column index 3 out of bounds for shape [2, 3]"]
    fn test_index_2d_col_out_of_bounds() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let _ = t[[1, 3]]; // Access col 3 (out of bounds)
    }

    #[test]
    #[should_panic = "Indexing with [row, col] requires a 2D tensor."]
    fn test_index_2d_on_1d_tensor() {
        let t = Tensor::new(vec![1, 2, 3], vec![3]); // 1D tensor
        let _ = t[[0, 0]]; // Attempt 2D indexing
    }

     #[test]
    #[should_panic = "Indexing with [row, col] requires a 2D tensor."]
    fn test_index_2d_on_3d_tensor() {
        let t = Tensor::new(vec![1], vec![1, 1, 1]); // 3D tensor
        let _ = t[[0, 0]]; // Attempt 2D indexing
    }
} 