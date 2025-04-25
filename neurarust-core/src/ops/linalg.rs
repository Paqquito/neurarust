// Ce module contiendra les opérations d'algèbre linéaire comme matmul.

use crate::tensor::Tensor;
use num_traits::Zero;
use std::ops::{Add, Mul};

impl<T> Tensor<T> {
    /// Performs matrix multiplication (matmul) between two 2D tensors (matrices).
    ///
    /// Calculates `self * other`.
    /// `self` must have shape `(M, K)` and `other` must have shape `(K, N)`.
    /// The resulting tensor will have shape `(M, N)`.
    ///
    /// Requires the element type `T` to implement `Add<Output = T>`, `Mul<Output = T>`, `Zero`, and `Copy`.
    /// Uses a naive triple-loop algorithm for now.
    ///
    /// # Panics
    /// - Panics if either tensor is not 2-dimensional.
    /// - Panics if the inner dimensions (`K`) do not match.
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy, // Constraints needed for matmul
    {
        // --- Shape Checks ---
        assert_eq!(self.shape.len(), 2, "Matmul requires the first tensor to be 2D.");
        assert_eq!(other.shape.len(), 2, "Matmul requires the second tensor to be 2D.");

        let m = self.shape[0];
        let k1 = self.shape[1]; // Inner dimension of self
        let k2 = other.shape[0]; // Inner dimension of other
        let n = other.shape[1];

        assert_eq!(k1, k2, "Inner dimensions ({}) and ({}) do not match for matmul.", k1, k2);

        // --- Initialization ---
        let result_shape = vec![m, n];
        let mut result_data = vec![T::zero(); m * n]; // Initialize result data with zeros

        // --- Naive Matmul Algorithm ---
        // C[i, j] = sum(A[i, k] * B[k, j])
        for i in 0..m {        // Iterate over rows of the result
            for j in 0..n {    // Iterate over columns of the result
                let mut sum = T::zero();
                for k in 0..k1 { // Iterate over the inner dimension (k1 or k2)
                    // Access elements using 2D indexing (implemented previously)
                    sum = sum + self[[i, k]] * other[[k, j]];
                }
                // Calculate the flat index for the result matrix
                let result_flat_index = i * n + j;
                result_data[result_flat_index] = sum;
            }
        }

        // Create the result tensor
        Tensor::new(result_data, result_shape)
    }
}


#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_matmul_2x2() {
        // A = [1, 2]  (shape [1, 2]) - incorrect, should be 2x2 for example
        //     [3, 4]
        // B = [5, 6]
        //     [7, 8]
        // C = A * B = [1*5+2*7, 1*6+2*8] = [19, 22]
        //             [3*5+4*7, 3*6+4*8]   [43, 50]

        let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let expected = Tensor::new(vec![19, 22, 43, 50], vec![2, 2]);

        let result = a.matmul(&b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        // A = [1, 2, 3]  (shape [2, 3])
        //     [4, 5, 6]
        // B = [7, 8 ]   (shape [3, 2])
        //     [9, 10]
        //     [11,12]
        // C = A * B = [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
        //             [4*7+5*9+6*11, 4*8+5*10+6*12]   [28+45+66, 32+50+72] = [139, 154]
        // Shape [2, 2]

        let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let b = Tensor::new(vec![7, 8, 9, 10, 11, 12], vec![3, 2]);
        let expected = Tensor::new(vec![58, 64, 139, 154], vec![2, 2]);

        let result = a.matmul(&b);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic = "Matmul requires the first tensor to be 2D."]
    fn test_matmul_first_arg_not_2d() {
        let a = Tensor::new(vec![1, 2, 3], vec![3]); // 1D
        let b = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic = "Matmul requires the second tensor to be 2D."]
    fn test_matmul_second_arg_not_2d() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let b = Tensor::new(vec![1, 2, 3], vec![3]); // 1D
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic = "Inner dimensions (2) and (3) do not match for matmul."]
    fn test_matmul_dimension_mismatch() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]); // K=2
        let b = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![3, 2]); // K=3
        let _ = a.matmul(&b);
    }
} 