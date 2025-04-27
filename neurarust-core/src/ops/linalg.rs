// Ce module contiendra les opérations d'algèbre linéaire comme matmul.

use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData; // Use correct path
use num_traits::Zero;
use std::ops::{Add, Mul, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

struct MatmulBackward<T> {
    input_a: Tensor<T>, // Need clones of inputs for matmul gradient
    input_b: Tensor<T>,
    input_a_ref: Weak<RefCell<TensorData<T>>>, // Use imported TensorData
    input_b_ref: Weak<RefCell<TensorData<T>>>, // Use imported TensorData
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MatmulBackward<T> 
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        println!("MatmulBackward: backward called (gradient accumulation pending)");
        // TODO: Implement gradient accumulation
        // Requires tensor transpose, matmul, and gradient accumulation logic
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        // Return the weak references to the inputs used for gradient propagation
        vec![self.input_a_ref.clone(), self.input_b_ref.clone()]
    }
}

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
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy + Clone + 'static,
    {
        // Get shapes (cloned)
        let self_shape = self.shape();
        let other_shape = other.shape();

        // --- Shape Checks ---
        assert_eq!(self_shape.len(), 2, "Matmul requires the first tensor to be 2D.");
        assert_eq!(other_shape.len(), 2, "Matmul requires the second tensor to be 2D.");

        let m = self_shape[0];
        let k1 = self_shape[1]; // Inner dimension of self
        let k2 = other_shape[0]; // Inner dimension of other
        let n = other_shape[1];

        assert_eq!(k1, k2, "Inner dimensions ({}) and ({}) do not match for matmul.", k1, k2);

        // --- Initialization ---
        let result_shape = vec![m, n];
        let mut result_data = vec![T::zero(); m * n]; // Initialize result data with zeros

        // --- Naive Matmul Algorithm ---
        // C[i, j] = sum(A[i, k] * B[k, j])
        for i in 0..m {        // Iterate over rows of the result
            for j in 0..n {    // Iterate over columns of the result
                let mut sum = T::zero();
                for k in 0..k1 { // Iterate over the inner dimension
                    // Access elements using the get_val helper method
                    sum = sum + self.get_val([i, k]) * other.get_val([k, j]);
                }
                // Calculate the flat index for the result matrix
                let result_flat_index = i * n + j;
                result_data[result_flat_index] = sum;
            }
        }

        // Create the result tensor and set up autograd context
        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, result_shape);
         if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = MatmulBackward {
                input_a: self.clone(), // Clone inputs needed for backward
                input_b: other.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            // Comment out this line for now as T might not satisfy all bounds required by BackwardOp at this point
            // result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn)); 
        }
        // TODO: Set grad_fn if requires_grad
        result
    }
}


#[cfg(test)]
mod tests {
    use crate::Tensor;

    // Helper
    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
     fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_matmul_2x2() {
        let a = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let b = create_test_tensor(vec![5, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![19, 22, 43, 50];
        let expected_shape = vec![2, 2];

        let result = a.matmul(&b);
        // Compare content
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let b = create_test_tensor(vec![7, 8, 9, 10, 11, 12], vec![3, 2]);
        let expected_data = vec![58, 64, 139, 154];
        let expected_shape = vec![2, 2];

        let result = a.matmul(&b);
        // Compare content
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

     #[test]
    fn test_matmul_propagate_requires_grad() {
        let a = create_test_tensor_with_grad::<i32>(vec![1, 2, 3, 4], vec![2, 2]);
        let b = create_test_tensor::<i32>(vec![5, 6, 7, 8], vec![2, 2]);
        let c = create_test_tensor_with_grad::<i32>(vec![1, 0, 0, 1], vec![2, 2]);

        let res1 = a.matmul(&b); // a requires grad
        assert!(res1.requires_grad());

        let res2 = b.matmul(&a); // a requires grad
        assert!(res2.requires_grad());

        let res3 = b.matmul(&b); // Neither requires grad
        assert!(!res3.requires_grad());

         let res4 = a.matmul(&c); // Both require grad
        assert!(res4.requires_grad());
    }

    #[test]
    #[should_panic = "Matmul requires the first tensor to be 2D."]
    fn test_matmul_first_arg_not_2d() {
        let a = create_test_tensor(vec![1, 2, 3], vec![3]);
        let b = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic = "Matmul requires the second tensor to be 2D."]
    fn test_matmul_second_arg_not_2d() {
        let a = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let b = create_test_tensor(vec![1, 2, 3], vec![3]);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic = "Inner dimensions (2) and (3) do not match for matmul."]
    fn test_matmul_dimension_mismatch() {
        let a = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let b = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![3, 2]);
        let _ = a.matmul(&b);
    }
} 