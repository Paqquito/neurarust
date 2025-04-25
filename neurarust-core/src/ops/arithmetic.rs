// Ce fichier contiendra les implémentations des opérations arithmétiques
// comme l'addition, la soustraction, etc.

use crate::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};

/// Implements element-wise addition for two Tensors.
///
/// Performs `&tensor1 + &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Add<Output = T>` and `Copy`.
impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Add<Output = T> + Copy, // T must support addition and be easily copyable
{
    type Output = Tensor<T>; // The addition results in a new owned Tensor

    /// Performs element-wise addition.
    ///
    /// # Panics
    /// Panics if the shapes of the two tensors are not identical.
    fn add(self, other: &'b Tensor<T>) -> Self::Output {
        assert_eq!(self.shape(), other.shape(), "Tensor shapes must match for element-wise addition.");

        let result_data: Vec<T> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a + b)
            .collect();

        // Determine if the result requires gradients
        let requires_grad = self.requires_grad || other.requires_grad;

        // Create the result tensor
        let mut result = Tensor::new(result_data, self.shape().to_vec());
        result.requires_grad = requires_grad;

        // TODO: If requires_grad, set up the grad_fn (backward context)
        // if requires_grad {
        //     result.grad_fn = Some(Rc::new(AddBackward::new(self.clone_for_graph(), other.clone_for_graph())));
        // }

        result
    }
}

/// Implements element-wise subtraction for two Tensors.
///
/// Performs `&tensor1 - &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Sub<Output = T>` and `Copy`.
impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Tensor<T>;

    /// Performs element-wise subtraction.
    ///
    /// # Panics
    /// Panics if the shapes of the two tensors are not identical.
    fn sub(self, other: &'b Tensor<T>) -> Self::Output {
        assert_eq!(self.shape(), other.shape(), "Tensor shapes must match for element-wise subtraction.");
        let result_data: Vec<T> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a - b)
            .collect();

        // Determine if the result requires gradients
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Tensor::new(result_data, self.shape().to_vec());
        result.requires_grad = requires_grad;

        // TODO: If requires_grad, set up the grad_fn for subtraction

        result
    }
}

/// Implements element-wise multiplication (Hadamard product) for two Tensors.
///
/// Performs `&tensor1 * &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Mul<Output = T>` and `Copy`.
/// Note: This is NOT matrix multiplication.
impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Tensor<T>;

    /// Performs element-wise multiplication.
    ///
    /// # Panics
    /// Panics if the shapes of the two tensors are not identical.
    fn mul(self, other: &'b Tensor<T>) -> Self::Output {
        assert_eq!(self.shape(), other.shape(), "Tensor shapes must match for element-wise multiplication.");
        let result_data: Vec<T> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a * b)
            .collect();

        // Determine if the result requires gradients
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Tensor::new(result_data, self.shape().to_vec());
        result.requires_grad = requires_grad;

        // TODO: If requires_grad, set up the grad_fn for multiplication

        result
    }
}

/// Implements element-wise division for two Tensors.
///
/// Performs `&tensor1 / &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Div<Output = T>` and `Copy`.
/// Division by zero behavior depends on the underlying type `T`.
impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Tensor<T>;

    /// Performs element-wise division.
    ///
    /// # Panics
    /// Panics if the shapes of the two tensors are not identical.
    /// Behavior on division by zero depends on the type `T` (e.g., floats might produce `inf` or `NaN`, integers might panic).
    fn div(self, other: &'b Tensor<T>) -> Self::Output {
        assert_eq!(self.shape(), other.shape(), "Tensor shapes must match for element-wise division.");
        let result_data: Vec<T> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a / b)
            .collect();

        // Determine if the result requires gradients
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Tensor::new(result_data, self.shape().to_vec());
        result.requires_grad = requires_grad;

        // TODO: If requires_grad, set up the grad_fn for division

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_add_tensors_ok() {
        let t1 = Tensor::new(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = Tensor::new(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = Tensor::new(vec![6_i32, 8, 10, 12], vec![2, 2]);

        // Use references for addition as per the impl signature
        let result = &t1 + &t2;

        assert_eq!(result, expected);
        assert!(!result.requires_grad); // Neither input requires grad
    }

     #[test]
    fn test_add_tensors_f32_ok() {
        let t1 = Tensor::new(vec![1.0_f32, 2.5, -3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0_f32, -1.5, 3.0, 0.0], vec![2, 2]);
        let expected = Tensor::new(vec![6.0_f32, 1.0, 0.0, 4.0], vec![2, 2]);

        let result = &t1 + &t2;

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_add_tensors_shape_mismatch() {
        let t1 = Tensor::new(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = Tensor::new(vec![5_i32, 6], vec![1, 2]); // Different shape

        // This should panic
        let _result = &t1 + &t2;
    }

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = Tensor::new(vec![6_i32, 8, 10, 12], vec![2, 2]);
        let t2 = Tensor::new(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = Tensor::new(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let result = &t1 - &t2;
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = Tensor::new(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = Tensor::new(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 - &t2;
    }

    #[test]
    fn test_mul_tensors_ok() {
        let t1 = Tensor::new(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = Tensor::new(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = Tensor::new(vec![5_i32, 12, 21, 32], vec![2, 2]);
        let result = &t1 * &t2;
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_mul_tensors_shape_mismatch() {
        let t1 = Tensor::new(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = Tensor::new(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 * &t2;
    }

    #[test]
    fn test_div_tensors_ok() {
        let t1 = Tensor::new(vec![10.0_f32, 12.0, 21.0, 32.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let expected = Tensor::new(vec![2.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = &t1 / &t2;
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_div_tensors_int_div_by_zero() {
        let t1 = Tensor::new(vec![10_i32], vec![1]);
        let t2 = Tensor::new(vec![0_i32], vec![1]);
        let _result = &t1 / &t2;
    }

    #[test]
    #[should_panic]
    fn test_div_tensors_shape_mismatch() {
        let t1 = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0_f32, 6.0], vec![1, 2]);
        let _result = &t1 / &t2;
    }
} 