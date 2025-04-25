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
    T: Add<Output = T> + Copy, // Copy needed for map logic
{
    type Output = Tensor<T>;

    /// Performs element-wise addition.
    ///
    /// # Panics
    /// Panics if the shapes of the two tensors are not identical.
    fn add(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape(); // Clones shape
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise addition.");

        let self_data_ref = self.borrow_data(); // Get Ref<Vec<T>>
        let other_data_ref = other.borrow_data();

        let result_data: Vec<T> = self_data_ref
            .iter()
            .zip(other_data_ref.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape); // Use original shape
        if requires_grad { // Only set if true, default is false
            result.set_requires_grad(true);
        }
        // TODO: Set grad_fn if requires_grad
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
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise subtraction.");

        let self_data_ref = self.borrow_data();
        let other_data_ref = other.borrow_data();

        let result_data: Vec<T> = self_data_ref
            .iter()
            .zip(other_data_ref.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
        }
        // TODO: Set grad_fn if requires_grad
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
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise multiplication.");

        let self_data_ref = self.borrow_data();
        let other_data_ref = other.borrow_data();

        let result_data: Vec<T> = self_data_ref
            .iter()
            .zip(other_data_ref.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
        }
        // TODO: Set grad_fn if requires_grad
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
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise division.");

        let self_data_ref = self.borrow_data();
        let other_data_ref = other.borrow_data();

        let result_data: Vec<T> = self_data_ref
            .iter()
            .zip(other_data_ref.iter())
            .map(|(&a, &b)| a / b) // Note: Potential division by zero depending on T
            .collect();

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
        }
        // TODO: Set grad_fn if requires_grad
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    // Helper pour créer des tenseurs dans les tests
    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_add_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = create_test_tensor(vec![6_i32, 8, 10, 12], vec![2, 2]);
        let result = &t1 + &t2;
        assert_eq!(result, expected);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_add_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]); // t2 requires grad
        let t3 = create_test_tensor::<f32>(vec![3.0], vec![1]);

        let res1 = &t1 + &t2;
        assert!(res1.requires_grad());

        let res2 = &t1 + &t3;
        assert!(!res2.requires_grad());

        let t1_grad = create_test_tensor_with_grad::<f32>(vec![4.0], vec![1]);
        let res3 = &t1_grad + &t2; // Both require grad
        assert!(res3.requires_grad());
    }

    #[test]
    #[should_panic]
    fn test_add_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 + &t2;
    }

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = create_test_tensor(vec![6_i32, 8, 10, 12], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let result = &t1 - &t2;
        assert_eq!(result, expected);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_sub_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let res = &t2 - &t1;
        assert!(res.requires_grad());
    }

    #[test]
    #[should_panic]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 - &t2;
    }

    #[test]
    fn test_mul_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = create_test_tensor(vec![5_i32, 12, 21, 32], vec![2, 2]);
        let result = &t1 * &t2;
        assert_eq!(result, expected);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_mul_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let res = &t1 * &t2;
        assert!(res.requires_grad());
    }

    #[test]
    #[should_panic]
    fn test_mul_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 * &t2;
    }

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0_f32, 12.0, 21.0, 32.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let expected = create_test_tensor(vec![2.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = &t1 / &t2;
        assert_eq!(result, expected);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_div_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor::<f32>(vec![2.0], vec![1]);
        let res = &t1 / &t2;
        assert!(res.requires_grad());
    }

    #[test]
    #[should_panic] // Integer division by zero should panic
    fn test_div_tensors_int_div_by_zero() {
        let t1 = create_test_tensor(vec![10_i32], vec![1]);
        let t2 = create_test_tensor(vec![0_i32], vec![1]);
        let _result = &t1 / &t2;
    }

    #[test]
    #[should_panic]
    fn test_div_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0_f32, 6.0], vec![1, 2]);
        let _result = &t1 / &t2;
    }
} 