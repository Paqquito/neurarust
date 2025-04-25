// Ce fichier contiendra les implémentations des opérations arithmétiques
// comme l'addition, la soustraction, etc.

use crate::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};
use crate::autograd::BackwardOp;
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::{RefCell};

/// Implements element-wise addition for two Tensors.
///
/// Performs `&tensor1 + &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Add<Output = T>` and `Copy`.
impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Add<Output = T> + Copy + Clone + 'static, // 'static needed for Rc<dyn Trait>
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
            // Create the backward operation context
            let grad_fn = AddBackward {
                input_a: self.get_weak_ref(), // Get Weak ref from Tensor wrapper
                input_b: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            // Store it in the result tensor's data
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
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

// Define the concrete BackwardOp struct for addition
// (Could be in autograd/add.rs later)
struct AddBackward<T> {
    // Need Weak refs to the input tensors' data RefCells to accumulate gradients.
    // We need the Tensor wrapper to manage the Rc<RefCell<...>> access.
    // Let's store Weak references to the *Tensor wrappers* for now.
    // We'll access their gradients via methods later in the backward pass.
    input_a: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>, // Use PhantomData if T is unused directly
}

impl<T> BackwardOp<T> for AddBackward<T> {
    fn backward(&self, upstream_grad: &Tensor<T>) {
        println!("AddBackward: Accumulating gradients...");

        // Attempt to upgrade weak references to strong ones (Rc)
        if let (Some(input_a_rc), Some(input_b_rc)) =
            (self.input_a.upgrade(), self.input_b.upgrade())
        {
            // Borrow gradients mutably
            let mut input_a_tensor_data = input_a_rc.borrow_mut();
            let mut input_b_tensor_data = input_b_rc.borrow_mut();

            // --- Accumulate gradient for input A --- 
            // This part requires Tensor addition and clone. 
            // Placeholder logic:
            if let Some(ref mut existing_grad_a) = input_a_tensor_data.grad {
                 // Need to implement += or similar for Tensor
                 // *existing_grad_a = existing_grad_a + upstream_grad; // Hypothetical
                 println!("  -> Accumulating grad for input A (Op needed)");
            } else {
                 input_a_tensor_data.grad = Some(upstream_grad.clone());
                 println!("  -> Setting grad for input A");
            }
            // Ensure requires_grad is true if we set a grad
            // input_a_tensor_data.requires_grad = true;

            // --- Accumulate gradient for input B --- 
            if let Some(ref mut existing_grad_b) = input_b_tensor_data.grad {
                 println!("  -> Accumulating grad for input B (Op needed)");
                 // *existing_grad_b = existing_grad_b + upstream_grad; // Hypothetical
            } else {
                 input_b_tensor_data.grad = Some(upstream_grad.clone());
                 println!("  -> Setting grad for input B");
            }
            // input_b_tensor_data.requires_grad = true;

        } else {
            eprintln!("Error: Could not upgrade weak references in AddBackward. Inputs might have been dropped.");
        }
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