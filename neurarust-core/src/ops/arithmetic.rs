// Ce fichier contiendra les implémentations des opérations arithmétiques
// comme l'addition, la soustraction, etc.

use crate::tensor::Tensor;
use std::ops::{Add, AddAssign, Sub, Mul, Div};
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
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise addition.");

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = AddBackward {
                input_a: self.get_weak_ref(),
                input_b: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
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
    T: Sub<Output = T> + Copy + Clone + 'static, // Add Clone and 'static
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

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = SubBackward {
                input_a: self.get_weak_ref(),
                input_b: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
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
    T: Mul<Output = T> + Copy + Clone + 'static, // Add Clone and 'static
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

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = MulBackward {
                input_a: self.clone(),
                input_b: other.clone(),
                input_a_grad: self.get_weak_ref(),
                input_b_grad: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
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
    T: Div<Output = T> + Copy + Clone + 'static, // Add Clone and 'static
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

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a / b) // Note: Potential division by zero depending on T
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = DivBackward {
                input_a: self.clone(),
                input_b: other.clone(),
                input_a_grad: self.get_weak_ref(),
                input_b_grad: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
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

impl<T> BackwardOp<T> for AddBackward<T>
where
    T: Add<Output = T> + Copy + Clone + 'static,
{
     fn backward(&self, upstream_grad: &Tensor<T>) {
        if let (Some(input_a_rc), Some(input_b_rc)) =
            (self.input_a.upgrade(), self.input_b.upgrade())
        {
            let input_a_tensor = Tensor(input_a_rc);
            let input_b_tensor = Tensor(input_b_rc);

            // Accumulate gradient for input A using Add + Clone
            let mut grad_a_opt = input_a_tensor.borrow_grad_mut();
            let new_grad_a = if let Some(existing_grad_a) = grad_a_opt.as_ref() {
                 // Use Add instead of AddAssign
                 &*existing_grad_a + upstream_grad // Requires Add<&Tensor<T>> for &Tensor<T>
            } else {
                 upstream_grad.clone()
            };
            *grad_a_opt = Some(new_grad_a);

            // Accumulate gradient for input B using Add + Clone
            let mut grad_b_opt = input_b_tensor.borrow_grad_mut();
            let new_grad_b = if let Some(existing_grad_b) = grad_b_opt.as_ref() {
                  &*existing_grad_b + upstream_grad
            } else {
                  upstream_grad.clone()
            };
             *grad_b_opt = Some(new_grad_b);

        } else {
            eprintln!("Error: Could not upgrade weak references in AddBackward. Inputs might have been dropped.");
        }
    }
}

struct SubBackward<T> {
    input_a: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SubBackward<T> {
    fn backward(&self, _upstream_grad: &Tensor<T>) {
        println!("SubBackward: backward called (gradient accumulation pending)");
        // TODO: Implement gradient accumulation (dA = dC * 1, dB = dC * -1)
        // Requires Tensor negation and addition/accumulation
    }
}

struct MulBackward<T> {
    input_a: Tensor<T>, // Need to store clones of inputs for Mul gradient
    input_b: Tensor<T>,
    input_a_grad: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b_grad: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MulBackward<T> {
    fn backward(&self, _upstream_grad: &Tensor<T>) {
        println!("MulBackward: backward called (gradient accumulation pending)");
        // TODO: Implement gradient accumulation (dA = dC * B, dB = dC * A)
        // Requires Tensor element-wise multiplication and addition/accumulation
    }
}

struct DivBackward<T> {
    input_a: Tensor<T>, // Need A for dB/dC
    input_b: Tensor<T>, // Need B for dA/dC and dB/dC
    input_a_grad: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b_grad: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for DivBackward<T> {
    fn backward(&self, _upstream_grad: &Tensor<T>) {
        println!("DivBackward: backward called (gradient accumulation pending)");
        // TODO: Implement gradient accumulation (dA = dC * (1/B), dB = dC * (-A / B^2))
        // Requires Tensor element-wise division, multiplication, negation, power/square, addition/accumulation
    }
}

// --- Addition Assign ---

/// Implements in-place element-wise addition (`+=`).
///
/// Performs `tensor1 += &tensor2`.
/// Modifies `tensor1` directly.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `AddAssign` and `Copy`.
impl<'a, T> AddAssign<&'a Tensor<T>> for Tensor<T>
where
    T: AddAssign + Copy, // T must support in-place addition and be copyable
{
    fn add_assign(&mut self, other: &'a Tensor<T>) {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for AddAssign.");

        let mut self_td_mut = self.borrow_tensor_data_mut();
        let other_td = other.borrow_tensor_data();

        self_td_mut.data.iter_mut()
            .zip(other_td.data.iter())
            .for_each(|(a, &b)| *a += b); // Perform in-place addition on elements
    }
}

#[cfg(test)]
mod tests {
    
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

    // --- Test AddAssign ---
    #[test]
    fn test_add_assign_ok() {
        let mut t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected = create_test_tensor(vec![6_i32, 8, 10, 12], vec![2, 2]);

        t1 += &t2; // Use AddAssign

        assert_eq!(t1, expected);
    }

    #[test]
    #[should_panic]
    fn test_add_assign_shape_mismatch() {
        let mut t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);

        t1 += &t2; // Should panic
    }
} 