// Ce fichier contiendra les implémentations des opérations arithmétiques
// comme l'addition, la soustraction, etc.

use crate::tensor::Tensor;
use std::ops::{Add, AddAssign, Sub, Mul, Div, Neg};
use crate::autograd::BackwardOp;
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::{RefCell};

/// Implements element-wise addition for two Tensors.
///
/// Performs `&tensor1 + &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Add<Output = T>`, `AddAssign` (for grad), `Copy` and `Clone`.
impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Add<Output = T> + AddAssign + Copy + Clone + 'static,
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
/// Requires the element type `T` to implement `Sub<Output = T>`, `Neg<Output = T>`, `AddAssign`, `Copy` and `Clone`.
impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Sub<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static,
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
    T: Mul<Output = T> + Copy + Clone + 'static + AddAssign, // Tentatively add AddAssign+Copy
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
                input_a_val: self.clone(),
                input_b_val: other.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
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
    T: Div<Output = T> + Copy + Clone + 'static + AddAssign, // Tentatively add AddAssign+Copy
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
                input_a_val: self.clone(),
                input_b_val: other.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

/// Implements the unary negation operation (`-tensor`).
/// 
/// Requires the element type `T` to implement `Neg<Output = T>` and `Copy`.
/// Note: This implementation does not currently participate in autograd graph building.
/// It's primarily intended for use within other backward operations.
impl<'a, T> Neg for &'a Tensor<T>
where
    T: Neg<Output = T> + Copy + Clone + 'static, // Add Clone + 'static for Tensor::new
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let td = self.borrow_tensor_data();
        let neg_data: Vec<T> = td.data.iter().map(|&x| -x).collect();
        let shape = td.shape.clone();
        drop(td);

        // Create a new tensor. For now, disable grad tracking for standalone negation.
        // If negation needs its own backward pass later, this needs refinement.
        Tensor::new(neg_data, shape)
    }
}

// Define the concrete BackwardOp struct for addition
struct AddBackward<T> {
    input_a: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for AddBackward<T>
where
    T: AddAssign + Copy + Clone + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // Accumulate gradient for Input A
        if let Some(input_a_rc) = self.input_a.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                if let Some(existing_grad_a) = input_a_td.grad.as_mut() {
                    *existing_grad_a += upstream_grad;
                } else {
                    // Initialize gradient with a deep copy
                    input_a_td.grad = Some(Tensor::new(upstream_grad.data(), upstream_grad.shape())); // Requires T: Clone
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input A in AddBackward.");
        }

        // Accumulate gradient for Input B
        if let Some(input_b_rc) = self.input_b.upgrade() {
            let mut input_b_td = input_b_rc.borrow_mut();
            if input_b_td.requires_grad {
                if let Some(existing_grad_b) = input_b_td.grad.as_mut() {
                    *existing_grad_b += upstream_grad;
                } else {
                    // Initialize gradient with a deep copy
                    input_b_td.grad = Some(Tensor::new(upstream_grad.data(), upstream_grad.shape())); // Requires T: Clone
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input B in AddBackward.");
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<crate::tensor::TensorData<T>>>> {
        vec![self.input_a.clone(), self.input_b.clone()] // Return cloned Weak refs
    }
}

struct SubBackward<T> {
    input_a: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SubBackward<T>
where
    // Need Neg for grad B, AddAssign for accum, Clone for init, Copy for ops, 'static
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // Accumulate gradient for Input A (dL/dA = upstream_grad)
        if let Some(input_a_rc) = self.input_a.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                if let Some(existing_grad_a) = input_a_td.grad.as_mut() {
                    *existing_grad_a += upstream_grad;
                } else {
                    input_a_td.grad = Some(Tensor::new(upstream_grad.data(), upstream_grad.shape()));
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input A in SubBackward.");
        }

        // Accumulate gradient for Input B (dL/dB = -upstream_grad)
        if let Some(input_b_rc) = self.input_b.upgrade() {
            let mut input_b_td = input_b_rc.borrow_mut();
            if input_b_td.requires_grad {
                // Calculate negated upstream gradient ONCE
                let neg_upstream_grad = -upstream_grad; // Use the Neg impl for &Tensor<T>

                if let Some(existing_grad_b) = input_b_td.grad.as_mut() {
                    *existing_grad_b += &neg_upstream_grad; // Accumulate the negated gradient
                } else {
                    // Initialize with the negated gradient (already a new tensor)
                    input_b_td.grad = Some(neg_upstream_grad);
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input B in SubBackward.");
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<crate::tensor::TensorData<T>>>> {
        vec![self.input_a.clone(), self.input_b.clone()]
    }
}

struct MulBackward<T> {
    // Store clones of inputs for gradient CALCULATION (dL/dA = dL/dC * B)
    input_a_val: Tensor<T>,
    input_b_val: Tensor<T>,
    // Store weak refs for gradient PROPAGATION
    input_a_ref: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b_ref: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MulBackward<T> {
    fn backward(&self, _upstream_grad: &Tensor<T>) {
        println!("MulBackward: backward called (gradient accumulation pending)");
        // TODO: Implement gradient accumulation (dA = dC * B, dB = dC * A)
        // Use self.input_a_val and self.input_b_val for calculations
        // Accumulate into refs obtained from self.input_a_ref.upgrade(), self.input_b_ref.upgrade()
    }

    fn inputs(&self) -> Vec<Weak<RefCell<crate::tensor::TensorData<T>>>> {
        vec![self.input_a_ref.clone(), self.input_b_ref.clone()]
    }
}

struct DivBackward<T> {
    // Store clones of inputs for gradient CALCULATION
    input_a_val: Tensor<T>, // Need A for dB/dC = -A / B^2
    input_b_val: Tensor<T>, // Need B for dA/dC = 1 / B and dB/dC
    // Store weak refs for gradient PROPAGATION
    input_a_ref: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b_ref: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for DivBackward<T> {
    fn backward(&self, _upstream_grad: &Tensor<T>) {
        println!("DivBackward: backward called (gradient accumulation pending)");
        // TODO: Implement gradient accumulation (dA = dC * (1/B), dB = dC * (-A / B^2))
        // Requires Tensor element-wise division, multiplication, negation, power/square, addition/accumulation
    }

    fn inputs(&self) -> Vec<Weak<RefCell<crate::tensor::TensorData<T>>>> {
        vec![self.input_a_ref.clone(), self.input_b_ref.clone()]
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
        let expected_data = vec![6_i32, 8, 10, 12];
        let expected_shape = vec![2, 2];
        let result = &t1 + &t2;
        
        // Compare content
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
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
        let expected_data = vec![1_i32, 2, 3, 4];
        let expected_shape = vec![2, 2];
        let result = &t1 - &t2;

        // Compare content
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
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
        let expected_data = vec![5_i32, 12, 21, 32];
        let expected_shape = vec![2, 2];
        let result = &t1 * &t2;
        
        // Compare content
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
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
        let expected_data = vec![2.0_f32, 2.0, 3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = &t1 / &t2;

        // Compare content
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
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
        let expected_data = vec![6_i32, 8, 10, 12];
        let expected_shape = vec![2, 2];

        t1 += &t2; // Use AddAssign

        // Compare content instead of Tensor identity
        assert_eq!(t1.data(), expected_data, "Data mismatch");
        assert_eq!(t1.shape(), expected_shape, "Shape mismatch");
    }

    #[test]
    #[should_panic]
    fn test_add_assign_shape_mismatch() {
        let mut t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);

        t1 += &t2; // Should panic
    }

    #[test]
    fn test_add_backward() {
        //Requires T: AddAssign + Copy + Clone + 'static + PartialEq + Debug + Zero
        let a = create_test_tensor_with_grad::<f32>(vec![2.0, 3.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0], vec![2]);

        // Perform addition - this sets up the grad_fn
        let c = &a + &b;
        assert!(c.requires_grad());
        let grad_fn_option = c.0.borrow().grad_fn.clone(); // Clone the Rc
        assert!(grad_fn_option.is_some());
        let grad_fn = grad_fn_option.unwrap(); // grad_fn is Rc<dyn BackwardOp<f32>>

        // Check initial grads of a and b are None
        assert!(a.borrow_grad().is_none());
        assert!(b.borrow_grad().is_none());

        // Create upstream gradient (gradient of loss w.r.t c)
        let upstream_grad = Tensor::new(vec![1.0, 1.0], vec![2]); // Usually starts with ones for scalars

        // Execute the backward pass for the Add operation
        grad_fn.backward(&upstream_grad);

        // Check gradients content
        {
            let grad_a = a.borrow_grad();
            let grad_b = b.borrow_grad();

            assert!(grad_a.is_some());
            assert!(grad_b.is_some());

            let expected_grad_data = vec![1.0, 1.0];
            let expected_grad_shape = vec![2];
            assert_eq!(grad_a.as_ref().unwrap().data(), expected_grad_data, "Grad A data mismatch");
            assert_eq!(grad_a.as_ref().unwrap().shape(), expected_grad_shape, "Grad A shape mismatch");
            assert_eq!(grad_b.as_ref().unwrap().data(), expected_grad_data, "Grad B data mismatch");
            assert_eq!(grad_b.as_ref().unwrap().shape(), expected_grad_shape, "Grad B shape mismatch");
        } // grad_a and grad_b (Ref<...>) are dropped here

        // Test accumulation: call backward again with a different upstream grad
        let upstream_grad_2 = Tensor::new(vec![0.5, -0.5], vec![2]);
        grad_fn.backward(&upstream_grad_2); // Should not panic now

        // Check accumulated gradients content
        let grad_a_accum = a.borrow_grad();
        let grad_b_accum = b.borrow_grad();
        let expected_accum_grad_data = vec![1.5, 0.5]; // 1.0 + 0.5, 1.0 - 0.5
        let expected_accum_grad_shape = vec![2];

        assert_eq!(grad_a_accum.as_ref().unwrap().data(), expected_accum_grad_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.as_ref().unwrap().shape(), expected_accum_grad_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().data(), expected_accum_grad_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().shape(), expected_accum_grad_shape, "Accum Grad B shape mismatch");
    }

    #[test]
    fn test_sub_backward() {
        // Requires T: Sub, Neg, AddAssign, Copy, Clone, 'static, PartialEq, Debug, Zero
        let a = create_test_tensor_with_grad::<f32>(vec![10.0, 20.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![3.0, 8.0], vec![2]);

        // Perform subtraction
        let c = &a - &b;
        assert!(c.requires_grad());
        let grad_fn_option = c.0.borrow().grad_fn.clone();
        assert!(grad_fn_option.is_some());
        let grad_fn = grad_fn_option.unwrap();

        // Check initial grads
        assert!(a.borrow_grad().is_none());
        assert!(b.borrow_grad().is_none());

        // Upstream gradient
        let upstream_grad = Tensor::new(vec![1.0, -1.0], vec![2]);

        // Execute backward
        grad_fn.backward(&upstream_grad);

        // Check gradients content
        {
            let grad_a = a.borrow_grad();
            let grad_b = b.borrow_grad();
            assert!(grad_a.is_some());
            assert!(grad_b.is_some());
            let expected_grad_a_data = vec![1.0, -1.0];
            let expected_grad_b_data = vec![-1.0, 1.0]; 
            let expected_shape = vec![2];
            assert_eq!(grad_a.as_ref().unwrap().data(), expected_grad_a_data, "Grad A data mismatch");
            assert_eq!(grad_a.as_ref().unwrap().shape(), expected_shape, "Grad A shape mismatch");
            assert_eq!(grad_b.as_ref().unwrap().data(), expected_grad_b_data, "Grad B data mismatch");
            assert_eq!(grad_b.as_ref().unwrap().shape(), expected_shape, "Grad B shape mismatch");
        }

        // Test accumulation
        let upstream_grad_2 = Tensor::new(vec![0.5, 0.5], vec![2]);
        grad_fn.backward(&upstream_grad_2);

        // Check accumulated gradients content
        let grad_a_accum = a.borrow_grad();
        let grad_b_accum = b.borrow_grad();
        let expected_accum_grad_a_data = vec![1.5, -0.5]; 
        let expected_accum_grad_b_data = vec![-1.5, 0.5];
        let expected_accum_shape = vec![2];

        assert_eq!(grad_a_accum.as_ref().unwrap().data(), expected_accum_grad_a_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().data(), expected_accum_grad_b_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad B shape mismatch");
    }
} 