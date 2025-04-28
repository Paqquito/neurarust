// neurarust-optim/src/sgd.rs

// Utiliser les types de neurarust_core
use neurarust_core::tensor::Tensor;
// Importer le trait Optimizer depuis lib.rs
use crate::Optimizer; 
// Import necessary traits for calculations
use std::ops::{SubAssign, Mul};
use num_traits::{FromPrimitive, Zero};
use std::fmt::Debug;

/// Implements stochastic gradient descent (optionally with momentum).
/// 
/// Updates parameters `p` according to the rule:
/// `p = p - lr * grad(p)`
#[derive(Debug)]
pub struct SGD<T: Clone> { 
    lr: T, // Learning rate
    // params: Vec<Tensor<T>>, // Removed: Optimizer is stateless regarding params list
}

impl<T> SGD<T>
where
    T: Copy + Clone + FromPrimitive + Debug,
{
    /// Creates a new SGD optimizer instance.
    ///
    /// # Arguments
    ///
    /// * `lr` - The learning rate.
    pub fn new(lr: T) -> Self {
        SGD { lr }
    }
}

impl<T> Optimizer<T> for SGD<T> 
where
    T: Copy + Clone + Debug + Zero + Mul<Output = T> + SubAssign + 'static,
{
    /// Performs a single optimization step (parameter update).
    fn step(&mut self, params: &mut [&mut Tensor<T>]) {
        for param in params.iter_mut() {
            if let Some(grad_ref) = param.grad() {
                // We need to calculate update = lr * grad element-wise
                // Get gradient data
                let grad_data = grad_ref.data();
                
                // Prepare storage for update data (same shape as gradient/param)
                let mut update_data_vec = Vec::with_capacity(grad_data.len());
                
                // Calculate update data: update_val = lr * grad_val
                for grad_val in grad_data.iter() {
                    // Requires T: Mul<Output = T>
                    update_data_vec.push(self.lr * *grad_val);
                }

                // Apply update using data_mut and element-wise subtraction
                let mut param_data = param.data_mut();
                assert_eq!(param_data.len(), update_data_vec.len(), "Shape mismatch during SGD step");
                for (p_val, u_val) in param_data.iter_mut().zip(update_data_vec.iter()) {
                    // Requires T: SubAssign
                    *p_val -= *u_val;
                }
            }
        }
    }

    /// Clears the gradients of all parameters managed by the optimizer.
    fn zero_grad(&self, params: &mut [&mut Tensor<T>]) {
        for param in params.iter_mut() {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import SGD
    use neurarust_core::tensor::Tensor; // Import Tensor from core
    use crate::Optimizer; // Import Optimizer trait
    use num_traits::{Float, Zero, FromPrimitive};
    use std::fmt::Debug;

    // Remove the local helper function that uses private methods
    // fn set_gradient_for_test<T: Clone + Debug + Zero + 'static>(tensor: &Tensor<T>, grad_data: Vec<T>) {
    //      assert_eq!(tensor.shape().iter().product::<usize>(), grad_data.len(), "Gradient data length mismatch");
    //      let grad_tensor = Tensor::new(grad_data, tensor.shape());
    //      // THIS RELIES ON A TEMPORARY CHANGE IN neurarust-core making borrow_tensor_data_mut pub
    //      let mut td = tensor.borrow_tensor_data_mut();
    //      td.grad = Some(grad_tensor);
    //      if !td.requires_grad {
    //          td.requires_grad = true; // Ensure requires_grad is set if gradient exists
    //      }
    // }

    // Helper to check approximate data equality
    fn check_data_approx<T>(tensor: &Tensor<T>, expected_data: &[T])
    where
        T: Float + Debug + FromPrimitive,
    {
        let tensor_data = tensor.data();
        assert_eq!(tensor_data.len(), expected_data.len());
        let tol = T::epsilon() * T::from_u32(100).unwrap_or_else(T::one);
        for (a, b) in tensor_data.iter().zip(expected_data.iter()) {
            assert!( (*a - *b).abs() < tol, "Data mismatch: expected {:?}, got {:?}. Diff: {:?}", expected_data, &*tensor_data, (*a - *b).abs());
        }
    }

    // --- Tests need rework to set gradients without private access --- 
    // TODO: Rework these tests after deciding on a public gradient setting method.

    /*
    #[test]
    fn test_sgd_zero_grad() {
        type TestFloat = f32;
        let mut p1 = Tensor::new(vec![1.0 as TestFloat, 2.0], vec![2]);
        let mut p2 = Tensor::new(vec![3.0 as TestFloat, 4.0], vec![2]);
        
        // Cannot set gradient easily for now
        // set_gradient_for_test(&p1, vec![0.1, 0.2]); 
        // Instead, maybe force a backward pass? Or add a public setter.
        // For now, let's just test zero_grad on tensors without grads initially.
        p1.zero_grad(); // Ensure it starts clean
        p2.zero_grad();

        let optim: SGD<TestFloat> = SGD::new(0.1);

        assert!(p1.grad().is_none(), "p1 should not have gradient initially");
        assert!(p2.grad().is_none(), "p2 should not have gradient initially");

        let mut params_slice = [&mut p1, &mut p2];
        optim.zero_grad(&mut params_slice);

        assert!(p1.grad().is_none(), "p1 gradient should remain None after zero_grad");
        assert!(p2.grad().is_none(), "p2 gradient should remain None after zero_grad");
    }

    #[test]
    fn test_sgd_step() {
        type TestFloat = f32;
        let mut p1 = Tensor::new(vec![1.0 as TestFloat, 2.0], vec![2]);
        let mut p2 = Tensor::new(vec![3.0 as TestFloat, 4.0], vec![1, 2]);
        let mut p3 = Tensor::new(vec![5.0 as TestFloat], vec![1]);

        // Cannot set gradient easily for now
        // set_gradient_for_test(&p1, vec![10.0, -20.0]);
        // set_gradient_for_test(&p2, vec![0.5, -0.5]);
        // We need a way to run backward() or similar to populate gradients for a real test.
        // For now, this test is invalid.

        // Placeholder: check if step runs without panic if grad is None
        let mut optim: SGD<TestFloat> = SGD::new(0.1);
        let initial_p1_data = p1.data().to_vec();
        let initial_p2_data = p2.data().to_vec();
        let initial_p3_data = p3.data().to_vec();

        let mut params_slice = [&mut p1, &mut p2, &mut p3];
        optim.step(&mut params_slice);

        // Check that parameters haven't changed since no gradient was present
        check_data_approx(&p1, &initial_p1_data);
        check_data_approx(&p2, &initial_p2_data);
        check_data_approx(&p3, &initial_p3_data);
    }
    */
} 