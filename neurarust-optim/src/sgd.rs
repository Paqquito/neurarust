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
            // Clone the gradient tensor if it exists to avoid borrow conflicts
            if let Some(grad_tensor) = param.grad().map(|g_ref| g_ref.clone()) {
                // We need to calculate update = lr * grad element-wise
                // Get gradient data from the clone
                let grad_data = grad_tensor.data(); // Immutable borrow on the clone
                
                // Prepare storage for update data (same shape as gradient/param)
                let mut update_data_vec = Vec::with_capacity(grad_data.len());
                
                // Calculate update data: update_val = lr * grad_val
                for grad_val in grad_data.iter() {
                    // Requires T: Mul<Output = T>
                    update_data_vec.push(self.lr * *grad_val);
                }
                // Grad borrow (on clone) is dropped here

                // Apply update using data_mut and element-wise subtraction
                // Borrow param mutably *after* gradient processing
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
    // use crate::Optimizer; // No longer needed in tests directly if we use concrete type
    use num_traits::{Float, Zero, One, FromPrimitive};
    use std::fmt::Debug;
    use std::ops::{Mul, AddAssign, Neg};
    use std::iter::Sum;

    // Helper to check approximate data equality - Use relative tolerance based on T::epsilon()
    fn check_data_approx<T>(tensor: &Tensor<T>, expected_data: &[T])
    where
        T: Float + Debug + FromPrimitive,
    {
        let tensor_data = tensor.data();
        assert_eq!(tensor_data.len(), expected_data.len(), "Tensor length mismatch");
        let tol = T::epsilon() * T::from_u32(100).unwrap_or_else(T::one);
        for (a, b) in tensor_data.iter().zip(expected_data.iter()) {
            assert!( (*a - *b).abs() < tol, "Data mismatch: expected {:?}, got {:?}. Diff: {:?}", expected_data, &*tensor_data, (*a - *b).abs());
        }
    }

    // Helper function to generate a gradient on a tensor using backward()
    // Creates a simple graph: loss = (tensor * constant_tensor).sum()
    // backward() on loss populates tensor.grad() with values from constant_tensor.
    fn generate_gradient<T>(
        tensor: &Tensor<T>, 
        grad_values: Vec<T>
    )
    where 
        T: Mul<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + Neg<Output=T>,
    {
        assert_eq!(tensor.numel(), grad_values.len(), "Tensor numel must match grad_values length");
        tensor.set_requires_grad(true);
        let constant = Tensor::new(grad_values, tensor.shape()); // Constant tensor with desired grad values
        
        // Simple operation: multiply by constant and sum to get scalar loss
        let mul_result = tensor * &constant;
        let loss = mul_result.sum(); // sum() is defined in ops::reduction::sum

        // Ensure loss requires grad (should inherit)
        assert!(loss.requires_grad());

        // Run backward to populate gradients
        loss.backward(None); // Use default upstream grad of 1.0

        // Check that the gradient was populated
        assert!(tensor.grad().is_some(), "Gradient was not generated by backward pass");
    }


    #[test]
    fn test_sgd_zero_grad() {
        type TestFloat = f32;
        let p1_data = vec![1.0 as TestFloat, 2.0];
        let mut p1 = Tensor::new(p1_data, vec![2]);
        let mut p2 = Tensor::new(vec![3.0 as TestFloat, 4.0], vec![2]); // No grad needed
        
        // Generate gradient on p1
        let initial_grad_p1 = vec![0.1, 0.2];
        generate_gradient(&p1, initial_grad_p1.clone());
        assert!(p1.grad().is_some(), "p1 should have gradient after generate_gradient");
        check_data_approx(&p1.grad().unwrap(), &initial_grad_p1);
        assert!(p2.grad().is_none(), "p2 should not have gradient initially");

        let optim: SGD<TestFloat> = SGD::new(0.1);
        let mut params_slice = [&mut p1, &mut p2];
        optim.zero_grad(&mut params_slice);

        // Check state after zero_grad
        // Tensor::zero_grad zeros the data, doesn't set Option to None
        assert!(p1.grad().is_some(), "p1 gradient should still exist after zero_grad");
        check_data_approx(&p1.grad().unwrap(), &[0.0, 0.0]); // Check if zeroed
        assert!(p2.grad().is_none(), "p2 gradient should remain None after zero_grad");
    }

    #[test]
    fn test_sgd_step() {
        type TestFloat = f32;
        let initial_p1_data = vec![1.0 as TestFloat, 2.0];
        let initial_p2_data = vec![3.0 as TestFloat, 4.0];
        let initial_p3_data = vec![5.0 as TestFloat];

        let mut p1 = Tensor::new(initial_p1_data.clone(), vec![2]);
        let mut p2 = Tensor::new(initial_p2_data.clone(), vec![1, 2]); // Different shape
        let mut p3 = Tensor::new(initial_p3_data.clone(), vec![1]); // No gradient needed initially

        // Generate gradients using backward()
        let grad_p1_data = vec![10.0, -20.0];
        let grad_p2_data = vec![0.5, -0.5];
        generate_gradient(&p1, grad_p1_data);
        generate_gradient(&p2, grad_p2_data);

        // Create optimizer
        let mut optim: SGD<TestFloat> = SGD::new(0.1);

        let mut params_slice = [&mut p1, &mut p2, &mut p3];
        optim.step(&mut params_slice);

        // Check p1: p1 = [1, 2] - 0.1 * [10, -20] = [0, 4]
        let expected_p1_data = vec![0.0, 4.0];
        check_data_approx(&p1, &expected_p1_data);

        // Check p2: p2 = [3, 4] - 0.1 * [0.5, -0.5] = [2.95, 4.05]
        let expected_p2_data = vec![2.95, 4.05];
        check_data_approx(&p2, &expected_p2_data);

        // Check p3: no gradient, should not change
        check_data_approx(&p3, &initial_p3_data);
    }
} 