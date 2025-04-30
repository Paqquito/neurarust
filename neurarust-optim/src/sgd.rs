// neurarust-optim/src/sgd.rs

// Utiliser les types de neurarust_core
use neurarust_core::nn::Parameter; // Import Parameter
// Importer le trait Optimizer depuis lib.rs
use crate::Optimizer; 
// Import necessary traits for calculations
use std::ops::{SubAssign, Mul, Sub, Add, Div, Neg, AddAssign};
use num_traits::{Float, FromPrimitive, Zero, One, Pow};
use std::fmt::Debug;
use std::iter::Sum as IterSum;
// Import arithmetic ops from core
// Import correct sum function via sum module
use std::marker::PhantomData;
use neurarust_core::error::NeuraRustError;

/// Stochastic Gradient Descent optimizer.
#[derive(Debug)]
pub struct SGD<T: Float> {
    params: Vec<Parameter<T>>, // Store parameters
    lr: T,                  // Learning rate
    _marker: PhantomData<T>, // If T is not used elsewhere
}

impl<T> SGD<T>
where
    T: Float + Copy + Debug,
{
    /// Creates a new SGD optimizer.
    /// # Arguments
    /// * `params` - A vector of parameters to optimize.
    /// * `lr` - Learning rate.
    pub fn new(params: Vec<Parameter<T>>, lr: T) -> Self {
        SGD {
            params,
            lr,
            _marker: PhantomData,
        }
    }
}

// Implement the Optimizer trait for SGD
impl<T> Optimizer<T> for SGD<T>
where
    // Copy the bounds from the Optimizer trait definition
    T: Float + FromPrimitive + Zero + One + Pow<T, Output = T> + Pow<i32, Output = T> 
    + AddAssign + Sub<Output = T> + Neg<Output = T> + Mul<Output = T> + Add<Output = T> 
    + Div<Output = T> + SubAssign + IterSum + Default + Debug + Copy + 'static + Send + Sync,
{
    /// Performs a single optimization step using SGD.
    fn step(&mut self) -> Result<(), NeuraRustError> {
        for param_wrapper in &self.params { // Iterate over stored params
            let param_tensor = &param_wrapper.0; // Access Tensor inside Parameter

            // 1. Read gradient data using an immutable borrow
            let grad_data_opt: Option<Vec<T>> = {
                let param_data_immut = param_tensor.borrow_tensor_data();
                param_data_immut.grad.as_ref().map(|g| g.data().to_vec())
            };

            // 2. If gradient exists, update param data using a mutable borrow
            if let Some(grad_clone) = grad_data_opt {
                 let mut param_data_mut = param_tensor.borrow_tensor_data_mut(); // Mutable borrow here
                 let numel = param_data_mut.data.len();
                 assert_eq!(grad_clone.len(), numel, "Gradient length mismatch");
                 // Perform update: param = param - lr * grad
                 for i in 0..numel {
                     // Use the cloned gradient data
                     param_data_mut.data[i] = param_data_mut.data[i] - self.lr * grad_clone[i];
                 }
             } // else: parameter has no gradient, skip update
        }
        Ok(())
    }

    /// Clears the gradients of all parameters managed by this SGD optimizer.
    fn zero_grad(&mut self) {
        for param_wrapper in &self.params { // Iterate over stored params
            param_wrapper.0.zero_grad(); // Call zero_grad on the inner Tensor
        }
    }

    /// Returns a clone of the parameters managed by this optimizer.
    fn get_params(&self) -> Vec<Parameter<T>> {
        self.params.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurarust_core::tensor::Tensor;
    use std::ops::AddAssign;
    

    // Helper to create Parameter for tests
    fn create_param<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Parameter<T> 
    where 
        T: Float + Debug + Default + Zero + AddAssign + Copy + Clone + 'static // Bounds for Parameter::new
    { 
        let tensor = Tensor::new_with_grad(data, shape).expect("Failed to create tensor");
        Parameter::new(tensor)
    }

    #[test]
    fn test_sgd_step() -> Result<(), NeuraRustError> { 
        let param1 = create_param(vec![1.0f32, 2.0, 3.0], vec![3]);
        let param2 = create_param(vec![4.0f32], vec![1]);

        let params = vec![param1.clone(), param2.clone()];
        let mut optimizer = SGD::new(params, 0.1);

        // Simulate gradients
        let grad1 = Tensor::new(vec![0.5, -1.0, 0.1], vec![3])?;
        let grad2 = Tensor::new(vec![2.0], vec![1])?;
        param1.0.borrow_tensor_data_mut().grad = Some(grad1);
        param2.0.borrow_tensor_data_mut().grad = Some(grad2);

        // Perform optimizer step
        optimizer.step()?; // Remove params arg

        // Check updated parameters
        let expected_param1 = vec![1.0 - 0.1 * 0.5, 2.0 - 0.1 * (-1.0), 3.0 - 0.1 * 0.1];
        let expected_param2 = vec![4.0 - 0.1 * 2.0];

        assert_eq!(param1.0.data().to_vec(), expected_param1);
        assert_eq!(param2.0.data().to_vec(), expected_param2);
        
        Ok(())
    }

    #[test]
    fn test_sgd_zero_grad() -> Result<(), NeuraRustError> {
        let param1 = create_param(vec![1.0f32, 2.0], vec![2]);
        let params = vec![param1.clone()];
        let mut optimizer = SGD::new(params, 0.1);

        // Add a gradient
        let grad1 = Tensor::new(vec![0.1, -0.1], vec![2])?;
        param1.0.borrow_tensor_data_mut().grad = Some(grad1);
        assert!(param1.0.grad().is_some());

        optimizer.zero_grad(); // Remove params arg
        assert!(param1.0.grad().is_some()); // Grad tensor should still exist
        assert_eq!(param1.0.grad().unwrap().data().to_vec(), vec![0.0, 0.0]);

        Ok(())
    }
} 