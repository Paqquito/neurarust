use crate::Optimizer;
use neurarust_core::nn::Parameter;
use num_traits::{Float, FromPrimitive, Zero, One, Pow};
use std::{collections::HashMap, fmt::Debug, iter::Sum as IterSum, default::Default};
use std::ops::{AddAssign, Sub, Neg, Mul, Div, Add, SubAssign};
use neurarust_core::error::NeuraRustError;

/// Adam optimizer implementation.
#[derive(Debug)]
pub struct Adam<T: Float> {
    params: Vec<Parameter<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    m: HashMap<*const (), Vec<T>>, // Store first moment vector (moving average of gradients)
    v: HashMap<*const (), Vec<T>>, // Store second moment vector (moving average of squared gradients)
    t: usize,                       // Timestep counter
}

impl<T> Adam<T>
where
    T: Float + FromPrimitive + Zero + 'static + std::fmt::Debug + Send + Sync,
{
    /// Creates a new Adam optimizer.
    ///
    /// # Arguments
    /// * `params` - A vector of parameters (tensors requiring gradients) to optimize.
    /// * `lr` - Learning rate (alpha).
    /// * `beta1` - Exponential decay rate for the first moment estimates.
    /// * `beta2` - Exponential decay rate for the second-moment estimates.
    /// * `eps` - Term added to the denominator to improve numerical stability.
    pub fn new(
        params: Vec<Parameter<T>>,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
    ) -> Self {
        let m = HashMap::new();
        let v = HashMap::new();
        Adam {
            params,
            lr,
            beta1,
            beta2,
            eps,
            m,
            v,
            t: 0,
        }
    }
}

impl<T> Optimizer<T> for Adam<T>
where
    // Add bounds required by Adam's implementation, even if covered by Optimizer trait bounds
    T: Float + FromPrimitive + Zero + One + Pow<T, Output = T> + Pow<i32, Output = T> 
    + AddAssign + Sub<Output = T> + Neg<Output = T> + Mul<Output = T> + Add<Output = T> 
    + Div<Output = T> + SubAssign + IterSum + Default + Debug + Copy + 'static + Send + Sync,
{
    fn step(&mut self) -> Result<(), NeuraRustError> {
        self.t += 1;
        let t_float = T::from_usize(self.t).unwrap_or_else(T::one); // Current timestep as float

        // Bias correction terms
        let beta1_t = self.beta1.powf(t_float);
        let beta2_t = self.beta2.powf(t_float);
        let lr_t = self.lr * (T::one() - beta2_t).sqrt() / (T::one() - beta1_t);

        for param_wrapper in &self.params {
            let param_tensor = &param_wrapper.0; // Access Tensor inside Parameter
            let param_id = param_tensor.id();
            
            // 1. Read gradient data using an immutable borrow
            let grad_data_opt: Option<Vec<T>> = {
                let param_data_immut = param_tensor.borrow_tensor_data(); // Use immutable borrow
                param_data_immut.grad.as_ref().map(|g| g.data().to_vec())
            };

            // 2. If gradient exists, update moments and param data using mutable borrow
            if let Some(grad_clone) = grad_data_opt { // Use grad_clone
                 let mut param_data_mut = param_tensor.borrow_tensor_data_mut(); // Mutable borrow here
                 let numel = grad_clone.len(); // Use grad_clone len
                 assert_eq!(param_data_mut.data.len(), numel, "Data-gradient length mismatch");

                 // Get or initialize moments for this parameter
                 let m_prev = self.m.entry(param_id).or_insert_with(|| vec![T::zero(); numel]);
                 let v_prev = self.v.entry(param_id).or_insert_with(|| vec![T::zero(); numel]);

                 for i in 0..numel {
                     let g = grad_clone[i]; // Use cloned gradient value

                     // Update biased first moment estimate
                     let m_new = self.beta1 * m_prev[i] + (T::one() - self.beta1) * g;
                     m_prev[i] = m_new;

                     // Update biased second raw moment estimate
                     let v_new = self.beta2 * v_prev[i] + (T::one() - self.beta2) * g * g;
                     v_prev[i] = v_new;

                     // Update parameters
                     let update = lr_t * m_new / (v_new.sqrt() + self.eps);
                     // Access data via mutable borrow
                     param_data_mut.data[i] = param_data_mut.data[i] - update;
                 }
             } else {
                // Optionally warn if a parameter has no gradient
                // eprintln!("Warning: Parameter {:?} has no gradient.", param_id);
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param_wrapper in &self.params {
            param_wrapper.0.zero_grad(); // Call zero_grad on the inner Tensor
        }
    }

    fn get_params(&self) -> Vec<Parameter<T>> {
        self.params.clone() // Clone the Vec<Parameter<T>>
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use neurarust_core::tensor::Tensor;
    // Removed unresolved import: use neurarust_core::assert_approx_eq; 

    // Helper to create Parameter for tests
    fn create_param<T: Clone + Float + std::fmt::Debug + 'static + Default + Zero + std::ops::AddAssign>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Parameter<T> {
        let tensor = Tensor::new_with_grad(data, shape).expect("Failed to create tensor");
        Parameter(tensor)
    }

    // Local helper for approximate comparison
    fn assert_local_approx_eq<T: Float + std::fmt::Debug>(a: &[T], b: &[T], tol: T) {
        assert_eq!(a.len(), b.len(), "Lengths mismatch");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((*x - *y).abs() <= tol, "Mismatch: {:?} vs {:?} with tol {:?}", x, y, tol);
        }
    }

    #[test]
    fn test_adam_step_basic() -> Result<(), NeuraRustError> { // Changed return type
        let param1_data = vec![1.0f32, 2.0];
        let param1_shape = vec![2];
        let param1 = create_param(param1_data.clone(), param1_shape.clone());

        let param2_data = vec![3.0f32];
        let param2_shape = vec![1];
        let param2 = create_param(param2_data.clone(), param2_shape.clone());

        let params = vec![param1.clone(), param2.clone()];
        // Call the corrected new function signature (5 args)
        let mut optimizer = Adam::new(params, 0.001, 0.9, 0.999, 1e-8);

        // Simulate some gradients
        let grad1 = Tensor::new(vec![0.1, -0.1], param1_shape.clone())?;
        let grad2 = Tensor::new(vec![0.5], param2_shape.clone())?;
        param1.0.borrow_tensor_data_mut().grad = Some(grad1);
        param2.0.borrow_tensor_data_mut().grad = Some(grad2);

        // Perform optimizer step
        optimizer.step()?;

        // Check updated parameters 
        let expected_param1 = vec![0.999f32, 2.001];
        let expected_param2 = vec![2.999f32];

        let updated_param1 = param1.0.data().to_vec();
        let updated_param2 = param2.0.data().to_vec();

        // Use approximate comparison
        let tolerance = 1e-6; // Adjusted tolerance slightly
        assert_local_approx_eq(&updated_param1, &expected_param1, tolerance);
        assert_local_approx_eq(&updated_param2, &expected_param2, tolerance);
        
        Ok(())
    }

    #[test]
    fn test_adam_zero_grad() -> Result<(), NeuraRustError> { // Changed return type
        let param1 = create_param(vec![1.0f32, 2.0], vec![2]);
        let params = vec![param1.clone()];
        let mut optimizer = Adam::new(params, 0.001, 0.9, 0.999, 1e-8);

        // Add a gradient
        let grad1 = Tensor::new(vec![0.1, -0.1], vec![2])?;
        param1.0.borrow_tensor_data_mut().grad = Some(grad1);
        assert!(param1.0.grad().is_some());

        optimizer.zero_grad();
        assert!(param1.0.grad().is_some()); // Grad tensor should still exist
        assert_eq!(param1.0.grad().unwrap().data().to_vec(), vec![0.0, 0.0]);

        Ok(())
    }
}