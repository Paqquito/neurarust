// neurarust-optim/src/sgd.rs

// Utiliser les types de neurarust_core
use neurarust_core::tensor::Tensor;
// Importer le trait Optimizer depuis lib.rs
use crate::Optimizer; 
use std::ops::{Mul, Sub, AddAssign};
use num_traits::FromPrimitive;
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
    pub fn new(lr: f64) -> Self {
        let lr_t = T::from_f64(lr).expect("Could not convert learning rate (lr) to type T.");
        SGD {
            // params: params.into_iter().collect(), // Removed
            lr: lr_t,
        }
    }
}

impl<T> Optimizer<T> for SGD<T> 
where
    T: Copy + Clone + Debug + FromPrimitive + 
       Sub<Output=T> + Mul<Output=T> + AddAssign + 'static, // Added 'static bound from Optimizer trait
{
    /// Performs a single optimization step (parameter update).
    fn step(&mut self, params: &mut [&mut Tensor<T>]) {
        for param in params {
            let grad_data_option = {
                let grad_option = param.borrow_grad();
                grad_option.as_ref().map(|grad_tensor| grad_tensor.data().to_vec())
            };

            if let Some(grad_data) = grad_data_option {
                let update_data: Vec<T> = grad_data.iter().map(|&g| g * self.lr).collect();
                let mut param_td_mut = param.borrow_tensor_data_mut();
                assert_eq!(param_td_mut.data.len(), update_data.len(), "Shape mismatch during SGD step");
                param_td_mut.data.iter_mut().zip(update_data.iter()).for_each(|(p, &u)| *p = *p - u);
            }
        }
    }

    /// Clears the gradients of all parameters managed by the optimizer.
    fn zero_grad(&self, params: &mut [&mut Tensor<T>]) {
        for param in params {
            let mut param_td_mut = param.borrow_tensor_data_mut(); 
            param_td_mut.grad = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import SGD
    use neurarust_core::tensor::Tensor; // Import Tensor from core
    use crate::Optimizer; // Import Optimizer trait

    // Helper to create f32 tensors
    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape)
    }
    
    // Helper to check approximate data equality
    fn check_data_approx(tensor: &Tensor<f32>, expected_data: &[f32]) {
        assert_eq!(tensor.data().len(), expected_data.len());
        for (a, b) in tensor.data().iter().zip(expected_data.iter()) {
            assert!((a - b).abs() < 1e-6, "Data mismatch: expected {:?}, got {:?}", expected_data, tensor.data());
        }
    }

    #[test]
    fn test_sgd_zero_grad() {
        let mut p1 = create_tensor_f32(vec![1., 2.], vec![2]);
        let mut p2 = create_tensor_f32(vec![3., 4.], vec![2]);
        // Give p1 an initial gradient
        p1.borrow_tensor_data_mut().grad = Some(create_tensor_f32(vec![0.1, 0.2], vec![2]));
        
        // Optimizer is created without params now
        let optim = SGD::new(0.1);

        assert!(p1.borrow_grad().is_some());
        assert!(p2.borrow_grad().is_none());

        // Pass params to zero_grad
        optim.zero_grad(&mut [&mut p1, &mut p2]);

        assert!(p1.borrow_grad().is_none(), "Grad of p1 should be None after zero_grad");
        assert!(p2.borrow_grad().is_none(), "Grad of p2 should be None after zero_grad");
    }

    #[test]
    fn test_sgd_step() {
        let mut p1 = create_tensor_f32(vec![1.0, 2.0], vec![2]);
        let mut p2 = create_tensor_f32(vec![3.0, 4.0], vec![1, 2]); // Different shape
        let mut p3 = create_tensor_f32(vec![5.0], vec![1]); // No gradient

        // Give gradients
        let grad1 = create_tensor_f32(vec![10.0, -20.0], vec![2]);
        let grad2 = create_tensor_f32(vec![0.5, -0.5], vec![1, 2]);
        p1.borrow_tensor_data_mut().grad = Some(grad1);
        p2.borrow_tensor_data_mut().grad = Some(grad2);

        // Create optimizer
        let mut optim = SGD::new(0.1); // lr = 0.1

        // Pass params to step
        optim.step(&mut [&mut p1, &mut p2, &mut p3]);

        // Check p1: p1 = p1 - lr * grad1 = [1, 2] - 0.1 * [10, -20] = [1, 2] - [1, -2] = [0, 4]
        check_data_approx(&p1, &[0.0, 4.0]);

        // Check p2: p2 = p2 - lr * grad2 = [3, 4] - 0.1 * [0.5, -0.5] = [3, 4] - [0.05, -0.05] = [2.95, 4.05]
        check_data_approx(&p2, &[2.95, 4.05]);

        // Check p3: no gradient, should not change
        check_data_approx(&p3, &[5.0]);
    }
} 