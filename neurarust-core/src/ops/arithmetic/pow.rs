// neurarust-core/src/ops/arithmetic/pow.rs

use crate::tensor::Tensor;
use crate::autograd::{BackwardOp, accumulate_gradient};
use crate::tensor_data::TensorData; 
use num_traits::{Pow, Zero, One}; // Pow for forward, Zero/One often needed
use std::ops::{Mul, AddAssign, Sub, Neg}; // For gradient calculation
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::collections::HashMap;
use std::iter::Sum as IterSum; // Add IterSum

// --- Backward Operation --- 

#[derive(Debug)]
struct PowBackward<T: Clone> {
    input: Tensor<T>, // Keep original input for gradient calculation x^(n-1)
    exponent: T,      // The scalar exponent n
    input_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for PowBackward<T>
where
    T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + Neg<Output=T> + IterSum + Default,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                let n = self.exponent;
                let one = T::one();
                let n_minus_1 = n - one; // Sub should work now
                
                let x_pow_n_minus_1_data: Vec<T> = self.input.data()
                    .iter()
                    .map(|&x| T::pow(x, n_minus_1)) // Use num_traits::Pow
                    .collect();
                let x_pow_n_minus_1 = Tensor::new(x_pow_n_minus_1_data, self.input.shape());

                let n_tensor = Tensor::new(vec![n], vec![]);
                let grad_factor = &n_tensor * &x_pow_n_minus_1; // Mul should work now

                let local_gradient = upstream_grad * &grad_factor; // Mul should work now
                local_gradient.set_requires_grad(false);

                accumulate_gradient(gradients, &self.input_ref, local_gradient);
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}


// --- Forward Operation --- 

impl<T> Tensor<T> {
    /// Raises each element of the tensor to the power of the given scalar exponent.
    /// Computes `x^n` element-wise.
    /// 
    /// # Arguments
    /// * `exponent` - The scalar exponent `n`.
    pub fn pow(&self, exponent: T) -> Tensor<T>
    where
        // Bounds for forward pass (Pow) and potential backward pass setup
        T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + Neg<Output=T> + IterSum + Default,
    {
        let input_td = self.borrow_tensor_data();
        let result_data: Vec<T> = input_td.data.iter()
            .map(|&x| T::pow(x, exponent)) // Use num_traits::Pow
            .collect();
        
        let result_shape = input_td.shape.clone();
        let requires_grad = input_td.requires_grad; // Check requires_grad from borrowed data
        
        // Store clones/refs needed for backward before dropping borrow
        let input_clone_for_backward = if requires_grad { Some(self.clone()) } else { None };
        let input_ref_for_backward = if requires_grad { Some(self.get_weak_ref()) } else { None };
        
        drop(input_td); // Drop borrow

        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = PowBackward {
                input: input_clone_for_backward.expect("Input clone missing for backward"), 
                exponent: exponent,
                input_ref: input_ref_for_backward.expect("Input ref missing for backward"),
                _phantom: PhantomData,
            };
            result.data.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::{Zero, One, Pow};
    use std::ops::{Sub, AddAssign, Mul, Neg}; // Add Neg
    use std::fmt::Debug;
    use std::iter::Sum as IterSum; // Add IterSum

    // Helper function with necessary bounds for pow tests
    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where 
        T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + PartialEq + Neg<Output=T> + IterSum + Default,
    {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_pow_forward() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![3];
        let t = Tensor::new(data, shape.clone());
        
        // Test integer power
        let result2 = t.pow(2.0_f32); // Square
        let expected_data2 = vec![1.0_f32, 4.0, 9.0];
        assert_eq!(result2.data().to_vec(), expected_data2);
        assert_eq!(result2.shape(), shape);
        assert!(!result2.requires_grad());

        // Test fractional power (requires f32/f64 usually)
        let result0_5 = t.pow(0.5_f32); // Square root
        let expected_data0_5 = vec![1.0_f32, 1.41421356, 1.73205081];
        assert!((result0_5.data()[0] - expected_data0_5[0]).abs() < 1e-6);
        assert!((result0_5.data()[1] - expected_data0_5[1]).abs() < 1e-6);
        assert!((result0_5.data()[2] - expected_data0_5[2]).abs() < 1e-6);
        assert_eq!(result0_5.shape(), shape);
    }

    #[test]
    fn test_pow_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        let result = t1.pow(3.0_f32);
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let t2 = Tensor::new(vec![3.0_f32], vec![1]);
        let result2 = t2.pow(2.0_f32);
        assert!(!result2.requires_grad());
        assert!(result2.grad_fn().is_none());
    }

    #[test]
    fn test_pow_backward() {
        let t1 = create_test_tensor_with_grad(vec![2.0_f32, 3.0], vec![2]);
        let exp = 3.0_f32;
        let result = t1.pow(exp); 
        
        let loss = result.sum(); 
        loss.backward(None); 

        let grad_t1_opt = t1.grad();
        assert!(grad_t1_opt.is_some(), "Gradient for t1 missing");
        let grad_t1 = grad_t1_opt.as_ref().unwrap();
        
        let expected_grad_data = vec![12.0_f32, 27.0];
        assert_eq!(grad_t1.shape(), vec![2]);
        assert!((grad_t1.data()[0] - expected_grad_data[0]).abs() < 1e-6);
        assert!((grad_t1.data()[1] - expected_grad_data[1]).abs() < 1e-6);
    }
} 