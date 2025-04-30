// neurarust-core/src/ops/arithmetic/pow.rs

use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData; 
use num_traits::{Pow, Zero, One}; // Pow for forward, Zero/One often needed
use std::ops::{Mul, AddAssign, Sub, Neg}; // For gradient calculation
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::collections::HashMap;
use std::iter::Sum as IterSum; // Add IterSum
use crate::error::NeuraRustError;
use crate::ops::arithmetic::{mul::mul, sub::sub}; // Import necessary ops

// Define the public, fallible function

/// Raises each element of the tensor to the power of the given scalar exponent.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn pow_scalar<T>(
    base: &Tensor<T>, 
    exponent: T
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + Neg<Output=T> + IterSum + Default + PartialEq,
{
    let input_td = base.borrow_tensor_data();
    let result_data: Vec<T> = input_td.data.iter()
        .map(|&x| T::pow(x, exponent)) // Use num_traits::Pow
        .collect();
    
    let result_shape = input_td.shape.clone();
    let requires_grad = input_td.requires_grad;
    
    let input_clone_for_backward = if requires_grad { Some(base.clone()) } else { None };
    let input_ref_for_backward = if requires_grad { Some(base.get_weak_ref()) } else { None };
    
    drop(input_td);

    // Use ? for Tensor::new error
    let result = Tensor::new(result_data, result_shape)?;

    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = PowBackward {
            input: input_clone_for_backward.expect("Input clone missing for backward"), 
            exponent: exponent,
            input_ref: input_ref_for_backward.expect("Input ref missing for backward"),
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }
    Ok(result)
}

// --- Backward Operation --- 

#[derive(Debug)]
struct PowBackward<T: Clone> {
    input: Tensor<T>,
    exponent: T,
    input_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for PowBackward<T>
where
    T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + Neg<Output=T> + IterSum + Default + PartialEq,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                let n = self.exponent;
                let one = T::one();
                
                // Use imported `sub` function
                let n_minus_1 = sub(&Tensor::scalar(n), &Tensor::scalar(one))
                    .expect("Internal error: Failed to calculate n-1 in pow backward");
                let n_minus_1_scalar = n_minus_1.data()[0]; 

                let x_pow_n_minus_1_data: Vec<T> = self.input.data()
                    .iter()
                    .map(|&x| T::pow(x, n_minus_1_scalar)) 
                    .collect();
                let x_pow_n_minus_1 = Tensor::new(x_pow_n_minus_1_data, self.input.shape())
                     .expect("Internal error: Failed to create x^(n-1) tensor in pow backward");

                let n_tensor = Tensor::scalar(n);
                
                // Use imported `mul` function
                let grad_factor = mul(&n_tensor, &x_pow_n_minus_1)
                     .expect("Internal error: Failed to calculate grad_factor in pow backward");

                // Use imported `mul` function
                let local_gradient = mul(upstream_grad, &grad_factor)
                     .expect("Internal error: Failed to calculate local_gradient in pow backward");
                local_gradient.set_requires_grad(false);

                // Assuming accumulate_gradient is available from autograd module
                crate::autograd::accumulate_gradient(gradients, &self.input_ref, local_gradient);
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}


// --- Tensor Method (calls fallible function) --- 

impl<T> Tensor<T> {
    pub fn pow(&self, exponent: T) -> Tensor<T>
    where
        T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + Neg<Output=T> + IterSum + Default + PartialEq,
    {
        // Call the fallible pow_scalar function and panic on error
        pow_scalar(self, exponent)
            .unwrap_or_else(|e| panic!("Tensor power operation failed: {:?}", e))
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*; // Import the new `pow_scalar` function
    use crate::Tensor;
    use num_traits::{Zero, One, Pow};
    use std::ops::{Sub, AddAssign, Mul, Neg};
    use std::fmt::Debug;
    use std::iter::Sum as IterSum;
     // Import NeuraRustError

    // Helper function needs PartialEq bound now due to backward changes
    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where 
        T: Pow<T, Output = T> + Mul<Output=T> + AddAssign + Copy + Clone + Debug + 'static + One + Zero + Sub<Output=T> + PartialEq + Neg<Output=T> + IterSum + Default,
    {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }

    #[test]
    fn test_pow_forward() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![3];
        let t = Tensor::new(data, shape.clone()).expect("Test setup failed");
        
        // Use fallible pow_scalar
        let result2 = pow_scalar(&t, 2.0_f32);
        assert!(result2.is_ok());
        let res2_tensor = result2.unwrap();
        let expected_data2 = vec![1.0_f32, 4.0, 9.0];
        assert_eq!(res2_tensor.data().to_vec(), expected_data2);
        assert_eq!(res2_tensor.shape(), shape);
        assert!(!res2_tensor.requires_grad());

        let result0_5 = pow_scalar(&t, 0.5_f32);
        assert!(result0_5.is_ok());
        let res0_5_tensor = result0_5.unwrap();
        let expected_data0_5 = vec![1.0_f32, 1.41421356, 1.73205081];
        assert!((res0_5_tensor.data()[0] - expected_data0_5[0]).abs() < 1e-6);
        assert!((res0_5_tensor.data()[1] - expected_data0_5[1]).abs() < 1e-6);
        assert!((res0_5_tensor.data()[2] - expected_data0_5[2]).abs() < 1e-6);
        assert_eq!(res0_5_tensor.shape(), shape);
        
        // Test the Tensor method .pow() which should panic on error (none here)
        let result_method = t.pow(2.0_f32);
        assert_eq!(result_method.data().to_vec(), expected_data2);
    }

    #[test]
    fn test_pow_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        // Use fallible pow_scalar
        let result = pow_scalar(&t1, 3.0_f32).unwrap();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let t2 = Tensor::new(vec![3.0_f32], vec![1]).expect("Test setup failed");
        let result2 = pow_scalar(&t2, 2.0_f32).unwrap();
        assert!(!result2.requires_grad());
        assert!(result2.grad_fn().is_none());
    }

    #[test]
    fn test_pow_backward() {
        let t1 = create_test_tensor_with_grad(vec![2.0_f32, 3.0], vec![2]);
        let exp = 3.0_f32;
        // Use fallible pow_scalar
        let result = pow_scalar(&t1, exp).expect("Pow failed in backward test setup"); 
        
        // Use tensor method sum (assuming it exists and works)
        // TODO: Refactor sum to return Result if needed
        let loss = result.sum(); 
        loss.backward(None); 

        let grad_t1_opt = t1.grad();
        assert!(grad_t1_opt.is_some(), "Gradient for t1 missing");
        let grad_t1 = grad_t1_opt.as_ref().unwrap();
        
        // d(x^n)/dx = n*x^(n-1)
        // n=3, n-1=2
        // grad = upstream * (3 * x^2)
        // If upstream is 1 (from sum): grad = [1 * 3 * 2^2, 1 * 3 * 3^2] = [12, 27]
        let expected_grad_data = vec![12.0_f32, 27.0];
        assert_eq!(grad_t1.shape(), vec![2]);
        assert!((grad_t1.data()[0] - expected_grad_data[0]).abs() < 1e-6);
        assert!((grad_t1.data()[1] - expected_grad_data[1]).abs() < 1e-6);
    }
} 