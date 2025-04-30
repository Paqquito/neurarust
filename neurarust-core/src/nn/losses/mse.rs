// neurarust-core/src/nn/losses/mse.rs

use crate::tensor::Tensor;
use crate::ops::arithmetic::{sub, mul, div, neg};
use crate::ops::reduction::sum::sum_axes;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::ops::{Sub, Mul, AddAssign, Neg, Div, Add};
use num_traits::{Zero, One, FromPrimitive};
use std::iter::Sum;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::collections::HashMap;
use crate::error::NeuraRustError;

/// Specifies the reduction to apply to the output:
/// 'none' | 'mean' | 'sum'
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    None, // Not yet implemented
    Mean,
    Sum,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

/// Computes the mean squared error (squared L2 norm) between each element
/// in the input `input` and target `target`.
///
/// Formula: loss(input, target) = mean or sum of (input_i - target_i)^2
#[derive(Debug, Default, Clone)]
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    /// Creates a new MSELoss module.
    pub fn new(reduction: Reduction) -> Self {
        // TODO: Add support for Reduction::None
        if reduction == Reduction::None {
            unimplemented!("Reduction::None is not yet supported for MSELoss");
        }
        MSELoss { reduction }
    }

    /// Default MSELoss with mean reduction.
    pub fn default() -> Self {
        MSELoss { reduction: Reduction::Mean }
    }

    /// Calculates the Mean Squared Error loss.
    /// Returns a Result wrapping a scalar Tensor (0-dim) or an error.
    pub fn forward<T>(&mut self, input: &Tensor<T>, target: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError> 
    where 
        T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + Neg<Output=T> 
         + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + PartialEq + FromPrimitive,
    {
        if input.shape() != target.shape() {
            return Err(NeuraRustError::IncompatibleShapes {
                shape1: input.shape(),
                shape2: target.shape(),
            });
        }

        let diff = sub(input, target)?;

        let sq_diff = mul(&diff, &diff)?;
        
        let sum_val = sum_axes(&sq_diff, &[], false)?;

        let n = input.numel();
        let n_t = T::from_usize(n).ok_or_else(|| 
            NeuraRustError::InternalError("Failed to convert numel to tensor type T".to_string())
        )?;
        let n_tensor = Tensor::scalar(n_t);
        
        // Correct conversion using from_f64
        let two = T::from_f64(2.0).expect("Cannot create 2.0 from f64 for type T");
        let two_tensor = Tensor::scalar(two);

        let result = match self.reduction {
            Reduction::Sum => sum_val,
            Reduction::Mean => div(&sum_val, &n_tensor)?,
            Reduction::None => unreachable!(),
        };

        if input.requires_grad() || target.requires_grad() {
            let grad_fn = MSEBackward {
                input_ref: input.get_weak_ref(),
                target_ref: target.get_weak_ref(),
                num_elements: n,
                reduction: self.reduction,
                _phantom: PhantomData,
            };
            result.set_grad_fn(Some(Rc::new(grad_fn)));
        }

        Ok(result)
    }
}

// --- Backward Operation ---

#[derive(Debug)]
struct MSEBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    target_ref: Weak<RefCell<TensorData<T>>>,
    num_elements: usize,
    reduction: Reduction,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MSEBackward<T> 
where 
    T: Sub<Output = T> + Div<Output = T> + Mul<Output = T> + Neg<Output=T> 
         + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + PartialEq + FromPrimitive,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) { 
        if let (Some(input_rc), Some(target_rc)) = (self.input_ref.upgrade(), self.target_ref.upgrade()) {
            let input_tensor = Tensor { data: input_rc.clone() };
            let target_tensor = Tensor { data: target_rc.clone() };
            let diff = sub(&input_tensor, &target_tensor)
                 .expect("Internal error: Failed subtraction in MSE backward");
            let two = T::from_f64(2.0).expect("Failed to create 2.0");
            assert_eq!(upstream_grad.numel(), 1, "MSE Loss must be scalar for backward");
            let upstream_scalar = upstream_grad.data()[0];
            let factor = match self.reduction {
                Reduction::Sum => two * upstream_scalar,
                Reduction::Mean => {
                     let n_t = T::from_usize(self.num_elements).expect("Failed to create n");
                     (two * upstream_scalar) / n_t 
                },
                Reduction::None => unreachable!(),
            };

            let factor_tensor = Tensor::scalar(factor);
                 
            let final_local_grad_res = mul(&factor_tensor, &diff);

            if input_tensor.requires_grad() {
                let final_local_grad = final_local_grad_res.as_ref()
                    .expect("Internal error: Failed scalar multiplication in MSE backward");
                crate::autograd::accumulate_gradient(gradients, &self.input_ref, final_local_grad.clone());
            }
            if target_tensor.requires_grad() {
                 let final_local_grad = final_local_grad_res
                      .expect("Internal error: Failed scalar multiplication (getting tensor for neg)");
                 let final_local_grad_neg = neg(&final_local_grad)
                      .expect("Internal error: Failed negation for target grad");
                crate::autograd::accumulate_gradient(gradients, &self.target_ref, final_local_grad_neg);
            }
        } else {
             eprintln!("MSEBackward: Input or Target tensor weak reference expired.");
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone(), self.target_ref.clone()]
    }
}

// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::error::NeuraRustError;
    use num_traits::{Zero, One, FromPrimitive};
    use std::ops::{Sub, Mul, AddAssign, Neg, Div, Add};
    use std::iter::Sum;
    use std::fmt::Debug;
    
    // Helper functions
    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    fn create_tensor_f32_grad(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }
    fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "assertion failed: `(left ≈ right)` (left: `{}`, right: `{}`)", a, b);
    }

    #[test]
    fn test_mse_loss_forward() -> Result<(), NeuraRustError> {
        let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5], vec![3]);
        let loss = mse.forward(&input, &target)?;
        assert_eq!(loss.shape(), Vec::<usize>::new());
        assert_approx_eq(loss.data()[0], 0.25, 1e-6);
        assert!(!loss.requires_grad());
        Ok(())
    }
    
     #[test]
    fn test_mse_loss_forward_shape_mismatch() {
        let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32(vec![1.5, 2.5], vec![2]);
        let result = mse.forward(&input, &target);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::IncompatibleShapes { .. }));
    }

    #[test]
    fn test_mse_loss_backward() -> Result<(), NeuraRustError> {
        let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5], vec![3]);
        let loss = mse.forward(&input, &target)?;
        assert!(loss.requires_grad());
        assert_eq!(loss.shape(), Vec::<usize>::new());
        assert_approx_eq(loss.data()[0], 0.25, 1e-6);
        
        loss.backward(None); 
        
        let grad_input_opt = input.grad();
        assert!(grad_input_opt.is_some());
        let grad_input = grad_input_opt.as_ref().unwrap();
        assert_eq!(grad_input.shape(), vec![3]);
        assert_approx_eq(grad_input.data()[0], -1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_input.data()[1], -1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_input.data()[2], -1.0 / 3.0, 1e-6);
        assert!(target.grad().is_none());
        Ok(())
    }
    
    #[test]
    fn test_mse_loss_backward_target_grad() -> Result<(), NeuraRustError> {
         let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32_grad(vec![1.5, 2.5, 3.5], vec![3]); 
        let loss = mse.forward(&input, &target)?;
        assert!(loss.requires_grad());
        
        loss.backward(None);
        
        assert!(input.grad().is_none());
        let grad_target_opt = target.grad();
        assert!(grad_target_opt.is_some());
        let grad_target = grad_target_opt.as_ref().unwrap();
        assert_eq!(grad_target.shape(), vec![3]);
        assert_approx_eq(grad_target.data()[0], 1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_target.data()[1], 1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_target.data()[2], 1.0 / 3.0, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_creation() {
        let mse_mean = MSELoss::new(Reduction::Mean);
        assert_eq!(mse_mean.reduction, Reduction::Mean);
        
        let mse_sum = MSELoss::new(Reduction::Sum);
        assert_eq!(mse_sum.reduction, Reduction::Sum);

        let mse_default = MSELoss::default();
        assert_eq!(mse_default.reduction, Reduction::Mean);
    }

    #[test]
    #[should_panic(expected = "not yet supported")]
    fn test_mse_loss_none_panic() {
        let _ = MSELoss::new(Reduction::None);
    }

    #[test]
    fn test_mse_loss_shape_mismatch() {
        let mut loss = MSELoss::default();
        let input = create_tensor_f32(vec![1., 2.], vec![2]);
        let target = create_tensor_f32(vec![1., 2., 3.], vec![3]);
        let result = loss.forward(&input, &target);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::IncompatibleShapes { shape1, shape2 } => {
                assert_eq!(shape1, vec![2]);
                assert_eq!(shape2, vec![3]);
            }
            _ => panic!("Expected IncompatibleShapes error, got something else."),
        }
    }

    #[test]
    fn test_mse_loss_forward_mean() {
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        assert_approx_eq(result.data()[0], 0.25, 1e-6);
        assert!(!result.requires_grad());
    }
    
    #[test]
    fn test_mse_loss_forward_sum() {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        assert_approx_eq(result.data()[0], 1.0, 1e-6);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_mse_loss_backward_mean() {
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        assert!(result.requires_grad());
        
        result.backward(None);

        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), vec![2, 2]);
        assert_approx_eq(grad.data()[0], -0.25, 1e-6);
        assert_approx_eq(grad.data()[1], -0.25, 1e-6);
        assert_approx_eq(grad.data()[2], -0.25, 1e-6);
        assert_approx_eq(grad.data()[3], -0.25, 1e-6);
    }
    
    #[test]
    fn test_mse_loss_backward_sum() {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        assert!(result.requires_grad());
        
        result.backward(None);

        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), vec![2, 2]);
        assert_approx_eq(grad.data()[0], -1.0, 1e-6);
        assert_approx_eq(grad.data()[1], -1.0, 1e-6);
        assert_approx_eq(grad.data()[2], -1.0, 1e-6);
        assert_approx_eq(grad.data()[3], -1.0, 1e-6);
    }

    #[test]
    fn test_mse_loss_reduction_mean() -> Result<(), NeuraRustError> {
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result_tensor = loss.forward(&input, &target)?;
        assert_approx_eq(result_tensor.data()[0], 0.25, 1e-6);
        assert!(!result_tensor.requires_grad());
        Ok(())
    }

    #[test]
    fn test_mse_loss_reduction_sum() -> Result<(), NeuraRustError> {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result_tensor = loss.forward(&input, &target)?;
        assert_approx_eq(result_tensor.data()[0], 1.0, 1e-6);
        assert!(!result_tensor.requires_grad());
        Ok(())
    }

    #[test]
    fn test_mse_loss_backward_reduction_mean() -> Result<(), NeuraRustError> {
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target)?;
        assert!(result.requires_grad()); 
        result.backward(None);
        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), vec![2, 2]);
        let expected_grad = vec![-0.25, -0.25, -0.25, -0.25];
        grad.data().iter().zip(expected_grad.iter()).for_each(|(a, e)| assert_approx_eq(*a, *e, 1e-6));
        Ok(())
    }

    #[test]
    fn test_mse_loss_backward_reduction_sum() -> Result<(), NeuraRustError> {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target)?;
        assert!(result.requires_grad()); 
        result.backward(None);
        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), vec![2, 2]);
        let expected_grad = vec![-1.0, -1.0, -1.0, -1.0];
         grad.data().iter().zip(expected_grad.iter()).for_each(|(a, e)| assert_approx_eq(*a, *e, 1e-6));
        Ok(())
    }

    #[test]
    fn test_mse_loss_no_grad_target_requires_grad() -> Result<(), NeuraRustError> {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32(vec![1.0], vec![1]); // No grad on input
        let target = create_tensor_f32_grad(vec![1.5], vec![1]); // Grad on target

        // Use fallible creation
        let result_tensor = loss.forward(&input, &target)?;
        assert_eq!(result_tensor.data().to_vec(), vec![0.25]);
        // Result should require grad because target requires grad
        assert!(result_tensor.requires_grad()); 
        Ok(())
    }

    #[test]
    fn test_mse_loss_no_grad_input_requires_grad() -> Result<(), NeuraRustError> {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32_grad(vec![1.0], vec![1]); // Grad on input
        let target = create_tensor_f32(vec![1.5], vec![1]); // No grad on target

        // Use fallible creation
        let result_tensor = loss.forward(&input, &target)?;
        assert_eq!(result_tensor.data().to_vec(), vec![0.25]);
         // Result should require grad because input requires grad
        assert!(result_tensor.requires_grad());
        Ok(())
    }
} 