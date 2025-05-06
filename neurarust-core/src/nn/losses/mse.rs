// neurarust-core/src/nn/losses/mse.rs

use crate::tensor::Tensor;
use crate::ops;
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

        let diff = ops::arithmetic::sub_op(input, target)?;

        let sq_diff = ops::arithmetic::mul_op(&diff, &diff)?;
        
        let sum_val = ops::reduction::sum::sum(&sq_diff)?;

        let n = input.numel();
        let n_t = T::from_usize(n).ok_or_else(|| 
            NeuraRustError::InternalError("Failed to convert numel to tensor type T".to_string())
        )?;
        let n_tensor = Tensor::scalar(n_t);
        
        let two = T::from_f64(2.0).expect("Cannot create 2.0 from f64 for type T");

        let result = match self.reduction {
            Reduction::Sum => sum_val,
            Reduction::Mean => ops::arithmetic::div_op(&sum_val, &n_tensor)?,
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
            let diff = ops::arithmetic::sub_op(&input_tensor, &target_tensor)
                 .expect("Internal error: Failed subtraction in MSE backward");
            let _two = T::from_f64(2.0).expect("Failed to create 2.0");
            assert_eq!(upstream_grad.numel(), 1, "MSE Loss must be scalar for backward");
            let upstream_scalar = upstream_grad.borrow_data_buffer()[0];
            let factor = match self.reduction {
                Reduction::Sum => T::from_f64(2.0).unwrap() * upstream_scalar,
                Reduction::Mean => {
                     let n_t = T::from_usize(self.num_elements).expect("Failed to create n");
                     (T::from_f64(2.0).unwrap() * upstream_scalar) / n_t 
                },
                Reduction::None => unreachable!(),
            };

            let factor_tensor = Tensor::scalar(factor);
                 
            let final_local_grad_res = ops::arithmetic::mul_op(&factor_tensor, &diff);

            if input_tensor.requires_grad() {
                let final_local_grad = final_local_grad_res.as_ref()
                    .expect("Internal error: Failed scalar multiplication in MSE backward");
                crate::autograd::accumulate_gradient(gradients, &self.input_ref, final_local_grad.clone());
            }
            if target_tensor.requires_grad() {
                 let final_local_grad = final_local_grad_res
                      .expect("Internal error: Failed scalar multiplication (getting tensor for neg)");
                 let final_local_grad_neg = ops::arithmetic::neg_op(&final_local_grad)
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

#[cfg(test)]
#[path = "mse_test.rs"]
mod tests; 