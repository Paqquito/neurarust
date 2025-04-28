// neurarust-core/src/nn/losses/mse.rs

use crate::tensor::Tensor;
use num_traits::{Zero, One, FromPrimitive};
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div, AddAssign, Neg};
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::marker::PhantomData;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::iter::Sum as IterSum;
use std::collections::HashMap;

/// Specifies the reduction to apply to the output:
/// 'none' | 'mean' | 'sum'
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    None, // Not yet implemented
    Mean,
    Sum,
}

/// Computes the mean squared error (squared L2 norm) between each element
/// in the input `input` and target `target`.
///
/// Formula: loss(input, target) = mean or sum of (input_i - target_i)^2
#[derive(Debug, Clone)]
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
}

// --- Forward Pass --- 
impl MSELoss {
    pub fn forward<T>(&self, input: &Tensor<T>, target: &Tensor<T>) -> Tensor<T> 
    where
        T: Debug + Copy + Clone + Zero + One + FromPrimitive + PartialEq + 
           Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
           Add<Output = T> + AddAssign + IterSum + 'static + 
           Neg<Output = T> + Default,
    {
        assert_eq!(input.shape(), target.shape(), "Input and target shapes must match for MSELoss");
        
        let diff = input - target;
        let sq_diff = &diff * &diff;
        
        let requires_grad = input.requires_grad();
        let n = input.numel(); 

        let loss_val_tensor = match self.reduction {
            Reduction::Sum => sq_diff.sum(),
            Reduction::Mean => { 
                 // Restore Mean reduction
                 let sum_val = sq_diff.sum(); 
                 let n_t = T::from_usize(n).expect("Could not convert numel to tensor type T");
                 let n_tensor = Tensor::new(vec![n_t], vec![1]);
                 let result: Tensor<T> = &sum_val / &n_tensor; // Should work now
                 result
            },
            Reduction::None => unreachable!(), 
        };
        
        if requires_grad {
            // Setup MSELossBackward call
            let grad_fn = MSELossBackward {
                input_ref: input.get_weak_ref(),
                // target_ref: target.get_weak_ref(), // Target ref not strictly needed if diff is stored
                diff: diff, // Store diff for backward calculation
                reduction: self.reduction,
                num_elements: n,
                _phantom: PhantomData::<T>,
            };
            loss_val_tensor.data.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        } 
        
        loss_val_tensor
    }
}

// --- Backward Operation ---

#[derive(Debug)]
struct MSELossBackward<T> {
    diff: Tensor<T>, // Store difference (input - target) from forward pass
    input_ref: Weak<RefCell<TensorData<T>>>, // Ref to input tensor data
    reduction: Reduction,
    num_elements: usize, // Store N for mean reduction normalization
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MSELossBackward<T>
where
    T: Debug + Copy + Clone + Zero + One + FromPrimitive + PartialEq + 
       Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
       Add<Output = T> + AddAssign + IterSum + 'static + Neg<Output=T> + Default,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        assert_eq!(upstream_grad.numel(), 1, "Upstream grad for Loss must be scalar");
        let upstream_scalar = upstream_grad.data()[0];

        if let Some(input_rc) = self.input_ref.upgrade() {
            let needs_grad = input_rc.borrow().requires_grad;
            if needs_grad { 
                drop(input_rc); 

                let two = T::from_f32(2.0).expect("Could not convert 2.0 to tensor type T");
                let scalar_factor = two * upstream_scalar;
                let diff_data = self.diff.data();
                let diff_shape = self.diff.shape();
                
                let final_grad_data: Vec<T> = match self.reduction {
                    Reduction::Sum => {
                         diff_data.iter()
                                 .map(|&d| scalar_factor * d)
                                 .collect()
                    },
                    Reduction::Mean => {
                        let n_t = T::from_usize(self.num_elements).expect("Could not convert numel to tensor type T");
                        diff_data.iter()
                                 .map(|&d| (scalar_factor * d) / n_t)
                                 .collect()
                    },
                    Reduction::None => unreachable!(), 
                };
                
                let final_local_grad = Tensor::new(final_grad_data, diff_shape);

                crate::autograd::accumulate_gradient(gradients, &self.input_ref, final_local_grad);
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import MSELoss, Reduction
    use crate::tensor::Tensor;

    // Helper
    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape)
    }
    fn create_tensor_f32_grad(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new_with_grad(data, shape)
    }

    // Helper to check tensor values approximately
    fn check_tensor_val_approx(tensor: &Tensor<f32>, expected_val: f32) {
        assert_eq!(tensor.numel(), 1, "Tensor should be scalar");
        assert!((tensor.data()[0] - expected_val).abs() < 1e-6, "Tensor value mismatch: expected {}, got {}", expected_val, tensor.data()[0]);
    }
    
    // Helper to check tensor data approximately
    fn check_tensor_data_approx(tensor: &Tensor<f32>, expected_data: &[f32]) {
        assert_eq!(tensor.data().len(), expected_data.len(), "Data length mismatch");
        for (a, b) in tensor.data().iter().zip(expected_data.iter()) {
            assert!((a - b).abs() < 1e-6, "Data mismatch: expected {:?}, got {:?}", expected_data, tensor.data());
        }
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
    #[should_panic(expected = "Input and target shapes must match for MSELoss")]
    fn test_mse_loss_shape_mismatch() {
        let loss = MSELoss::default();
        let input = create_tensor_f32(vec![1., 2.], vec![2]);
        let target = create_tensor_f32(vec![1., 2., 3.], vec![3]);
        loss.forward(&input, &target); // Should panic
    }

    #[test]
    fn test_mse_loss_forward_mean() {
        let loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        // diff = [-0.5, -0.5, -0.5, -0.5]
        // sq_diff = [0.25, 0.25, 0.25, 0.25]
        // sum = 1.0
        // mean = 1.0 / 4 = 0.25
        let result = loss.forward(&input, &target);
        check_tensor_val_approx(&result, 0.25);
        assert!(!result.requires_grad());
    }
    
    #[test]
    fn test_mse_loss_forward_sum() {
        let loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        // diff = [-0.5, -0.5, -0.5, -0.5]
        // sq_diff = [0.25, 0.25, 0.25, 0.25]
        // sum = 1.0
        let result = loss.forward(&input, &target);
        check_tensor_val_approx(&result, 1.0);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_mse_loss_backward_mean() {
        let loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        // diff = [-0.5, -0.5, -0.5, -0.5]
        let result = loss.forward(&input, &target); // loss = 0.25
        assert!(result.requires_grad());
        
        result.backward(None); // upstream grad = 1.0

        // Expected grad = 2 * diff / N = 2 * [-0.5, -0.5, -0.5, -0.5] / 4
        // = [-1.0, -1.0, -1.0, -1.0] / 4 = [-0.25, -0.25, -0.25, -0.25]
        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), vec![2, 2]);
        check_tensor_data_approx(&grad, &[-0.25, -0.25, -0.25, -0.25]);
    }
    
    #[test]
    fn test_mse_loss_backward_sum() {
        let loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        // diff = [-0.5, -0.5, -0.5, -0.5]
        let result = loss.forward(&input, &target); // loss = 1.0
        assert!(result.requires_grad());
        
        result.backward(None); // upstream grad = 1.0

        // Expected grad = 2 * diff = 2 * [-0.5, -0.5, -0.5, -0.5]
        // = [-1.0, -1.0, -1.0, -1.0]
        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), vec![2, 2]);
        check_tensor_data_approx(&grad, &[-1.0, -1.0, -1.0, -1.0]);
    }
} 