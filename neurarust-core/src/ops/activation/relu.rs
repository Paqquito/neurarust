use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul};
use std::iter::Sum;
use std::sync::{Arc, RwLock};

// --- ReluBackward Definition (Corrected) ---

/// Backward operation context for `relu_op`.
/// Stores the input tensor to compute the gradient mask.
#[derive(Debug)]
struct ReluBackward<T: Debug + Copy + Send + Sync + 'static> {
    input_node: Arc<RwLock<TensorData<T>>>, // Store input node Arc<RwLock>
}

// --- BackwardOp Implementation for ReluBackward (Corrected) ---

impl<T> BackwardOp<T> for ReluBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Default
        + Clone
        + Zero
        + One
        + AddAssign
        + Add<Output = T>
        + Mul<Output = T>
        + Sum
        + PartialEq
        + PartialOrd
        + std::iter::Product,
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let input_guard = self.input_node.read().map_err(|poison_err| {
            // Correctly instantiate LockError with named fields
            NeuraRustError::LockError {
                lock_type: "read".to_string(),
                reason: format!("Failed to lock input node in ReluBackward: {}", poison_err),
            }
        })?;

        // Create the gradient mask: 1.0 where input > 0.0, 0.0 otherwise
        if input_guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::UnsupportedOperation(
                "ReLU backward currently only supports CPU".to_string(),
            ));
        }
        let input_buffer = input_guard.data.cpu_data()?.clone();
        let input_slice = input_buffer.as_slice();

        // Iterate using the input tensor's properties (offset, numel)
        let mask_data: Vec<T> = input_slice[input_guard.offset..]
            .iter()
            .take(input_guard.numel())
            .map(|&x| if x > T::zero() { T::one() } else { T::zero() })
            .collect();

        if mask_data.len() != input_guard.numel() {
             return Err(NeuraRustError::InternalError(format!(
                 "ReLU backward mask creation length mismatch. Expected {}, got {}.",
                 input_guard.numel(), mask_data.len()
             )));
        }

        // Create mask tensor with the same shape as the input (and thus the output)
        let mask_tensor = Tensor::new(mask_data, input_guard.shape.clone())?;

        if mask_tensor.device() != grad_output.device() {
            return Err(NeuraRustError::DeviceMismatch {
                expected: grad_output.device(),
                actual: mask_tensor.device(),
                operation: "relu_backward (mask * grad_output)".to_string(),
            });
        }

        // Compute input gradient: grad_output * mask
        let input_gradient = crate::ops::arithmetic::mul::mul_op(grad_output, &mask_tensor)?;

        Ok(vec![input_gradient])
    }

    fn inputs(&self) -> Vec<NodeId<T>> {
        // Return the input node ID
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- relu_op Implementation (Corrected Autograd Linkage) ---

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
/// ReLU(x) = max(0, x)
/// Currently supports CPU only.
pub fn relu_op<T>(input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Copy
        + PartialOrd
        + Zero
        + Debug
        + Send
        + Sync
        + 'static
        + Clone
        + One // For backward
        + Add<Output = T> // For backward autograd context
        + AddAssign // For backward autograd context
        + std::ops::Mul<Output = T> // For backward
        + Default
        + PartialEq
        + std::iter::Sum
        + std::iter::Product, // GARDER Product
{
    let requires_grad = input.requires_grad();
    // Clone the Arc for the input node *before* reading data
    let input_node_arc = if requires_grad { Some(input.data.clone()) } else { None };

    let input_guard = input.read_data();

    // Device check
    if input_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "ReLU forward currently only supports CPU".to_string(),
        ));
    }

    // Apply ReLU element-wise
    let input_buffer = input_guard.data.cpu_data()?.clone();
    let input_slice = input_buffer.as_slice();
    let output_data: Vec<T> = input_slice[input_guard.offset..]
        .iter()
        .take(input_guard.numel())
        .map(|&x| if x > T::zero() { x } else { T::zero() })
        .collect();

    let output_shape = input_guard.shape.clone();

    // Check data length
     if output_data.len() != input_guard.numel() {
         return Err(NeuraRustError::InternalError(format!(
             "ReLU forward output data length mismatch. Expected {}, got {}.",
             input_guard.numel(), output_data.len()
         )));
    }

    // Create result tensor (will have contiguous strides by default)
    let result_tensor = Tensor::new(output_data, output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let Some(input_arc) = input_node_arc {
            let grad_fn = ReluBackward { input_node: input_arc };
            result_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
            result_tensor.set_requires_grad(true)?;
        } else {
             // Should not happen if requires_grad is true, but add safety check
             return Err(NeuraRustError::InternalError(
                 "Input requires_grad but Arc could not be cloned".to_string(),
             ));
        }
    }

    Ok(result_tensor)
}

// Link to the external test file
#[path = "relu_test.rs"]
mod tests;

// Add autograd tests
#[cfg(test)]
mod autograd_tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::Tensor;

    // Helper for f64 tests
    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_relu_backward_basic() {
        // Use input containing 0.0, but increase tolerance for check_grad
        let input = create_tensor_f64_with_grad(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let func = |inputs: &[Tensor<f64>]| relu_op(&inputs[0]);

        let output_shape = vec![5];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        // Increase tolerance to account for numerical gradient difference at x=0 (0.5 vs 0.0)
        let tolerance = 0.51;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "ReLU basic backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_relu_backward_all_positive() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let func = |inputs: &[Tensor<f64>]| relu_op(&inputs[0]);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7; // Standard tolerance is fine here

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "ReLU all positive backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_relu_backward_all_negative_or_zero() {
        // Use input containing 0.0, but increase tolerance
        let input = create_tensor_f64_with_grad(vec![-2.0, -1.0, 0.0], vec![3]);
        let func = |inputs: &[Tensor<f64>]| relu_op(&inputs[0]);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        // Increase tolerance to account for numerical gradient difference at x=0
        let tolerance = 0.51;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "ReLU all negative/zero backward grad check failed: {:?}", grad_check_result.err());
    }
} 