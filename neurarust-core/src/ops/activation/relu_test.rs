#[cfg(test)]
use crate::tensor::{self, Tensor};
use crate::ops::activation::relu::relu_op;
use crate::error::NeuraRustError;
use crate::utils::testing::check_tensor_near;
use crate::autograd::grad_check::{check_grad, GradCheckError};

// --- Merged Tests (using F32) ---

#[test]
fn test_relu_forward() -> Result<(), NeuraRustError> {
    let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2])?;
    let output = relu_op(&input)?;
    let expected_data = vec![0.0, 0.0, 1.0, 2.0];
    check_tensor_near(&output, &[2, 2], &expected_data, 1e-9);
    Ok(())
}

#[test]
fn test_relu_backward() -> Result<(), NeuraRustError> {
    let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2])?;
    input.set_requires_grad(true)?;

    let output = relu_op(&input)?;
    assert!(output.requires_grad());
    assert!(output.grad_fn().is_some());

    let grad_output = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2])?;

    let grad_fn = output.grad_fn().unwrap();
    let grad_inputs = grad_fn.backward(&grad_output)?;

    assert_eq!(grad_inputs.len(), 1);
    let grad_input = &grad_inputs[0];
    let expected_grad_input = vec![0.0, 0.0, 0.3, 0.4];
    check_tensor_near(grad_input, &[2, 2], &expected_grad_input, 1e-9);
    Ok(())
}

#[test]
fn test_relu_forward_zeros() -> Result<(), NeuraRustError> {
    let input = Tensor::new(vec![0.0, 0.0, 0.0], vec![3])?;
    let output = relu_op(&input)?;
    check_tensor_near(&output, &[3], &[0.0, 0.0, 0.0], 1e-9);
    Ok(())
}

#[test]
fn test_relu_forward_positive() -> Result<(), NeuraRustError> {
    let input = Tensor::new(vec![1.0, 10.0, 0.1], vec![3])?;
    let output = relu_op(&input)?;
    check_tensor_near(&output, &[3], &[1.0, 10.0, 0.1], 1e-9);
    Ok(())
}

#[test]
fn test_relu_backward_mixed() -> Result<(), NeuraRustError> {
    let input = Tensor::new(vec![-2.0, 3.0, 0.0, -5.0, 6.0], vec![5])?;
    input.set_requires_grad(true)?;
    let output = relu_op(&input)?;
    let grad_output = tensor::full(&input.shape(), 1.0)?;
    
    let grad_fn = output.grad_fn().unwrap();
    let grad_inputs = grad_fn.backward(&grad_output)?;
    
    assert_eq!(grad_inputs.len(), 1);
    check_tensor_near(&grad_inputs[0], &[5], &[0.0, 1.0, 0.0, 0.0, 1.0], 1e-9);
    Ok(())
}

// --- Autograd Tests using check_grad (Converted to F32 and Ignored) ---

#[test]
#[ignore = "ReLU grad check unstable near 0 due to derivative discontinuity"]
fn test_relu_check_grad_basic() -> Result<(), GradCheckError> {
    let input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5])?;
    input.set_requires_grad(true)?;
    let func = |inputs: &[Tensor]| relu_op(&inputs[0]);

    let output_grad = tensor::full(&input.shape(), 1.0)?;
    
    check_grad(func, &[input], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
fn test_relu_check_grad_all_positive() -> Result<(), GradCheckError> {
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    input.set_requires_grad(true)?;
    let func = |inputs: &[Tensor]| relu_op(&inputs[0]);

    let output_grad = tensor::full(&input.shape(), 1.0)?;

    check_grad(func, &[input], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
#[ignore = "ReLU grad check unstable near 0 due to derivative discontinuity"]
fn test_relu_check_grad_all_negative_or_zero() -> Result<(), GradCheckError> {
    let input = Tensor::new(vec![-2.0, -1.0, 0.0], vec![3])?;
    input.set_requires_grad(true)?;
    let func = |inputs: &[Tensor]| relu_op(&inputs[0]);

    let output_grad = tensor::full(&input.shape(), 1.0)?;

    check_grad(func, &[input], &output_grad, 1e-3, 1e-4, 1e-3)
} 