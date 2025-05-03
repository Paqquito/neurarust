use super::*;
use crate::autograd::grad_check::check_grad;
use crate::tensor::Tensor;
use approx::assert_relative_eq;

// Helper to create tensors with requires_grad = true (using f64 for numerical stability)
fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    let t = Tensor::new(data, shape).unwrap();
    t.set_requires_grad(true).unwrap();
    t
}

#[test]
fn test_pow_forward() {
    let base = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
    let exponent = Tensor::new(vec![3.0, 2.0], vec![2]).unwrap();
    let result = pow_op(&base, &exponent).unwrap();
    let expected_data = vec![8.0, 9.0]; // 2^3, 3^2
    assert_eq!(result.shape(), vec![2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
}

#[test]
fn test_pow_forward_broadcast() {
    let base = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let exponent = Tensor::new(vec![3.0, 0.5], vec![1, 2]).unwrap();
    let result = pow_op(&base, &exponent).unwrap();
    // Expected shape: [2, 2]
    // [[2^3, 2^0.5], [3^3, 3^0.5]] = [[8.0, 1.414..], [27.0, 1.732..]]
    let expected_data = vec![8.0, 2.0f64.sqrt(), 27.0, 3.0f64.sqrt()];
    assert_eq!(result.shape(), vec![2, 2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-9);
}

// --- Autograd Tests ---

#[test]
fn test_pow_backward_simple() {
    let base = create_tensor_f64_with_grad(vec![2.0, 3.0], vec![2]);
    let exponent = create_tensor_f64_with_grad(vec![3.0, 2.0], vec![2]);

    let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-4; // Increased epsilon for pow
    let tolerance = 1e-5; // Increased tolerance for pow

    let grad_check_result = check_grad(func, &[base, exponent], &output_grad, epsilon, tolerance);
    assert!(grad_check_result.is_ok(), "Simple pow backward grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_pow_backward_broadcast_exponent() {
    let base = create_tensor_f64_with_grad(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
    let exponent = create_tensor_f64_with_grad(vec![2.0], vec![1]); // Broadcast exponent

    let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-4;
    let tolerance = 1e-5;

    let grad_check_result = check_grad(func, &[base, exponent], &output_grad, epsilon, tolerance);
    assert!(grad_check_result.is_ok(), "Pow backward (broadcast exponent) grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_pow_backward_broadcast_base() {
    let base = create_tensor_f64_with_grad(vec![2.0], vec![1]); // Broadcast base
    let exponent = create_tensor_f64_with_grad(vec![3.0, 4.0, 1.0, 0.5], vec![2, 2]);

    let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-4;
    let tolerance = 1e-5;

    let grad_check_result = check_grad(func, &[base, exponent], &output_grad, epsilon, tolerance);
    assert!(grad_check_result.is_ok(), "Pow backward (broadcast base) grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_pow_backward_only_base_grad() {
    let base = create_tensor_f64_with_grad(vec![2.0, 3.0], vec![2]);
    let exponent_no_grad = Tensor::new(vec![3.0, 2.0], vec![2]).unwrap();
    // exponent_no_grad.set_requires_grad(false).unwrap(); // Default

    let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &exponent_no_grad);

    let output_shape = vec![2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-4;
    let tolerance = 1e-5;

    // Only check gradient for `base` (inputs[0])
    let grad_check_result = check_grad(func, &[base.clone()], &output_grad, epsilon, tolerance);
    assert!(grad_check_result.is_ok(), "Pow backward (only base grad) grad check failed: {:?}", grad_check_result.err());

    // // Verify exponent has no grad
    // let c = pow_op(&base, &exponent_no_grad).unwrap();
    // c.backward(Some(output_grad)).unwrap();
    // assert!(exponent_no_grad.grad().is_none());
    // assert!(base.grad().is_some());
}

#[test]
fn test_pow_backward_only_exponent_grad() {
    let base_no_grad = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
    let exponent = create_tensor_f64_with_grad(vec![3.0, 2.0], vec![2]);

    let func = |inputs: &[Tensor<f64>]| pow_op(&base_no_grad, &inputs[0]);

    let output_shape = vec![2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-4;
    let tolerance = 1e-5;

    // Only check gradient for `exponent` (inputs[0])
    let grad_check_result = check_grad(func, &[exponent.clone()], &output_grad, epsilon, tolerance);
    assert!(grad_check_result.is_ok(), "Pow backward (only exponent grad) grad check failed: {:?}", grad_check_result.err());

    // // Verify base has no grad
    // let c = pow_op(&base_no_grad, &exponent).unwrap();
    // c.backward(Some(output_grad)).unwrap();
    // assert!(base_no_grad.grad().is_none());
    // assert!(exponent.grad().is_some());
} 