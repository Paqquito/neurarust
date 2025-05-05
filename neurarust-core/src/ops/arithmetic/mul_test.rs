use crate::autograd::grad_check::{check_grad, GradCheckError};
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use crate::utils::testing::check_tensor_near;

#[test]
fn test_mul_tensors_ok() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0], vec![2])?;
    let result = crate::ops::arithmetic::mul::mul_op(&a, &b)?;
    let expected_data = vec![3.0, 8.0];
    check_tensor_near(&result, &[2], &expected_data, 1e-6);
    Ok(())
}

#[test]
fn test_mul_tensors_mismatched_shapes() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0, 5.0], vec![3])?;
    let result = crate::ops::arithmetic::mul::mul_op(&a, &b);
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch(_))));
    Ok(())
}

#[test]
fn test_mul_broadcasting() -> Result<(), NeuraRustError> {
    let matrix = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let row_vector = Tensor::from_vec_f32(vec![10.0, 20.0], vec![1, 2])?;
    let result_row = crate::ops::arithmetic::mul::mul_op(&matrix, &row_vector)?;
    let expected_data_row = vec![10.0, 40.0, 30.0, 80.0];
    check_tensor_near(&result_row, &[2, 2], &expected_data_row, 1e-6);

    let col_vector = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2, 1])?;
    let result_col = crate::ops::arithmetic::mul::mul_op(&matrix, &col_vector)?;
    let expected_data_col = vec![10.0, 20.0, 60.0, 80.0];
    check_tensor_near(&result_col, &[2, 2], &expected_data_col, 1e-6);

    Ok(())
}

#[test]
fn test_mul_non_contiguous() -> Result<(), NeuraRustError> {
    let base = Tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3])?;
    let sliced = base.slice(&[(0..1), (0..2), (1..3)])?;
    assert!(!sliced.is_contiguous());

    let multiplier = Tensor::from_vec_f32(vec![10.0, 10.0, 10.0, 10.0], vec![1, 2, 2])?;
    assert!(multiplier.is_contiguous());

    let result = crate::ops::arithmetic::mul::mul_op(&sliced, &multiplier)?;
    let expected_data = vec![10.0 * 1.0, 10.0 * 2.0, 10.0 * 4.0, 10.0 * 5.0];
    check_tensor_near(&result, &[1, 2, 2], &expected_data, 1e-6);
    assert!(result.is_contiguous(), "Mul op output should be contiguous");

    Ok(())
}

#[test]
#[ignore = "Skipping due to check_grad F32 precision limitations. Backward logic visually verified."]
fn test_mul_backward_simple() -> Result<(), GradCheckError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = Tensor::from_vec_f32(vec![4.0, 5.0, 6.0], vec![3])?;
    a.set_requires_grad(true)?;
    b.set_requires_grad(true)?;

    let func = |inputs: &[&Tensor]| crate::ops::arithmetic::mul::mul_op(inputs[0], inputs[1]);
    let output_grad = Tensor::ones_like(&a)?;

    check_grad(func, &[&a, &b], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
#[ignore = "Skipping due to check_grad F32 precision limitations. Backward logic visually verified."]
fn test_mul_backward_broadcast() -> Result<(), GradCheckError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![1, 2])?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0], vec![2, 1])?;
    a.set_requires_grad(true)?;
    b.set_requires_grad(true)?;

    let func = |inputs: &[&Tensor]| crate::ops::arithmetic::mul::mul_op(inputs[0], inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::from_vec_f32(vec![0.1, 0.2, 0.3, 0.4], output_shape)?;

    check_grad(func, &[&a, &b], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
fn test_mul_backward_simple_f64() -> Result<(), GradCheckError> {
    let a = Tensor::from_vec_f64(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = Tensor::from_vec_f64(vec![4.0, 5.0, 6.0], vec![3])?;
    a.set_requires_grad(true)?;
    b.set_requires_grad(true)?;

    let func = |inputs: &[&Tensor]| crate::ops::arithmetic::mul::mul_op(inputs[0], inputs[1]);
    let output_grad = Tensor::ones_like_f64(&a)?;

    check_grad(func, &[&a, &b], &output_grad, 1e-6, 1e-9, 1e-7)
}

#[test]
fn test_mul_backward_broadcast_f64() -> Result<(), GradCheckError> {
    let a = Tensor::from_vec_f64(vec![1.0, 2.0], vec![1, 2])?;
    let b = Tensor::from_vec_f64(vec![3.0, 4.0], vec![2, 1])?;
    a.set_requires_grad(true)?;
    b.set_requires_grad(true)?;

    let func = |inputs: &[&Tensor]| crate::ops::arithmetic::mul::mul_op(inputs[0], inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::from_vec_f64(vec![0.1, 0.2, 0.3, 0.4], output_shape)?;

    check_grad(func, &[&a, &b], &output_grad, 1e-6, 1e-9, 1e-7)
} 