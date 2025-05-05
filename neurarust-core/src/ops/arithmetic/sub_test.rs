#[cfg(test)] // Add cfg(test) for the whole file
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::utils::testing::check_tensor_near;

#[test]
fn test_sub_tensors_ok() -> Result<(), NeuraRustError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
    let result = crate::ops::arithmetic::sub::sub_op(&a, &b)?;
    let expected_data = vec![-3.0, -3.0, -3.0];
    check_tensor_near(&result, &[3], &expected_data, 1e-6);
    Ok(())
}

#[test]
fn test_sub_tensors_shape_mismatch() -> Result<(), NeuraRustError> {
    let a = Tensor::new(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
    let result = crate::ops::arithmetic::sub::sub_op(&a, &b);
    assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
    Ok(())
}

#[test]
fn test_sub_broadcasting() -> Result<(), NeuraRustError> {
    let matrix = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
    let scalar = Tensor::new(vec![5.0], vec![1])?;
    let result1 = crate::ops::arithmetic::sub::sub_op(&matrix, &scalar)?;
    let expected_data1 = vec![5.0, 15.0, 25.0, 35.0];
    check_tensor_near(&result1, &[2, 2], &expected_data1, 1e-6);

    let t3 = Tensor::new(vec![10.0, 20.0], vec![2, 1])?;
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    let result2 = crate::ops::arithmetic::sub::sub_op(&t3, &t1)?;
    let expected_data2 = vec![9.0, 8.0, 7.0, 19.0, 18.0, 17.0];
    check_tensor_near(&result2, &[2, 3], &expected_data2, 1e-6);

    Ok(())
}

#[test]
fn test_sub_backward_simple() -> Result<(), NeuraRustError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
    a.set_requires_grad(true)?;
    b.set_requires_grad(true)?;

    let result = crate::ops::arithmetic::sub::sub_op(&a, &b)?;
    let output_grad = Tensor::new(vec![0.1, 0.2, 0.3], vec![3])?;
    result.backward(Some(output_grad))?;

    check_tensor_near(&a.grad().unwrap(), &[3], &vec![0.1, 0.2, 0.3], 1e-6);
    check_tensor_near(&b.grad().unwrap(), &[3], &vec![-0.1, -0.2, -0.3], 1e-6);
    Ok(())
}

#[test]
fn test_sub_backward_broadcast() -> Result<(), NeuraRustError> {
    let a = Tensor::new(vec![10.0, 20.0], vec![2, 1])?;
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    a.set_requires_grad(true)?;
    b.set_requires_grad(true)?;

    let result = crate::ops::arithmetic::sub::sub_op(&a, &b)?;
    let output_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3])?;
    result.backward(Some(output_grad))?;

    check_tensor_near(&a.grad().unwrap(), &[2, 1], &vec![0.6, 1.5], 1e-6);
    check_tensor_near(&b.grad().unwrap(), &[3], &vec![-0.5, -0.7, -0.9], 1e-6);
    Ok(())
} 