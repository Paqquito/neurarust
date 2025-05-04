use super::*;
// use crate::autograd::grad_check::check_grad;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use num_traits::{One, Zero};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Neg, Sub};
use crate::utils::testing::check_tensor_near; // Utiliser pour comparer f32

// Helper (non-generic) pour obtenir les données f32
fn get_f32_data(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
    let guard = tensor.read_data();
    if guard.dtype != crate::types::DType::F32 || guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation("Test helper requires F32 CPU tensor".to_string()));
    }
    match &*guard.buffer {
        crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F32(data_arc)) => Ok(data_arc.to_vec()),
        _ => Err(NeuraRustError::UnsupportedOperation("Buffer type not CpuF32".to_string())),
    }
}

#[test]
fn test_sub_tensors_ok() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2])?; // f32
    let b = Tensor::from_vec_f32(vec![3.0, 4.0], vec![2])?;  // f32
    let result = sub_op(&a, &b)?;
    let expected_data = vec![7.0, 16.0];
    assert_eq!(result.shape(), &[2]);
    let res_data = get_f32_data(&result)?;
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_sub_tensors_shape_mismatch() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0, 5.0], vec![3])?;
    let result = sub_op(&a, &b);
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch(_))));
    Ok(())
}

#[test]
fn test_sub_broadcasting() -> Result<(), NeuraRustError> {
    let matrix = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?; // f32
    let scalar = Tensor::from_vec_f32(vec![5.0], vec![1])?;                     // f32
    let result = sub_op(&matrix, &scalar)?;
    let expected_data = vec![5.0, 15.0, 25.0, 35.0];
    assert_eq!(result.shape(), &[2, 2]);
    let res_data = get_f32_data(&result)?;
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    Ok(())
}

// --- Autograd Tests (Manual Checks) ---

#[test]
fn test_sub_backward_simple() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2])?;
    a.set_requires_grad(true)?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0], vec![2])?;
    b.set_requires_grad(true)?;

    let result = sub_op(&a, &b)?;
    let output_grad = Tensor::from_vec_f32(vec![0.1, 0.2], vec![2])?;
    result.backward(Some(output_grad))?;

    // grad_a = output_grad = [0.1, 0.2]
    // grad_b = -output_grad = [-0.1, -0.2]
    check_tensor_near(&a.grad().unwrap(), &[2], &vec![0.1, 0.2], 1e-6);
    check_tensor_near(&b.grad().unwrap(), &[2], &vec![-0.1, -0.2], 1e-6);
    Ok(())
}

#[test]
fn test_sub_backward_broadcast() -> Result<(), NeuraRustError> {
    let matrix = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
    matrix.set_requires_grad(true)?;
    let scalar = Tensor::from_vec_f32(vec![5.0], vec![1])?;
    scalar.set_requires_grad(true)?;

    let result = sub_op(&matrix, &scalar)?;
    let output_grad = Tensor::from_vec_f32(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2])?;
    result.backward(Some(output_grad))?;

    // grad_matrix = output_grad = [0.1, 0.2, 0.3, 0.4]
    // grad_scalar = sum(-output_grad) = -(0.1 + 0.2 + 0.3 + 0.4) = -1.0
    check_tensor_near(&matrix.grad().unwrap(), &[2, 2], &vec![0.1, 0.2, 0.3, 0.4], 1e-6);
    check_tensor_near(&scalar.grad().unwrap(), &[1], &vec![-1.0], 1e-6);
    Ok(())
} 