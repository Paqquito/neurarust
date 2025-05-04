use super::*;
use crate::autograd::grad_check::check_grad;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use num_traits::{One, Zero};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul};
use crate::utils::testing::check_tensor_near;
use crate::autograd::grad_check::{GradCheckError};

// Helper to create tensors for tests
fn create_test_tensor<T>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> Tensor<T>
where
    T: Mul<Output = T>
        + Add<Output = T>
        + AddAssign
        + Sum
        + PartialOrd
        + std::iter::Product
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Default
        + Zero
        + One
        + PartialEq,
{
    Tensor::new(data, shape).expect("Failed to create test tensor")
}

// Helper to create tensors with requires_grad = true
fn create_test_tensor_with_grad<T>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> Tensor<T>
where
    T: Mul<Output = T>
        + Add<Output = T>
        + AddAssign
        + Sum
        + PartialOrd
        + std::iter::Product
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Default
        + Zero
        + One
        + PartialEq,
{
    let t = create_test_tensor(data, shape);
    t.set_requires_grad(true)
        .expect("Failed to set requires_grad");
    t
}

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
fn test_mul_tensors_ok() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0], vec![2])?;
    let result = mul_op(&a, &b)?;
    let expected_data = vec![3.0, 8.0];
    assert_eq!(result.shape(), &[2]);
    let res_data = get_f32_data(&result)?;
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_mul_tensors_shape_mismatch() -> Result<(), NeuraRustError> {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::from_vec_f32(vec![3.0, 4.0, 5.0], vec![3])?;
    let result = mul_op(&a, &b);
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch(_))));
    Ok(())
}

#[test]
fn test_mul_broadcasting() -> Result<(), NeuraRustError> {
    let matrix = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let row_vector = Tensor::from_vec_f32(vec![10.0, 20.0], vec![1, 2])?;
    let result = mul_op(&matrix, &row_vector)?;
    let expected_data = vec![10.0, 40.0, 30.0, 80.0]; // [[1*10, 2*20], [3*10, 4*20]]
    assert_eq!(result.shape(), &[2, 2]);
    let res_data = get_f32_data(&result)?;
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    Ok(())
}

// --- Autograd Tests ---

#[test]
fn test_mul_backward_simple() -> Result<(), GradCheckError> {
    let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2])?;
    a.set_requires_grad(true)?;
    let b = Tensor::from_vec_f32(vec![2.0, 5.0], vec![2])?;
    b.set_requires_grad(true)?;

    let func = |inputs: &[Tensor]| mul_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2];
    let output_grad = Tensor::from_vec_f32(vec![1.0, 1.0], output_shape)?;
    
    let epsilon = 1e-4;
    let tolerance = 1e-4;

    check_grad(func, &[a, b], &output_grad, epsilon, tolerance)
}

#[test]
fn test_mul_backward_broadcast() -> Result<(), GradCheckError> {
    let matrix = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    matrix.set_requires_grad(true)?;
    let row_vector = Tensor::from_vec_f32(vec![10.0, 20.0], vec![1, 2])?;
    row_vector.set_requires_grad(true)?;

    let func = |inputs: &[Tensor]| mul_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::from_vec_f32(vec![0.1, 0.2, 0.3, 0.4], output_shape)?;

    let epsilon = 1e-4;
    let tolerance = 1e-4;

    check_grad(func, &[matrix, row_vector], &output_grad, epsilon, tolerance)
} 