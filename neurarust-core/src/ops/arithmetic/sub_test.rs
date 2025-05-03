use super::*;
use crate::autograd::grad_check::check_grad;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use num_traits::{One, Zero};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Neg, Sub};

// Helper to create tensors for tests
fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: Sub<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + 'static
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + std::iter::Product,
{
    Tensor::new(data, shape).expect("Failed to create test tensor")
}

// Helper to create tensors with requires_grad = true
fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: Sub<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + 'static
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + std::iter::Product,
{
    let t = create_test_tensor(data, shape);
    t.set_requires_grad(true)
        .expect("Failed to set requires_grad");
    t
}

#[test]
fn test_sub_tensors_ok() {
    let a = create_test_tensor(vec![10.0, 20.0], vec![2]);
    let b = create_test_tensor(vec![3.0, 4.0], vec![2]);
    let result = sub_op(&a, &b).unwrap();
    let expected_data = vec![7.0, 16.0];
    assert_eq!(result.shape(), vec![2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
}

#[test]
fn test_sub_tensors_shape_mismatch() {
    let a = create_test_tensor(vec![1.0, 2.0], vec![2]);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = sub_op(&a, &b);
    assert!(matches!(
        result,
        Err(NeuraRustError::BroadcastError { .. })
    ));
}

#[test]
fn test_sub_broadcasting() {
    let matrix = create_test_tensor(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let row_vector = create_test_tensor(vec![1.0, 2.0], vec![1, 2]);
    let result = sub_op(&matrix, &row_vector).unwrap();
    let expected_data = vec![9.0, 18.0, 29.0, 38.0]; // [[10-1, 20-2], [30-1, 40-2]]
    assert_eq!(result.shape(), vec![2, 2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());

    let col_vector = create_test_tensor(vec![5.0, 10.0], vec![2, 1]);
    let result2 = sub_op(&matrix, &col_vector).unwrap();
    let expected_data2 = vec![5.0, 15.0, 20.0, 30.0]; // [[10-5, 20-5], [30-10, 40-10]]
    assert_eq!(result2.shape(), vec![2, 2]);
    let res_data2 = result2.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data2.as_slice(), expected_data2.as_slice());

    let scalar = create_test_tensor(vec![100.0], vec![]);
    let result3 = sub_op(&matrix, &scalar).unwrap();
    let expected_data3 = vec![-90.0, -80.0, -70.0, -60.0];
    assert_eq!(result3.shape(), vec![2, 2]);
    let res_data3 = result3.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data3.as_slice(), expected_data3.as_slice());

    let result4 = sub_op(&scalar, &matrix).unwrap();
    let expected_data4 = vec![90.0, 80.0, 70.0, 60.0];
    assert_eq!(result4.shape(), vec![2, 2]);
    let res_data4 = result4.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data4.as_slice(), expected_data4.as_slice());
}

// --- Autograd Tests ---

#[test]
fn test_sub_backward_simple() {
    let a = create_test_tensor_with_grad(vec![10.0, 20.0], vec![2]);
    let b = create_test_tensor_with_grad(vec![3.0, 4.0], vec![2]);

    let func = |inputs: &[Tensor<f64>]| sub_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = grad_output = [1, 1]
    // grad_b = -grad_output = [-1, -1]
    // let a_grad = a.grad().unwrap();
    // let b_grad = b.grad().unwrap();
    // let expected_a_grad = vec![1.0, 1.0];
    // let expected_b_grad = vec![-1.0, -1.0];
    // assert_relative_eq!(a_grad.data().as_slice(), expected_a_grad.as_slice(), epsilon = tolerance);
    // assert_relative_eq!(b_grad.data().as_slice(), expected_b_grad.as_slice(), epsilon = tolerance);

    assert!(grad_check_result.is_ok(), "Simple subtraction backward grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_sub_backward_broadcast() {
    let matrix = create_test_tensor_with_grad(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let row_vector = create_test_tensor_with_grad(vec![1.0, 2.0], vec![1, 2]);

    let func = |inputs: &[Tensor<f64>]| sub_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[matrix.clone(), row_vector.clone()], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = grad_output = [[1, 1], [1, 1]]
    // grad_b_unreduced = -grad_output = [[-1, -1], [-1, -1]]
    // grad_b = reduce(grad_b_unreduced) to [1, 2] by summing dim 0 = [-1 + -1, -1 + -1] = [-2, -2]
    // let a_grad = matrix.grad().unwrap();
    // let b_grad = row_vector.grad().unwrap();
    // let expected_a_grad = vec![1.0, 1.0, 1.0, 1.0];
    // let expected_b_grad = vec![-2.0, -2.0];
    // assert_eq!(a_grad.shape(), vec![2, 2]);
    // assert_eq!(b_grad.shape(), vec![1, 2]);
    // assert_relative_eq!(a_grad.data().as_slice(), expected_a_grad.as_slice(), epsilon = tolerance);
    // assert_relative_eq!(b_grad.data().as_slice(), expected_b_grad.as_slice(), epsilon = tolerance);

    assert!(grad_check_result.is_ok(), "Broadcast subtraction backward grad check failed: {:?}", grad_check_result.err());
} 