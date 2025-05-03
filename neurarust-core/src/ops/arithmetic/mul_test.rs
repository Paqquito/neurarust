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

#[test]
fn test_mul_tensors_ok() {
    let a = create_test_tensor(vec![1.0, 2.0], vec![2]);
    let b = create_test_tensor(vec![3.0, 4.0], vec![2]);
    let result = mul_op(&a, &b).unwrap();
    let expected_data = vec![3.0, 8.0];
    assert_eq!(result.shape(), vec![2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
}

#[test]
fn test_mul_tensors_shape_mismatch() {
    let a = create_test_tensor(vec![1.0, 2.0], vec![2]);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = mul_op(&a, &b);
    assert!(matches!(
        result,
        Err(NeuraRustError::BroadcastError { .. })
    ));
}

#[test]
fn test_mul_broadcasting() {
    let matrix = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let row_vector = create_test_tensor(vec![10.0, 20.0], vec![1, 2]);
    let result = mul_op(&matrix, &row_vector).unwrap();
    let expected_data = vec![10.0, 40.0, 30.0, 80.0]; // [[1*10, 2*20], [3*10, 4*20]]
    assert_eq!(result.shape(), vec![2, 2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
}

// --- Autograd Tests ---

#[test]
fn test_mul_backward_simple() {
    let a = create_test_tensor_with_grad(vec![1.0, 2.0], vec![2]);
    let b = create_test_tensor_with_grad(vec![3.0, 4.0], vec![2]);

    let func = |inputs: &[Tensor<f64>]| mul_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = grad_output * b = [1, 1] * [3, 4] = [3, 4]
    // grad_b = grad_output * a = [1, 1] * [1, 2] = [1, 2]
    // let a_grad = a.grad().unwrap();
    // let b_grad = b.grad().unwrap();
    // let expected_a_grad = vec![3.0, 4.0];
    // let expected_b_grad = vec![1.0, 2.0];
    // assert_relative_eq!(a_grad.data().as_slice(), expected_a_grad.as_slice(), epsilon = tolerance);
    // assert_relative_eq!(b_grad.data().as_slice(), expected_b_grad.as_slice(), epsilon = tolerance);

     assert!(grad_check_result.is_ok(), "Simple multiplication backward grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_mul_backward_broadcast() {
    let matrix = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let row_vector = create_test_tensor_with_grad(vec![10.0, 20.0], vec![1, 2]);

    let func = |inputs: &[Tensor<f64>]| mul_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[matrix.clone(), row_vector.clone()], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = grad_output * b (broadcasted) = [[1, 1], [1, 1]] * [[10, 20], [10, 20]] = [[10, 20], [10, 20]]
    // grad_b_unreduced = grad_output * a = [[1, 1], [1, 1]] * [[1, 2], [3, 4]] = [[1, 2], [3, 4]]
    // grad_b = reduce(grad_b_unreduced) to [1, 2] by summing dim 0 = [1+3, 2+4] = [4, 6]
    // let a_grad = matrix.grad().unwrap();
    // let b_grad = row_vector.grad().unwrap();
    // let expected_a_grad = vec![10.0, 20.0, 10.0, 20.0];
    // let expected_b_grad = vec![4.0, 6.0];
    // assert_eq!(a_grad.shape(), vec![2, 2]);
    // assert_eq!(b_grad.shape(), vec![1, 2]);
    // assert_relative_eq!(a_grad.data().as_slice(), expected_a_grad.as_slice(), epsilon = tolerance);
    // assert_relative_eq!(b_grad.data().as_slice(), expected_b_grad.as_slice(), epsilon = tolerance);

    assert!(grad_check_result.is_ok(), "Broadcast multiplication backward grad check failed: {:?}", grad_check_result.err());
} 