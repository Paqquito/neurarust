use super::*;
use crate::autograd::grad_check::check_grad;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use num_traits::{One, Zero};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Neg};

fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    Tensor::new(data, shape).expect("Failed to create test tensor")
}

fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    let t = create_test_tensor(data, shape);
    t.set_requires_grad(true)
        .expect("Failed to set requires_grad");
    t
}

#[test]
fn test_neg_ok() {
    let a = create_test_tensor(vec![1.0, -2.0, 3.0], vec![3]);
    let result = neg_op(&a).unwrap();
    let expected_data = vec![-1.0, 2.0, -3.0];
    assert_eq!(result.shape(), vec![3]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());

    // Test using the trait implementation
    let result_trait = (-&a).unwrap();
    assert_eq!(result_trait.shape(), vec![3]);
    let res_data_trait = result_trait.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data_trait.as_slice(), expected_data.as_slice());
}

// --- Autograd Tests ---

#[test]
fn test_neg_backward() {
    let a = create_test_tensor_with_grad(vec![1.0, -2.0, 3.0], vec![3]);

    let func = |inputs: &[Tensor<f64>]| neg_op(&inputs[0]);

    let output_shape = vec![3];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[a], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = -grad_output = -[1, 1, 1] = [-1, -1, -1]
    // let a_grad = a.grad().unwrap();
    // let expected_grad = vec![-1.0, -1.0, -1.0];
    // assert_relative_eq!(a_grad.data().as_slice(), expected_grad.as_slice(), epsilon = tolerance);

    assert!(grad_check_result.is_ok(), "Negation backward grad check failed: {:?}", grad_check_result.err());
} 