#[cfg(test)] // Add cfg(test) for the whole file
use crate::autograd::grad_check::{check_grad, GradCheckError};
use crate::error::NeuraRustError;
use crate::tensor::{self, Tensor}; // Import tensor for creation funcs
use crate::utils::testing::check_tensor_near;

#[test]
fn test_neg_ok() -> Result<(), NeuraRustError> {
    let t1 = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2])?;
    let r = crate::ops::arithmetic::neg::neg_op(&t1)?;
    let expected_data = vec![-1.0, 2.0, -3.0, 4.0];
    check_tensor_near(&r, &t1.shape(), &expected_data, 1e-6);
    Ok(())
}

// --- Autograd Tests ---

#[test]
fn test_neg_backward() -> Result<(), GradCheckError> {
    let input_data = vec![1.0f32, -2.0, 3.0, -4.0, 0.0];
    let input_shape = vec![5];
    let input = Tensor::new(input_data, input_shape)?;
    input.set_requires_grad(true)?;

    let neg_fn_for_check = |inputs: &[Tensor]| crate::ops::arithmetic::neg::neg_op(&inputs[0]);

    let output_grad = tensor::full(&input.shape(), 1.0)?;

    check_grad(
        neg_fn_for_check,
        &[input],
        &output_grad,
        1e-5,
        2e-3,
        2e-3,
    )
}

#[test]
fn test_neg_backward_f64() -> Result<(), GradCheckError> {
    let input_data = vec![1.0f64, -2.0, 3.0, -4.0, 0.0];
    let input_shape = vec![5];
    let input = Tensor::new_f64(input_data, input_shape)?;
    input.set_requires_grad(true)?;

    let neg_fn_for_check = |inputs: &[Tensor]| crate::ops::arithmetic::neg::neg_op(&inputs[0]);

    let output_grad = tensor::full_f64(&input.shape(), 1.0)?;

    check_grad(
        neg_fn_for_check,
        &[input],
        &output_grad,
        1e-6,
        1e-9,
        1e-7,
    )
} 