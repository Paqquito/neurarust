use super::*;
use crate::autograd::grad_check::check_grad;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use crate::error::NeuraRustError;

// Helper to create tensors for tests (spécifique f64)
fn create_test_tensor(
    data: Vec<f64>,
    shape: Vec<usize>,
) -> Tensor<f64>
{
    Tensor::new(data, shape).expect("Failed to create test tensor")
}

// Helper to create tensors with requires_grad = true (spécifique f64)
fn create_test_tensor_with_grad(
    data: Vec<f64>,
    shape: Vec<usize>,
) -> Tensor<f64>
{
    let t = create_test_tensor(data, shape);
    t.set_requires_grad(true)
        .expect("Failed to set requires_grad");
    t
}

#[test]
fn test_div_tensors_ok() {
    let a = create_test_tensor(vec![10.0, 20.0], vec![2]);
    let b = create_test_tensor(vec![2.0, 5.0], vec![2]);
    let result = div_op(&a, &b).unwrap();
    let expected_data = vec![5.0, 4.0];
    assert_eq!(result.shape(), vec![2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
}

#[test]
fn test_div_by_zero() {
    let a = create_test_tensor(vec![10.0], vec![1]);
    let b = create_test_tensor(vec![0.0], vec![1]);
    let result = div_op(&a, &b);
    assert!(matches!(result, Err(NeuraRustError::DivisionByZero)));
}

#[test]
fn test_div_broadcasting() {
    let matrix = create_test_tensor(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let row_vector = create_test_tensor(vec![2.0, 5.0], vec![1, 2]);
    let result = div_op(&matrix, &row_vector).unwrap();
    let expected_data = vec![5.0, 4.0, 15.0, 8.0]; // [[10/2, 20/5], [30/2, 40/5]]
    assert_eq!(result.shape(), vec![2, 2]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
}

// --- Autograd Tests ---

#[test]
fn test_div_backward_simple() {
    let a = create_test_tensor_with_grad(vec![10.0, 20.0], vec![2]);
    let b = create_test_tensor_with_grad(vec![2.0, 5.0], vec![2]);

    let func = |inputs: &[Tensor<f64>]| div_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = 1 / b = [1/2, 1/5] = [0.5, 0.2]
    // grad_b = -a / b^2 = [-10/4, -20/25] = [-2.5, -0.8]
    // let a_grad = a.grad().unwrap();
    // let b_grad = b.grad().unwrap();
    // let expected_a_grad = vec![0.5, 0.2];
    // let expected_b_grad = vec![-2.5, -0.8];
    // assert_relative_eq!(a_grad.data().as_slice(), expected_a_grad.as_slice(), epsilon = tolerance);
    // assert_relative_eq!(b_grad.data().as_slice(), expected_b_grad.as_slice(), epsilon = tolerance);

     assert!(grad_check_result.is_ok(), "Simple division backward grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_div_backward_broadcast() {
    let matrix = create_test_tensor_with_grad(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let row_vector = create_test_tensor_with_grad(vec![2.0, 5.0], vec![1, 2]);

    let func = |inputs: &[Tensor<f64>]| div_op(&inputs[0], &inputs[1]);

    let output_shape = vec![2, 2];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[matrix.clone(), row_vector.clone()], &output_grad, epsilon, tolerance);

    // Optional detailed checks:
    // grad_a = grad_output / b (broadcasted) = [[1/2, 1/5], [1/2, 1/5]] = [[0.5, 0.2], [0.5, 0.2]]
    // grad_b_unreduced = grad_output * (-result / b) = grad_output * (-a / b^2)
    //                  = [[1, 1], [1, 1]] * (- [[10, 20], [30, 40]] / [[4, 25], [4, 25]])
    //                  = [[1, 1], [1, 1]] * [[-2.5, -0.8], [-7.5, -1.6]]
    //                  = [[-2.5, -0.8], [-7.5, -1.6]]
    // grad_b = reduce(grad_b_unreduced) to [1, 2] by summing dim 0
    //        = [-2.5 + -7.5, -0.8 + -1.6] = [-10.0, -2.4]
    // let a_grad = matrix.grad().unwrap();
    // let b_grad = row_vector.grad().unwrap();
    // let expected_a_grad = vec![0.5, 0.2, 0.5, 0.2];
    // let expected_b_grad = vec![-10.0, -2.4];
    // assert_eq!(a_grad.shape(), vec![2, 2]);
    // assert_eq!(b_grad.shape(), vec![1, 2]);
    // assert_relative_eq!(a_grad.data().as_slice(), expected_a_grad.as_slice(), epsilon = tolerance);
    // assert_relative_eq!(b_grad.data().as_slice(), expected_b_grad.as_slice(), epsilon = tolerance);

     assert!(grad_check_result.is_ok(), "Broadcast division backward grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_div_backward_with_zero_divisor() {
    // Setup where the backward pass might encounter division by zero if not careful
    let a = create_test_tensor_with_grad(vec![10.0], vec![1]);
    let b = create_test_tensor_with_grad(vec![0.0001], vec![1]); // Small divisor
    let output_grad = Tensor::ones(vec![1]).unwrap();

    // The forward pass should work
    let c = div_op(&a, &b).unwrap();

    // The backward pass should also work (calculates 1/b and -a/b^2)
    let backward_result = c.backward(Some(output_grad));
    assert!(backward_result.is_ok(), "Backward failed for small divisor: {:?}", backward_result.err());

    // Check gradients are large but finite
    let a_grad_val = a.grad().unwrap().get(&[0]).unwrap();
    let b_grad_val = b.grad().unwrap().get(&[0]).unwrap();
    assert!(a_grad_val.is_finite());
    assert!(b_grad_val.is_finite());
    assert_relative_eq!(a_grad_val, 1.0 / 0.0001, epsilon = 1e-7);
    assert_relative_eq!(b_grad_val, -10.0 / (0.0001 * 0.0001), epsilon = 1e-7);
} 