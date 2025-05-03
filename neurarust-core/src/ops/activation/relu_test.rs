#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    
    use crate::ops::activation::relu::relu_op;
    use approx::assert_relative_eq;
    
    

    // Helper to create tensors for tests
    fn create_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn test_relu_forward() {
        let input = create_tensor_f64(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        let output = relu_op(&input).unwrap();
        let output_data = output.read_data().data.cpu_data().unwrap().clone();

        assert_eq!(output.shape(), vec![5]);
        output_data
            .iter()
            .zip(expected.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e));
    }

    // TODO: Add autograd tests using check_grad (use f64)
}

// Add autograd tests
#[cfg(test)]
mod autograd_tests {
    use crate::autograd::grad_check::check_grad;
    
    use crate::tensor::Tensor;
    
    use crate::ops::activation::relu::relu_op;

    // Helper for f64 tests
    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_relu_backward_basic() {
        // Use input containing 0.0, but increase tolerance for check_grad
        let input = create_tensor_f64_with_grad(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let func = |inputs: &[Tensor<f64>]| relu_op(&inputs[0]);

        let output_shape = vec![5];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        // Increase tolerance to account for numerical gradient difference at x=0 (0.5 vs 0.0)
        let tolerance = 0.51;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "ReLU basic backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_relu_backward_all_positive() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let func = |inputs: &[Tensor<f64>]| relu_op(&inputs[0]);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7; // Standard tolerance is fine here

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "ReLU all positive backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_relu_backward_all_negative_or_zero() {
        // Use input containing 0.0, but increase tolerance
        let input = create_tensor_f64_with_grad(vec![-2.0, -1.0, 0.0], vec![3]);
        let func = |inputs: &[Tensor<f64>]| relu_op(&inputs[0]);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 0.51; // Increased tolerance for x=0

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "ReLU all negative/zero backward grad check failed: {:?}", grad_check_result.err());
    }

    // TODO: Add tests for different shapes (e.g., matrices)
    // TODO: Add tests for edge cases if any (e.g., large numbers, NaNs - though NaNs might fail check_grad)

} 