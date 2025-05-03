#[cfg(test)]
mod tests {
    use crate::autograd::grad_check::check_grad;
    use crate::Tensor;
    use approx::assert_relative_eq;
    use crate::ops::linalg::matmul::matmul_op;
    use crate::error::NeuraRustError;

    // Helper for f64 tests
    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_matmul_forward() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let output = matmul_op(&a, &b).unwrap();
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected_data = vec![19.0, 22.0, 43.0, 50.0];
        let output_data = output.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(output.shape(), vec![2, 2]);
        output_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e, epsilon = 1e-7));
    }

     #[test]
    fn test_matmul_forward_non_square() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::<f64>::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let output = matmul_op(&a, &b).unwrap();
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        let expected_data = vec![58.0, 64.0, 139.0, 154.0];
        let output_data = output.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(output.shape(), vec![2, 2]);
        output_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e, epsilon = 1e-7));
    }

     #[test]
    fn test_matmul_shape_mismatch_inner() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0], vec![3, 1]).unwrap(); // Inner dim mismatch (2 vs 3)
        let result = matmul_op(&a, &b);
        assert!(result.is_err());
        if let Err(NeuraRustError::ShapeMismatch { .. }) = result {
            // Correct error type
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }

     #[test]
    fn test_matmul_shape_mismatch_rank() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]).unwrap(); // Rank mismatch (2 vs 3)
        let result = matmul_op(&a, &b);
         assert!(result.is_err());
        if let Err(NeuraRustError::ShapeMismatch { operation, .. }) = result {
             assert!(operation.contains("inputs must be 2D"));
        } else {
            panic!("Expected ShapeMismatch error due to rank");
        }
    }

     #[test]
    fn test_matmul_backward_simple() {
        let a = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = create_tensor_f64_with_grad(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Matmul simple backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_matmul_backward_non_square() {
        let a = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = create_tensor_f64_with_grad(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Matmul non-square backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_matmul_backward_only_a_grad() {
        let a = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap(); // b requires_grad=false
        b.set_requires_grad(false).unwrap();

        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        // We only expect a gradient for 'a' (index 0)
        let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

        // Verify only a gradient for 'a' was computed and it's correct
        assert!(grad_check_result.is_ok(), "Matmul only A grad check failed: {:?}", grad_check_result.err());
        // Further checks on the gradient value could be added here if needed
    }

     #[test]
    fn test_matmul_backward_only_b_grad() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(); // a requires_grad=false
        a.set_requires_grad(false).unwrap();
        let b = create_tensor_f64_with_grad(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        // We only expect a gradient for 'b' (index 1)
        let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

        // Verify only a gradient for 'b' was computed and it's correct
        assert!(grad_check_result.is_ok(), "Matmul only B grad check failed: {:?}", grad_check_result.err());
        // Further checks on the gradient value could be added here if needed
    }
} 