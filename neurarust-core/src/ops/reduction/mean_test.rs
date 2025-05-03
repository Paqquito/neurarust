#[cfg(test)]
mod tests {
    use crate::Tensor;
    use approx::assert_relative_eq;
    use crate::autograd::grad_check::check_grad;
    use crate::ops::reduction::mean::mean_op;
    
    use crate::utils::testing::{check_tensor_near, create_test_tensor, create_test_tensor_with_grad};

    #[test]
    fn test_mean_all() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = mean_op(&t, None, false).unwrap();
        assert_eq!(result.shape(), vec![], "Result shape should be scalar");
        let result_data = result.get_f32_data().expect("Failed to get result data");
        assert_eq!(result_data.len(), 1, "Result should have 1 element");
        assert_relative_eq!(result_data[0], 3.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_axis_0() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = mean_op(&t, Some(&[0]), false).unwrap();
        let expected_data = vec![2.5, 3.5, 4.5]; // (1+4)/2, (2+5)/2, (3+6)/2
        check_tensor_near(&result, &[3], &expected_data, 1e-6);
    }

    // TODO: Add tests for keep_dims

    // --- Autograd Tests ---

    #[test]
    #[ignore = "Skipping mean backward tests until mean backward is implemented/adapted"]
    fn test_mean_all_backward() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor]| mean_op(&inputs[0], None, false);

        let output_shape = mean_op(&input, None, false).unwrap().shape();
        let output_grad = Tensor::from_vec_f32(vec![1.0; 1], output_shape)
            .expect("Failed to create output grad");
        let epsilon = 1e-5;
        let tolerance = 1e-4;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Mean all backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    #[ignore = "Skipping mean backward tests until mean backward is implemented/adapted"]
    fn test_mean_axis_0_backward() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor]| mean_op(&inputs[0], Some(&[0]), false);

        let output_shape = mean_op(&input, Some(&[0]), false).unwrap().shape();
        let numel_out = output_shape.iter().product();
        let output_grad = Tensor::from_vec_f32(vec![1.0; numel_out], output_shape)
            .expect("Failed to create output grad");
        let epsilon = 1e-5;
        let tolerance = 1e-4;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean axis 0 backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    #[ignore = "Skipping mean backward tests until mean backward is implemented/adapted"]
    fn test_mean_axis_1_keep_dims_backward() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor]| mean_op(&inputs[0], Some(&[1]), true);

        let output_shape = mean_op(&input, Some(&[1]), true).unwrap().shape();
        let numel_out = output_shape.iter().product();
        let output_grad = Tensor::from_vec_f32(vec![1.0; numel_out], output_shape)
            .expect("Failed to create output grad");
        let epsilon = 1e-5;
        let tolerance = 1e-4;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean axis 1 keep_dims backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    #[ignore = "Skipping mean backward tests until mean backward is implemented/adapted"]
    fn test_mean_multiple_axes_backward() {
        let input = create_test_tensor_with_grad((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4]);
        let func = |inputs: &[Tensor]| mean_op(&inputs[0], Some(&[0, 2]), false);

        let output_shape = mean_op(&input, Some(&[0, 2]), false).unwrap().shape();
        let numel_out = output_shape.iter().product();
        let output_grad = Tensor::from_vec_f32(vec![1.0; numel_out], output_shape)
            .expect("Failed to create output grad");
        let epsilon = 1e-5;
        let tolerance = 1e-4;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean multiple axes backward grad check failed: {:?}", grad_check_result.err());
    }
} 