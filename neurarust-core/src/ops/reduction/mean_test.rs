#[cfg(test)]
mod tests {
    use crate::Tensor;
    use approx::assert_relative_eq;
    use crate::autograd::grad_check::check_grad;
    use crate::ops::reduction::mean::mean_axes;

    fn create_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn test_mean_all() {
        let t = create_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = mean_axes(&t, &[], false).unwrap();
        assert_eq!(result.shape(), vec![]);
        assert_relative_eq!(result.get(&[]).unwrap(), 3.5);
    }

    #[test]
    fn test_mean_axis_0() {
        let t = create_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = mean_axes(&t, &[0], false).unwrap();
        assert_eq!(result.shape(), vec![3]);
        let expected_data = vec![2.5, 3.5, 4.5]; // (1+4)/2, (2+5)/2, (3+6)/2
        let res_data = result.read_data().data.cpu_data().unwrap().clone();
        res_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(r, e)| assert_relative_eq!(*r, *e));
    }

    // TODO: Add tests for keep_dims

    // --- Autograd Tests ---

    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_mean_all_backward() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor<f64>]| mean_axes(&inputs[0], &[], false);

        let output_shape = vec![];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Mean all backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_mean_axis_0_backward() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor<f64>]| mean_axes(&inputs[0], &[0], false);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean axis 0 backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_mean_axis_1_keep_dims_backward() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor<f64>]| mean_axes(&inputs[0], &[1], true);

        let output_shape = vec![2, 1];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean axis 1 keep_dims backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_mean_multiple_axes_backward() {
        let input = create_tensor_f64_with_grad((1..=24).map(|x| x as f64).collect(), vec![2, 3, 4]);
        let func = |inputs: &[Tensor<f64>]| mean_axes(&inputs[0], &[0, 2], false);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean multiple axes backward grad check failed: {:?}", grad_check_result.err());
    }
} 