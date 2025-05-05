#[cfg(test)]
mod tests {
    
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::utils::testing::create_test_tensor_with_grad;

    #[test]
    fn test_reshape_backward() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].reshape(vec![3, 2])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=6).map(|x| x as f64).collect(),
            vec![2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![3, 2]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7, 1e-5);
        assert!(result.is_ok(), "Gradient check failed for reshape: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward_flatten() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            let numel = inputs[0].numel();
            inputs[0].reshape(vec![numel])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=12).map(|x| x as f64).collect(),
            vec![2, 2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![12]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7, 1e-5);
        assert!(result.is_ok(), "Gradient check failed for flatten reshape: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward_add_dim() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].reshape(vec![2, 2, 1, 3])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=12).map(|x| x as f64).collect(),
            vec![2, 2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![2, 2, 1, 3]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7, 1e-5);
        assert!(result.is_ok(), "Gradient check failed for reshape adding dim: {:?}", result.err());
    }

    #[test]
    fn test_reshape_contiguous() {
        let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let reshaped = t.reshape(&[3, 2]).unwrap();
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn test_reshape_on_views() {
        let t_orig = crate::tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]).unwrap();
        let t_transposed = t_orig.transpose(0, 1).unwrap(); // Shape [2, 2, 3]
        assert!(!t_transposed.is_contiguous());
    }

    #[test]
    fn test_reshape_non_contiguous_error() {
        let t_orig = crate::tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]).unwrap();
        let t_transposed = t_orig.transpose(0, 1).unwrap(); // Non-contiguous
        let result = t_transposed.reshape(&[4, 3]);
    }

    #[test]
    fn test_reshape_to_scalar() {
        let t = crate::tensor::from_vec_f32(vec![5.0], vec![1]).unwrap();
        let reshaped = t.reshape(&[]).unwrap();
        assert_eq!(reshaped.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_reshape_from_scalar() {
        let t = crate::tensor::from_vec_f32(vec![5.0], vec![]).unwrap(); // Scalar
        let reshaped = t.reshape(&[1, 1, 1]).unwrap();
        assert_eq!(reshaped.shape(), &[1, 1, 1]);
    }

    #[test]
    fn test_reshape_backward() {
        let t_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t_shape = vec![2, 3];
        let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
        t.set_requires_grad(true).unwrap();
    }

    #[test]
    fn test_reshape_backward_flatten() {
        let t_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t_shape = vec![2, 3];
        let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
        t.set_requires_grad(true).unwrap();
    }

    #[test]
    fn test_reshape_backward_add_dim() {
        let t_data = vec![1.0, 2.0, 3.0, 4.0];
        let t_shape = vec![4];
        let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
        t.set_requires_grad(true).unwrap();
    }
} 