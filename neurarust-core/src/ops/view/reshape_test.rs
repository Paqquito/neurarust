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
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
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
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
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
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for reshape adding dim: {:?}", result.err());
    }
} 