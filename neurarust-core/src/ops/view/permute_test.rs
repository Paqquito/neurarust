#[cfg(test)]
mod tests {
    use crate::ops::view::permute::permute_op;
    use crate::tensor::{self, Tensor};
    use crate::error::NeuraRustError;
    use crate::autograd::grad_check::{check_grad, GradCheckError};
    

    #[test]
    fn test_permute_basic() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let permuted = permute_op(&t, &[1, 0])?;
        assert_eq!(permuted.shape(), &[3, 2]);
        assert_eq!(permuted.strides(), &[1, 3]);
        Ok(())
    }

    #[test]
    fn test_permute_identity() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let permuted = permute_op(&t, &[0, 1])?;
        assert_eq!(permuted.shape(), t.shape());
        assert_eq!(permuted.strides(), t.strides());
        Ok(())
    }

    #[test]
    fn test_permute_higher_dim() -> Result<(), NeuraRustError> {
        let t_data = (0..24).map(|x| x as f32).collect();
        let t_shape = vec![2, 3, 4];
        let t = Tensor::new(t_data, t_shape)?;
        let permuted = permute_op(&t, &[2, 0, 1])?;
        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(permuted.strides(), &[1, 12, 4]);
        Ok(())
    }

    #[test]
    fn test_permute_invalid_axes_length() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0f32, 2.0], vec![2])?;
        let result1 = permute_op(&t, &[0, 1]);
        assert!(matches!(result1, Err(NeuraRustError::RankMismatch { .. })));
        let result2 = permute_op(&t, &[0, 1, 0]);
        assert!(matches!(result2, Err(NeuraRustError::RankMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_permute_invalid_axis_value() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let result = permute_op(&t, &[0, 2]);
        assert!(matches!(result, Err(NeuraRustError::IndexOutOfBounds { .. })));
        Ok(())
    }

    #[test]
    fn test_permute_duplicate_axis() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let result = permute_op(&t, &[0, 0]);
        assert!(matches!(result, Err(NeuraRustError::InvalidPermutation { .. })));
        Ok(())
    }

    #[test]
    #[ignore = "Skipping due to check_grad F32 precision limitations. Backward logic visually verified."]
    fn test_permute_backward() -> Result<(), GradCheckError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        t.set_requires_grad(true)?;
        let axes = &[1, 0];

        let func = |inputs: &[Tensor]| permute_op(&inputs[0], axes);

        let output_shape = vec![3, 2];
        let output_grad = tensor::ones(&output_shape)?;

        check_grad(func, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    #[ignore = "Skipping due to check_grad F32 precision limitations. Backward logic visually verified."]
    fn test_permute_backward_higher_dim() -> Result<(), GradCheckError> {
        let t_data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
        let t_shape = vec![2, 3, 4];
        let t = Tensor::new(t_data.clone(), t_shape.clone())?;
        t.set_requires_grad(true)?;
        let axes = &[2, 0, 1];

        let func = |inputs: &[Tensor]| permute_op(&inputs[0], axes);

        let output_shape = vec![4, 2, 3];
        let output_grad = tensor::ones(&output_shape)?;

        check_grad(func, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_permute_backward_f64() -> Result<(), GradCheckError> {
        let t_data = (0..24).map(|x| x as f64).collect::<Vec<_>>();
        let t_shape = vec![2, 3, 4];
        let t = Tensor::new_f64(t_data.clone(), t_shape.clone())?;
        t.set_requires_grad(true)?;
        let axes = &[2, 0, 1];

        let func = |inputs: &[Tensor]| permute_op(&inputs[0], axes);

        let output_shape = vec![4, 2, 3];
        let output_grad = tensor::ones_f64(&output_shape)?;

        check_grad(func, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
    }
} 