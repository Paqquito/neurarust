#[cfg(test)]
mod tests {
    use crate::ops::view::reshape::reshape_op;
    use crate::tensor::{self, Tensor};
    use crate::error::NeuraRustError;
    use crate::autograd::grad_check::{check_grad, GradCheckError};

    #[test]
    fn test_reshape_contiguous() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let reshaped = reshape_op(&t, vec![3, 2])?;
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert!(reshaped.is_contiguous());
        assert_eq!(reshaped.strides(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn test_reshape_to_scalar() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![5.0], vec![1])?;
        let reshaped = reshape_op(&t, vec![])?;
        assert_eq!(reshaped.shape(), &[] as &[usize]);
        assert_eq!(reshaped.strides(), &[] as &[usize]);
        Ok(())
    }

    #[test]
    fn test_reshape_from_scalar() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![5.0], vec![])?;
        let reshaped = reshape_op(&t, vec![1, 1, 1])?;
        assert_eq!(reshaped.shape(), &[1, 1, 1]);
        assert_eq!(reshaped.strides(), &[1, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_reshape_numel_mismatch() -> Result<(), NeuraRustError>{
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = reshape_op(&t, vec![2, 2]);
        assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_reshape_non_contiguous_error() -> Result<(), NeuraRustError> {
        let t_orig = Tensor::new((0..12).map(|x| x as f32).collect(), vec![2, 2, 3])?;
        let t_transposed = t_orig.transpose(0, 1)?;
        assert!(!t_transposed.is_contiguous());
        let result = reshape_op(&t_transposed, vec![4, 3]);
        assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(_))));
        Ok(())
    }

    #[test]
    fn test_reshape_backward_f32() -> Result<(), GradCheckError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        t.set_requires_grad(true)?;
        let target_shape = vec![3, 2];

        let reshape_fn = |inputs: &[Tensor]| reshape_op(&inputs[0], target_shape.clone());

        let output_grad = tensor::ones(&target_shape)?;

        check_grad(reshape_fn, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_reshape_backward_f64() -> Result<(), GradCheckError> {
        let t_data = (1..=6).map(|x| x as f64).collect();
        let t_shape = vec![2, 3];
        let t = Tensor::new_f64(t_data, t_shape)?;
        t.set_requires_grad(true)?;
        let target_shape = vec![3, 2];

        let reshape_fn = |inputs: &[Tensor]| reshape_op(&inputs[0], target_shape.clone());

        let output_grad = tensor::ones_f64(&target_shape)?;

        check_grad(reshape_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
    }

    #[test]
    fn test_reshape_backward_flatten_f64() -> Result<(), GradCheckError> {
        let t_data = (1..=12).map(|x| x as f64).collect();
        let t_shape = vec![2, 2, 3];
        let t = Tensor::new_f64(t_data, t_shape)?;
        t.set_requires_grad(true)?;
        let target_shape = vec![12];

        let reshape_fn = |inputs: &[Tensor]| reshape_op(&inputs[0], target_shape.clone());
        
        let output_grad = tensor::ones_f64(&target_shape)?;
        
        check_grad(reshape_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
    }

    #[test]
    fn test_reshape_backward_add_dim_f64() -> Result<(), GradCheckError> {
        let t_data = (1..=12).map(|x| x as f64).collect();
        let t_shape = vec![2, 2, 3];
        let t = Tensor::new_f64(t_data, t_shape)?;
        t.set_requires_grad(true)?;
        let target_shape = vec![2, 2, 1, 3];

        let reshape_fn = |inputs: &[Tensor]| reshape_op(&inputs[0], target_shape.clone());
        
        let output_grad = tensor::ones_f64(&target_shape)?;
        
        check_grad(reshape_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
    }
} 