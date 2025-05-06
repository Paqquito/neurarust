#[cfg(test)]
mod tests {
    use crate::ops::view::transpose::transpose_op;
    use crate::tensor::{self, Tensor};
    use crate::error::NeuraRustError;
    use crate::autograd::grad_check::{check_grad, GradCheckError};
    use std::sync::Arc;

    #[test]
    fn test_transpose_basic() -> Result<(), NeuraRustError> {
        let tensor = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3])?;
        let transposed = transpose_op(&tensor, 0, 1)?;
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());

        assert!(Arc::ptr_eq(&tensor.read_data().buffer, &transposed.read_data().buffer));
        assert_eq!(tensor.read_data().offset, transposed.read_data().offset);

        let expected_data = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        let contiguous_transposed = transposed.contiguous()?;
        assert_eq!(contiguous_transposed.get_f32_data()?, expected_data);
        Ok(())
    }

    #[test]
    fn test_transpose_higher_dim() -> Result<(), NeuraRustError> {
        let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
        let transposed = transpose_op(&t, 1, 2)?;
        assert_eq!(transposed.shape(), &[2, 4, 3]);
        assert_eq!(transposed.strides(), &[12, 1, 4]);
        assert!(!transposed.is_contiguous());

        assert!(Arc::ptr_eq(&t.read_data().buffer, &transposed.read_data().buffer));
        assert_eq!(t.read_data().offset, transposed.read_data().offset);

        let expected_contiguous_data = vec![
             0.0,  4.0,  8.0,    1.0,  5.0,  9.0,    2.0,  6.0, 10.0,    3.0,  7.0, 11.0,
            12.0, 16.0, 20.0,   13.0, 17.0, 21.0,   14.0, 18.0, 22.0,   15.0, 19.0, 23.0,
        ];
        let contiguous_transposed = transposed.contiguous()?;
        assert_eq!(contiguous_transposed.get_f32_data()?, expected_contiguous_data);
        Ok(())
    }

    #[test]
    fn test_transpose_invalid_dims() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        assert!(transpose_op(&t, 0, 1).is_ok());
        let result_err = transpose_op(&t, 0, 2);
        assert!(matches!(result_err, Err(NeuraRustError::IndexOutOfBounds { .. })), "Expected IndexOutOfBounds for dim2 > rank");
        let result_same = transpose_op(&t, 1, 1);
        assert!(result_same.is_ok(), "Transpose with dim1 == dim2 should be Ok");
        if let Ok(transposed_same) = result_same {
            assert_eq!(transposed_same.shape(), t.shape());
            assert_eq!(transposed_same.strides(), t.strides());
        }
        Ok(())
    }

    #[test]
    fn test_transpose_backward_f32() -> Result<(), GradCheckError> {
        let input = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        input.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| transpose_op(&inputs[0], 0, 1);

        let output_shape = vec![3, 2];
        let output_grad = tensor::ones(&output_shape)?;

        check_grad(func, &[input], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_transpose_backward_higher_dim_f32() -> Result<(), GradCheckError> {
        let input_data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
        let input = Tensor::new(input_data, vec![2, 3, 4])?;
        input.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| transpose_op(&inputs[0], 1, 2);

        let output_shape = vec![2, 4, 3];
        let output_grad = tensor::ones(&output_shape)?;

        check_grad(func, &[input], &output_grad, 1e-3, 1e-2, 1e-2)
    }

    #[test]
    fn test_transpose_backward_f64() -> Result<(), GradCheckError> {
        let input_data = (0..24).map(|x| x as f64).collect();
        let input = Tensor::new_f64(input_data, vec![2, 3, 4])?;
        input.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| transpose_op(&inputs[0], 0, 1);

        let output_shape = vec![3, 2, 4];
        let output_grad = tensor::ones_f64(&output_shape)?;

        check_grad(func, &[input], &output_grad, 1e-5, 1e-7, 1e-5)
    }
} 