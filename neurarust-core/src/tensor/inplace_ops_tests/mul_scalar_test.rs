#[cfg(test)]
pub mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::types::DType;

    fn tensor_new_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new(data, shape)
    }

    fn tensor_new_f64(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new_f64(data, shape)
    }

    #[test]
    fn test_mul_scalar_inplace_simple_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        a.mul_scalar_f32(10.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[10.0, 20.0, 30.0, 40.0]);
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_simple_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        a.mul_scalar_f64(10.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[10.0, 20.0, 30.0, 40.0]);
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_negative_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, -2.0, 3.0], vec![3])?;
        a.mul_scalar_f32(-2.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[-2.0, 4.0, -6.0]);
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_by_zero_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 20.0, -5.0], vec![3])?;
        a.mul_scalar_f32(0.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_autograd_error_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let _ = a.set_requires_grad(true);
        let result = a.mul_scalar_f32(2.0f32);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_dtype_mismatch_f64_tensor_call_f32_method() -> Result<(), NeuraRustError> {
        let mut a_f64 = tensor_new_f64(vec![1.0], vec![1])?;
        let result = a_f64.mul_scalar_f32(2.0f32);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch {expected: DType::F32, actual: DType::F64, .. })));
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_non_contiguous_f32() -> Result<(), NeuraRustError> {
        let base = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f32_data()?;

        a_view.mul_scalar_f32(3.0f32)?;

        // Expected data for a_view (transposed base, then scalar multiplied)
        // base = [[1,2,3],[4,5,6],[7,8,9]]
        // a_view (original) = [[1,4,7],[2,5,8],[3,6,9]]
        // a_view (multiplied by 3) = [[3,12,21],[6,15,24],[9,18,27]]
        let expected_a_view_data = vec![3.0, 12.0, 21.0, 6.0, 15.0, 24.0, 9.0, 18.0, 27.0];
        let mut current_a_view_data = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                current_a_view_data.push(a_view.at_f32(&[i, j])?);
            }
        }
        assert_eq!(current_a_view_data, expected_a_view_data, "Tensor 'a_view' was not modified correctly.");
        assert_eq!(base.get_f32_data()?, base_data_before_op, "Tensor 'base' should not be modified after CoW on view.");
        Ok(())
    }

    #[test]
    fn test_mul_scalar_inplace_non_contiguous_f64() -> Result<(), NeuraRustError> {
        let base = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f64_data()?;

        a_view.mul_scalar_f64(2.0f64)?;
        
        // a_view (multiplied by 2) = [[2,8,14],[4,10,16],[6,12,18]]
        let expected_a_view_data = vec![2.0, 8.0, 14.0, 4.0, 10.0, 16.0, 6.0, 12.0, 18.0];
        let mut current_a_view_data = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                current_a_view_data.push(a_view.at_f64(&[i, j])?);
            }
        }
        assert_eq!(current_a_view_data, expected_a_view_data, "Tensor 'a_view' was not modified correctly.");
        assert_eq!(base.get_f64_data()?, base_data_before_op, "Tensor 'base' should not be modified after CoW on view.");
        Ok(())
    }
} 