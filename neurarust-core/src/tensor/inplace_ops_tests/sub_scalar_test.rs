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
    fn test_sub_scalar_inplace_simple_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        a.sub_scalar_f32(5.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[5.0, 15.0, 25.0, 35.0]);
        Ok(())
    }

    #[test]
    fn test_sub_scalar_inplace_simple_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        a.sub_scalar_f64(5.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[5.0, 15.0, 25.0, 35.0]);
        Ok(())
    }

    #[test]
    fn test_sub_scalar_inplace_positive_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 0.0], vec![2])?;
        a.sub_scalar_f32(5.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[5.0, -5.0]);
        Ok(())
    }
    
    #[test]
    fn test_sub_scalar_inplace_negative_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 0.0], vec![2])?;
        a.sub_scalar_f32(-5.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[15.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_sub_scalar_inplace_autograd_error_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let _ = a.set_requires_grad(true);
        let result = a.sub_scalar_f32(2.0f32);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_sub_scalar_inplace_dtype_mismatch_f64_tensor_call_f32_method() -> Result<(), NeuraRustError> {
        let mut a_f64 = tensor_new_f64(vec![1.0], vec![1])?;
        let result = a_f64.sub_scalar_f32(2.0f32);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch {expected: DType::F32, actual: DType::F64, .. })));
        Ok(())
    }

    #[test]
    fn test_sub_scalar_inplace_non_contiguous_f32() -> Result<(), NeuraRustError> {
        let base = tensor_new_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f32_data()?;

        a_view.sub_scalar_f32(5.0f32)?;

        // Expected data for a_view (transposed base, then scalar subtracted)
        // base = [[10,20,30],[40,50,60],[70,80,90]]
        // a_view (original) = [[10,40,70],[20,50,80],[30,60,90]]
        // a_view (subtracted 5) = [[5,35,65],[15,45,75],[25,55,85]]
        let expected_a_view_data = vec![5.0, 35.0, 65.0, 15.0, 45.0, 75.0, 25.0, 55.0, 85.0];
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
    fn test_sub_scalar_inplace_non_contiguous_f64() -> Result<(), NeuraRustError> {
        let base = tensor_new_f64(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f64_data()?;

        a_view.sub_scalar_f64(5.0f64)?;
        
        let expected_a_view_data = vec![5.0, 35.0, 65.0, 15.0, 45.0, 75.0, 25.0, 55.0, 85.0];
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