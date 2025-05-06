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
    fn test_pow_inplace_simple_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        a.pow_f32(2.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
        Ok(())
    }

    #[test]
    fn test_pow_inplace_simple_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        a.pow_f64(3.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[1.0, 8.0, 27.0, 64.0]);
        Ok(())
    }

    #[test]
    fn test_pow_inplace_fractional_exponent_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2])?;
        a.pow_f32(0.5f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_pow_inplace_negative_base_integer_exponent_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![-2.0, -3.0], vec![2])?;
        a.pow_f32(2.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[4.0, 9.0]);
        let mut b = tensor_new_f32(vec![-2.0, -3.0], vec![2])?;
        b.pow_f32(3.0f32)?;
        assert_eq!(b.get_f32_data().unwrap(), &[-8.0, -27.0]);
        Ok(())
    }
    
    #[test]
    fn test_pow_inplace_negative_base_fractional_exponent_error_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![-4.0], vec![1])?;
        let result = a.pow_f32(0.5f32);
        assert!(matches!(result, Err(NeuraRustError::ArithmeticError(_))));
        Ok(())
    }

    #[test]
    fn test_pow_inplace_zero_base_zero_exponent_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![0.0, 1.0, 0.0], vec![3])?;
        a.pow_f32(0.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[1.0, 1.0, 1.0]); // 0^0 = 1
        Ok(())
    }

    #[test]
    fn test_pow_inplace_zero_base_positive_exponent_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![0.0, 2.0], vec![2])?;
        a.pow_f32(2.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[0.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_pow_inplace_autograd_error_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let _ = a.set_requires_grad(true);
        let result = a.pow_f32(2.0f32);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_pow_inplace_dtype_mismatch_error() -> Result<(), NeuraRustError> {
        // let mut a_f32 = tensor_new_f32(vec![1.0], vec![1])?; // Unused
        
        // To correctly test this, we need a Tensor method that could potentially accept a mismatched exponent type
        // but our pow_f32 and pow_f64 are strictly typed for the exponent. 
        // The DTypeMismatch for the *tensor itself* is tested by calling the wrong function, e.g.:
        let mut a_f64 = tensor_new_f64(vec![1.0], vec![1])?;
        let result_f32_on_f64 = a_f64.pow_f32(2.0f32); // Calling pow_f32 on F64 tensor
        assert!(matches!(result_f32_on_f64, Err(NeuraRustError::DataTypeMismatch {expected: DType::F32, actual: DType::F64, .. })));
        Ok(())
    }

    #[test]
    fn test_pow_inplace_non_contiguous_f32() -> Result<(), NeuraRustError> {
        let base = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f32_data()?;

        a_view.pow_f32(2.0f32)?;

        // Expected data for a_view (transposed base, then squared)
        // base = [[1,2,3],[4,5,6],[7,8,9]]
        // a_view (original) = [[1,4,7],[2,5,8],[3,6,9]]
        // a_view (squared) = [[1,16,49],[4,25,64],[9,36,81]]
        let expected_a_view_data = vec![1.0, 16.0, 49.0, 4.0, 25.0, 64.0, 9.0, 36.0, 81.0];
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

    // F64 Variants
    #[test]
    fn test_pow_inplace_fractional_exponent_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2])?;
        a.pow_f64(0.5f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_pow_inplace_negative_base_integer_exponent_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![-2.0, -3.0], vec![2])?;
        a.pow_f64(2.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[4.0, 9.0]);
        let mut b = tensor_new_f64(vec![-2.0, -3.0], vec![2])?;
        b.pow_f64(3.0f64)?;
        assert_eq!(b.get_f64_data().unwrap(), &[-8.0, -27.0]);
        Ok(())
    }

    #[test]
    fn test_pow_inplace_negative_base_fractional_exponent_error_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![-4.0], vec![1])?;
        let result = a.pow_f64(0.5f64);
        assert!(matches!(result, Err(NeuraRustError::ArithmeticError(_))));
        Ok(())
    }

    #[test]
    fn test_pow_inplace_zero_base_zero_exponent_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![0.0, 1.0, 0.0], vec![3])?;
        a.pow_f64(0.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[1.0, 1.0, 1.0]); // 0^0 = 1
        Ok(())
    }

    #[test]
    fn test_pow_inplace_zero_base_positive_exponent_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![0.0, 2.0], vec![2])?;
        a.pow_f64(2.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[0.0, 4.0]);
        Ok(())
    }
    
    // Note: Corrected the return type for error case from `Ok(())` to nothing.
} 