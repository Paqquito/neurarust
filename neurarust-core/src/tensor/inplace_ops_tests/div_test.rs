#[cfg(test)]
pub mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::ops::view::SliceArg;

    fn tensor_new_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new(data, shape)
    }

    fn tensor_new_f64(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new_f64(data, shape)
    }

    #[test]
    fn test_div_inplace_simple_correctness_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![2.0, 5.0, 2.0, 4.0], vec![2, 2])?;
        a.div_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[5.0, 4.0, 15.0, 10.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_simple_correctness_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![2.0, 5.0, 2.0, 4.0], vec![2, 2])?;
        a.div_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[5.0, 4.0, 15.0, 10.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![2.0], vec![1])?;
        a.div_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_scalar_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![2.0], vec![1])?;
        a.div_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_rhs_row_vector_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 200.0, 30.0, 400.0, 50.0, 600.0], vec![3, 2])?;
        let b = tensor_new_f32(vec![10.0, 100.0], vec![1, 2])?;
        a.div_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_rhs_row_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![10.0, 200.0, 30.0, 400.0, 50.0, 600.0], vec![3, 2])?;
        let b = tensor_new_f64(vec![10.0, 100.0], vec![1, 2])?;
        a.div_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_rhs_col_vector_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 20.0, 300.0, 400.0, 5000.0, 6000.0], vec![3, 2])?;
        let b = tensor_new_f32(vec![10.0, 100.0, 1000.0], vec![3, 1])?;
        a.div_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_rhs_col_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![10.0, 20.0, 300.0, 400.0, 5000.0, 6000.0], vec![3, 2])?;
        let b = tensor_new_f64(vec![10.0, 100.0, 1000.0], vec![3, 1])?;
        a.div_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_rhs_vector_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 200.0, 30.0, 400.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![10.0, 100.0], vec![2])?;
        a.div_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcasting_rhs_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![10.0, 200.0, 30.0, 400.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![10.0, 100.0], vec![2])?;
        a.div_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_div_inplace_dtype_mismatch_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let b = tensor_new_f64(vec![1.0], vec![1])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_div_inplace_autograd_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let _ = a.set_requires_grad(true);
        let b = tensor_new_f32(vec![1.0], vec![1])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_div_inplace_broadcast_error_shape_change() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0], vec![2])?;
        let b = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
        Ok(())
    }

    #[test]
    fn test_div_inplace_buffer_shared_error_try_get_mut() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2])?;
        let _a_view = a.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 2, 1)])?; 

        let b = tensor_new_f32(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2])?;
        
        let result = a.div_(&b);
        
        match result {
            Err(NeuraRustError::BufferSharedError { operation }) => {
                assert_eq!(operation, "div_ (buffer is shared)", "Unexpected operation string in BufferSharedError.");
            }
            _ => panic!("Expected BufferSharedError with operation 'div_ (buffer is shared)', got {:?}", result),
        }
        
        Ok(())
    }

    #[test]
    fn test_div_inplace_non_contiguous_self() -> Result<(), NeuraRustError> {
        let base = tensor_new_f32(vec![2.0, 8.0, 14.0, 4.0, 10.0, 16.0, 6.0, 12.0, 18.0], vec![3, 3])?;
        let mut a = base.transpose(0, 1)?;
        let b = tensor_new_f32(vec![2.0; 9], vec![3, 3])?;
        
        let result = a.div_(&b);

        match result {
            Err(NeuraRustError::BufferSharedError { operation }) => {
                assert_eq!(operation, "div_ (buffer is shared)", "Unexpected operation string in BufferSharedError for non_contiguous_self.");
            }
            _ => panic!("Expected BufferSharedError for non_contiguous_self, got {:?}", result),
        }
        Ok(())
    }

    #[test]
    fn test_div_inplace_non_contiguous_self_f64() -> Result<(), NeuraRustError> {
        let base = tensor_new_f64(vec![2.0, 8.0, 14.0, 4.0, 10.0, 16.0, 6.0, 12.0, 18.0], vec![3, 3])?;
        let mut a = base.transpose(0, 1)?;
        let b = tensor_new_f64(vec![2.0; 9], vec![3, 3])?;

        let result = a.div_(&b);

        match result {
            Err(NeuraRustError::BufferSharedError { operation }) => {
                assert_eq!(operation, "div_ (buffer is shared)", "Unexpected operation string in BufferSharedError for non_contiguous_self_f64.");
            }
            _ => panic!("Expected BufferSharedError for non_contiguous_self_f64, got {:?}", result),
        }
        Ok(())
    }

    #[test]
    fn test_div_inplace_division_by_zero_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0], vec![2])?;
        let b = tensor_new_f32(vec![1.0, 0.0], vec![2])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::ArithmeticError(message)) if message == "Division by zero in div_ (F32)."));
        Ok(())
    }

    #[test]
    fn test_div_inplace_division_by_zero_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0], vec![2])?;
        let b = tensor_new_f64(vec![1.0, 0.0], vec![2])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::ArithmeticError(message)) if message == "Division by zero in div_ (F64)."));
        Ok(())
    }

     #[test]
    fn test_div_inplace_division_by_zero_broadcast_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![0.0], vec![1])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::ArithmeticError(message)) if message == "Division by zero in div_ (F32)."));
        Ok(())
    }

    #[test]
    fn test_div_inplace_division_by_zero_broadcast_scalar_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![0.0], vec![1])?;
        let result = a.div_(&b);
        assert!(matches!(result, Err(NeuraRustError::ArithmeticError(message)) if message == "Division by zero in div_ (F64)."));
        Ok(())
    }
} 