#[cfg(test)]
pub mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    // use crate::types::DType; // Unused import
    use crate::ops::view::SliceArg;

    // Helper function to create a new tensor for testing, similar to what might be in tensor::create
    fn tensor_new_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new(data, shape)
    }

    fn tensor_new_f64(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new_f64(data, shape)
    }

    #[test]
    fn test_mul_inplace_simple_correctness_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[5.0, 12.0, 21.0, 32.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_simple_correctness_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[5.0, 12.0, 21.0, 32.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![2.0], vec![1])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_scalar_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![2.0], vec![1])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_rhs_row_vector_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
        let b = tensor_new_f32(vec![10.0, 100.0], vec![1, 2])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[10.0, 200.0, 30.0, 400.0, 50.0, 600.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_rhs_row_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
        let b = tensor_new_f64(vec![10.0, 100.0], vec![1, 2])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[10.0, 200.0, 30.0, 400.0, 50.0, 600.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_rhs_col_vector_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
        let b = tensor_new_f32(vec![10.0, 100.0, 1000.0], vec![3, 1])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[10.0, 20.0, 300.0, 400.0, 5000.0, 6000.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_rhs_col_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
        let b = tensor_new_f64(vec![10.0, 100.0, 1000.0], vec![3, 1])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[10.0, 20.0, 300.0, 400.0, 5000.0, 6000.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_rhs_vector_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f32(vec![10.0, 100.0], vec![2])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f32_data().unwrap(), &[10.0, 200.0, 30.0, 400.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcasting_rhs_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = tensor_new_f64(vec![10.0, 100.0], vec![2])?;
        a.mul_(&b)?;
        assert_eq!(a.get_f64_data().unwrap(), &[10.0, 200.0, 30.0, 400.0]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace_dtype_mismatch_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let b = tensor_new_f64(vec![1.0], vec![1])?;
        let result = a.mul_(&b);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_mul_inplace_autograd_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let _ = a.set_requires_grad(true);
        let b = tensor_new_f32(vec![1.0], vec![1])?;
        let result = a.mul_(&b);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_mul_inplace_broadcast_error_shape_change() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0], vec![2])?;
        let b = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let result = a.mul_(&b);
        assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
        Ok(())
    }

    #[test]
    fn test_mul_inplace_buffer_shared_error_try_get_mut() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let _a_view = a.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 2, 1)])?; 

        let b = tensor_new_f32(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2])?;
        
        let result = a.mul_(&b);
        assert!(result.is_ok(), "mul_ should succeed with CoW, got {:?}", result);

        assert_eq!(a.get_f32_data().unwrap(), &[2.0, 4.0, 6.0, 8.0], "Tensor 'a' was not modified as expected after CoW mul_.");
        
        Ok(())
    }

    #[test]
    fn test_mul_inplace_non_contiguous_self() -> Result<(), NeuraRustError> {
        let base = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a = base.transpose(0, 1)?;
        let b = tensor_new_f32(vec![2.0; 9], vec![3, 3])?;
        
        // Capture data of base *before* mul_ on a
        let base_data_before_op = base.get_f32_data()?;

        a.mul_(&b)?;

        let expected_a_data_after_mul = vec![2.0, 8.0, 14.0, 4.0, 10.0, 16.0, 6.0, 12.0, 18.0];
        let mut current_a_data = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                current_a_data.push(a.at_f32(&[i,j])?);
            }
        }
        assert_eq!(current_a_data.as_slice(), expected_a_data_after_mul.as_slice(), "Tensor 'a' (view) was not modified correctly.");

        // With CoW, base should NOT be modified.
        assert_eq!(base.get_f32_data()?.as_slice(), base_data_before_op.as_slice(), "Tensor 'base' should not be modified after CoW on view 'a'.");
        Ok(())
    }

    #[test]
    fn test_mul_inplace_non_contiguous_self_f64() -> Result<(), NeuraRustError> {
        let base = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a = base.transpose(0, 1)?;
        let b = tensor_new_f64(vec![2.0; 9], vec![3, 3])?;

        // Capture data of base *before* mul_ on a
        let base_data_before_op = base.get_f64_data()?;

        a.mul_(&b)?;

        let expected_a_data_after_mul = vec![2.0, 8.0, 14.0, 4.0, 10.0, 16.0, 6.0, 12.0, 18.0];
        let mut current_a_data: Vec<f64> = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 { 
                current_a_data.push(a.at_f64(&[i,j])?);
            }
        }
        assert_eq!(current_a_data.as_slice(), expected_a_data_after_mul.as_slice(), "Tensor 'a' (view) was not modified correctly.");

        // With CoW, base should NOT be modified.
        assert_eq!(base.get_f64_data()?.as_slice(), base_data_before_op.as_slice(), "Tensor 'base' should not be modified after CoW on view 'a'.");
        Ok(())
    }
} 