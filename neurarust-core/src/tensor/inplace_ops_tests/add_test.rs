#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::types::DType; 
    use crate::ops::view::SliceArg;

    // Helper pour créer un tenseur F32 pour les tests
    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).unwrap()
    }

    // Helper pour créer un tenseur F64 pour les tests
    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new_f64(data, shape).unwrap()
    }

    #[test]
    fn test_add_inplace_simple_correctness() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        a.add_(&b)?;

        let expected_data = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_vector() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![10.0, 20.0, 30.0], vec![3]); // Broadcast [3] to [2,3]
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_row_vector() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![10.0, 20.0, 30.0], vec![1, 3]); // Broadcast [1,3] to [2,3]
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_col_vector() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let b = tensor_f32(vec![5.0, 100.0], vec![2, 1]); // Broadcast [2,1] to [2,3]
        
        a.add_(&b)?;

        // ligne0: [1,2,3] + 5   = [6, 7, 8]
        // ligne1: [10,20,30] + 100 = [110, 120, 130]
        let expected_data = vec![6.0, 7.0, 8.0, 110.0, 120.0, 130.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_scalar() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f32(vec![10.0], vec![]); // Scalar
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 12.0, 13.0, 14.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_autograd_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0], vec![1]);
        let b = tensor_f32(vec![1.0], vec![1]);
        a.set_requires_grad(true)?;
        
        let result = a.add_(&b);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::InplaceModificationError { operation, reason: _ } => {
                assert_eq!(operation, "add_");
            }
            _ => panic!("Expected InplaceModificationError"),
        }
        Ok(())
    }

    #[test]
    fn test_add_inplace_dtype_mismatch_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0], vec![1]);
        let b_f64 = Tensor::new_f64(vec![1.0], vec![1])?;
        
        let result = a.add_(&b_f64);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::DataTypeMismatch { expected, actual, operation } => {
                assert_eq!(expected, DType::F32);
                assert_eq!(actual, DType::F64);
                assert_eq!(operation, "in-place addition (add_)");
            }
            other_err => panic!("Expected DataTypeMismatch, got {:?}", other_err),
        }
        Ok(())
    }
    
    #[test]
    fn test_add_inplace_buffer_shared_error_try_get_mut() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tensor_f32(vec![10.0], vec![1]);
        
        let _a_view = a.slice(&[SliceArg::Slice(0, 3, 1)])?; 

        let result = a.add_(&b);
        
        assert!(result.is_err(), "Expected error for in-place op on shared buffer, got {:?}", result);
        match result.err().unwrap() {
            NeuraRustError::BufferSharedError { operation } => {
                assert_eq!(operation, "add_ (buffer is shared)", "Operation was: {}", operation);
            }
            other_err => panic!("Expected BufferSharedError, got {:?}", other_err),
        }
        
        Ok(())
    }

    #[test]
    fn test_add_inplace_non_contiguous_self() -> Result<(), NeuraRustError> {
        let a_orig = tensor_f32(vec![1.,2.,3., 4.,5.,6., 7.,8.,9., 10.,11.,12.], vec![4,3]);
        let b = tensor_f32(vec![10., 20., 30., 40.], vec![4]);
        let mut a_transposed = a_orig.transpose(0,1)?;
        
        let result = a_transposed.add_(&b);

        assert!(result.is_err(), "Expected BufferSharedError because a_transposed shares buffer with a_orig");
        match result.err().unwrap() {
            NeuraRustError::BufferSharedError { operation } => {
                assert_eq!(operation, "add_ (buffer is shared)");
            }
            other_err => panic!("Expected BufferSharedError for non_contiguous test, got {:?}", other_err),
        }
        Ok(())
    }

    // --- Tests F64 ---
    #[test]
    fn test_add_inplace_simple_correctness_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        a.add_(&b)?;

        let expected_data = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![10.0, 20.0, 30.0], vec![3]);
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_row_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![10.0, 20.0, 30.0], vec![1, 3]);
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_col_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let b = tensor_f64(vec![5.0, 100.0], vec![2, 1]);
        
        a.add_(&b)?;

        let expected_data = vec![6.0, 7.0, 8.0, 110.0, 120.0, 130.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_scalar_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f64(vec![10.0], vec![]); 
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 12.0, 13.0, 14.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_non_contiguous_self_f64() -> Result<(), NeuraRustError> {
        let a_orig = tensor_f64(vec![1.,2.,3., 4.,5.,6., 7.,8.,9., 10.,11.,12.], vec![4,3]);
        let b = tensor_f64(vec![10., 20., 30., 40.], vec![4]);
        let mut a_transposed = a_orig.transpose(0,1)?;
        
        let result = a_transposed.add_(&b);
        assert!(result.is_err(), "Expected BufferSharedError because a_transposed shares buffer with a_orig (f64)");
        match result.err().unwrap() {
            NeuraRustError::BufferSharedError { operation } => {
                assert_eq!(operation, "add_ (buffer is shared)", 
                        "Unexpected BufferSharedError operation: {}", operation);
            }
            other_err => panic!("Expected BufferSharedError for non_contiguous_f64 test, got {:?}", other_err),
        }
        Ok(())
    }
} 