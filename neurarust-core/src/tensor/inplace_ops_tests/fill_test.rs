#[cfg(test)]
pub mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;

    // Helper to create tensors easily
    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new(data, shape)
    }

    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new_f64(data, shape)
    }

    // --- Tests --- 

    #[test]
    fn test_fill_inplace_simple_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let fill_value = 5.0f32;
        t.fill_(fill_value)?;
        assert_eq!(t.get_f32_data()?, vec![5.0, 5.0, 5.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_fill_inplace_simple_f64() -> Result<(), NeuraRustError> {
        let mut t = tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
        let fill_value = -1.0f64;
        t.fill_(fill_value)?;
        assert_eq!(t.get_f64_data()?, vec![-1.0, -1.0, -1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn test_fill_inplace_autograd_error_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![1.0, 2.0], vec![2])?;
        let _ = t.set_requires_grad(true);
        let result = t.fill_(0.0f32);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_fill_inplace_non_contiguous_error() -> Result<(), NeuraRustError> {
        let base = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let mut view = base.transpose(0, 1)?; // Transpose creates a non-contiguous view
        assert!(!view.is_contiguous()); 
        let result = view.fill_(99.0f32);
        assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(msg)) if msg.contains("contiguous")));
        Ok(())
    }

    #[test]
    fn test_fill_inplace_dtype_mismatch_error() -> Result<(), NeuraRustError> {
        // Attempt to fill f32 tensor with f64 value
        let mut t_f32 = tensor_f32(vec![1.0], vec![1])?;
        let result_f64_val = t_f32.fill_(1.0f64);
        assert!(matches!(result_f64_val, Err(NeuraRustError::DataTypeMismatch { .. })));

        // Attempt to fill f64 tensor with f32 value
        let mut t_f64 = tensor_f64(vec![1.0], vec![1])?;
        let result_f32_val = t_f64.fill_(1.0f32);
        assert!(matches!(result_f32_val, Err(NeuraRustError::DataTypeMismatch { .. })));
        
        Ok(())
    }

} 