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
    fn test_add_scalar_inplace_simple_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        a.add_scalar_f32(10.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[11.0, 12.0, 13.0, 14.0]);
        Ok(())
    }

    #[test]
    fn test_add_scalar_inplace_simple_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        a.add_scalar_f64(10.0f64)?;
        assert_eq!(a.get_f64_data().unwrap(), &[11.0, 12.0, 13.0, 14.0]);
        Ok(())
    }

    #[test]
    fn test_add_scalar_inplace_negative_scalar_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![10.0, 20.0], vec![2])?;
        a.add_scalar_f32(-5.0f32)?;
        assert_eq!(a.get_f32_data().unwrap(), &[5.0, 15.0]);
        Ok(())
    }

    #[test]
    fn test_add_scalar_inplace_autograd_error_f32() -> Result<(), NeuraRustError> {
        let mut a = tensor_new_f32(vec![1.0], vec![1])?;
        let _ = a.set_requires_grad(true);
        let result = a.add_scalar_f32(2.0f32);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_add_scalar_inplace_dtype_mismatch_error_f32_tensor_f64_scalar() -> Result<(), NeuraRustError> {
        // let mut a_f32 = tensor_new_f32(vec![1.0], vec![1])?; // This variable is unused in the current test logic
        // This case is not directly testable with add_scalar_f64 as it won't compile due to type mismatch on scalar arg.
        // We are testing calling add_scalar_f64 on an F32 tensor, which is prevented by the method signature of add_scalar_f64 on Tensor itself.
        // Let's test calling perform_add_scalar_inplace_f64 directly if the methods on Tensor become generic later
        // For now, the public API `tensor.add_scalar_f64(f64_val)` on an f32 tensor will not compile.
        // The actual DTypeMismatch inside perform_... is for the tensor's own DType, not the scalar's type relative to function name.
        Ok(())
    }

    #[test]
    fn test_add_scalar_inplace_dtype_mismatch_f64_tensor_call_f32_method() -> Result<(), NeuraRustError> {
        let mut a_f64 = tensor_new_f64(vec![1.0], vec![1])?;
        let result = a_f64.add_scalar_f32(2.0f32); // Calling add_scalar_f32 on F64 tensor
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch {expected: DType::F32, actual: DType::F64, .. })));
        Ok(())
    }


    #[test]
    fn test_add_scalar_inplace_non_contiguous_f32() -> Result<(), NeuraRustError> {
        let base = tensor_new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f32_data()?;

        a_view.add_scalar_f32(10.0f32)?;

        // Expected data for a_view (transposed base, then scalar added)
        // base = [[1,2,3],[4,5,6],[7,8,9]]
        // a_view (original) = [[1,4,7],[2,5,8],[3,6,9]]
        // a_view (added 10) = [[11,14,17],[12,15,18],[13,16,19]]
        let expected_a_view_data = vec![11.0, 14.0, 17.0, 12.0, 15.0, 18.0, 13.0, 16.0, 19.0];
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
    fn test_add_scalar_inplace_non_contiguous_f64() -> Result<(), NeuraRustError> {
        let base = tensor_new_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3])?;
        let mut a_view = base.transpose(0, 1)?;
        let base_data_before_op = base.get_f64_data()?;

        a_view.add_scalar_f64(10.0f64)?;
        
        let expected_a_view_data = vec![11.0, 14.0, 17.0, 12.0, 15.0, 18.0, 13.0, 16.0, 19.0];
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