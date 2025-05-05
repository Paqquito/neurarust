// neurarust-core/src/ops/comparison/equal_test.rs

#[cfg(test)]
mod tests {
    use crate::ops::comparison::equal_op;
    use crate::utils::testing::check_tensor_near;
    use crate::error::NeuraRustError;

    #[test]
    fn test_equal_simple() -> Result<(), NeuraRustError> {
        let a = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3])?;
        let b = crate::tensor::from_vec_f32(vec![1.0, 0.0, 3.0], vec![3])?;
        let result = equal_op(&a, &b)?;
        let expected_data = vec![1.0, 0.0, 1.0];
        check_tensor_near(&result, &[3], &expected_data, 1e-7); 
        // Check that output does not require grad
        assert!(!result.requires_grad(), "Equal op output should not require grad");
        assert!(result.grad_fn().is_none(), "Equal op output should not have grad_fn");
        Ok(())
    }

    #[test]
    fn test_equal_broadcast_lhs() -> Result<(), NeuraRustError> {
        let a_scalar = crate::tensor::from_vec_f32(vec![2.0], vec![1])?;
        let b_mat = crate::tensor::from_vec_f32(vec![1.0, 2.0, 2.0, 3.0], vec![2, 2])?;
        let result = equal_op(&a_scalar, &b_mat)?;
        let expected_data = vec![0.0, 1.0, 1.0, 0.0];
        check_tensor_near(&result, &[2, 2], &expected_data, 1e-7);
        Ok(())
    }

    #[test]
    fn test_equal_broadcast_rhs() -> Result<(), NeuraRustError> {
        let a_mat = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b_row = crate::tensor::from_vec_f32(vec![3.0, 4.0], vec![1, 2])?;
        let result = equal_op(&a_mat, &b_row)?;
        let expected_data = vec![0.0, 0.0, 1.0, 1.0];
        check_tensor_near(&result, &[2, 2], &expected_data, 1e-7);
        Ok(())
    }

    #[test]
    fn test_equal_broadcast_incompatible() {
        let a = crate::tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let b = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = equal_op(&a, &b);
        assert!(matches!(result, Err(crate::error::NeuraRustError::BroadcastError { .. })));
    }
    
    #[test]
    fn test_equal_float_epsilon() -> Result<(), NeuraRustError> {
        let a = crate::tensor::from_vec_f32(vec![1.0, 2.0 + 1e-7, 3.0], vec![3])?;
        let b = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0 - 1e-7], vec![3])?;
        
        // Should be equal within default epsilon (1e-6)
        let result_ab = equal_op(&a, &b)?;
        check_tensor_near(&result_ab, &[3], &[1.0, 1.0, 1.0], 1e-7);
        
        // Should NOT be equal within default epsilon
        let result_ac = equal_op(&a, &a)?;
        check_tensor_near(&result_ac, &[3], &[1.0, 1.0, 1.0], 1e-7);
        Ok(())
    }
} 