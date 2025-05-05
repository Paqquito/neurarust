// neurarust-core/src/ops/comparison/equal_test.rs

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::ops::comparison::equal_op;
    use crate::utils::testing::check_tensor_near;

    #[test]
    fn test_equal_simple() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
        let result = equal_op(&a, &b).unwrap();
        let expected_data = vec![1.0, 0.0, 1.0, 0.0];
        check_tensor_near(&result, &[2, 2], &expected_data, 1e-7); 
        // Check that output does not require grad
        assert!(!result.requires_grad(), "Equal op output should not require grad");
        assert!(result.grad_fn().is_none(), "Equal op output should not have grad_fn");
    }

    #[test]
    fn test_equal_broadcast_lhs() {
        let a = Tensor::new(vec![3.0], vec![1]).unwrap(); // Scalar-like
        let b = Tensor::new(vec![1.0, 3.0, 3.0, 0.0], vec![2, 2]).unwrap();
        let result = equal_op(&a, &b).unwrap();
        let expected_data = vec![0.0, 1.0, 1.0, 0.0];
        check_tensor_near(&result, &[2, 2], &expected_data, 1e-7);
    }

    #[test]
    fn test_equal_broadcast_rhs() {
        let a = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(); // Broadcast along dim 0
        let result = equal_op(&a, &b).unwrap();
        let expected_data = vec![1.0, 1.0, 1.0, 1.0];
        check_tensor_near(&result, &[2, 2], &expected_data, 1e-7);
    }

    #[test]
    fn test_equal_broadcast_incompatible() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = equal_op(&a, &b);
        assert!(matches!(result, Err(crate::error::NeuraRustError::BroadcastError { .. })));
    }
    
    #[test]
    fn test_equal_float_epsilon() {
        let a = Tensor::new(vec![1.0], vec![1]).unwrap();
        let b = Tensor::new(vec![1.0000001], vec![1]).unwrap(); // Slightly different
        let c = Tensor::new(vec![1.00001], vec![1]).unwrap(); // More different
        
        // Should be equal within default epsilon (1e-6)
        let result_ab = equal_op(&a, &b).unwrap();
        check_tensor_near(&result_ab, &[1], &[1.0], 1e-7);
        
        // Should NOT be equal within default epsilon
        let result_ac = equal_op(&a, &c).unwrap();
        check_tensor_near(&result_ac, &[1], &[0.0], 1e-7);
    }
} 