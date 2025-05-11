// neurarust-core/src/ops/comparison/equal_test.rs

#[cfg(test)]
mod tests {
    use crate::ops::comparison::equal_op;
    use crate::tensor::Tensor;

    #[test]
    fn test_equal_f32() {
        let t1 = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 0.0, 3.0], vec![3]).unwrap();
        let result = equal_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, true]);
        assert!(!result.requires_grad());
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn test_equal_f32_broadcast() {
        let a_scalar = Tensor::new(vec![2.0f32], vec![1]).unwrap();
        let b_mat = Tensor::new(vec![1.0f32, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let result = equal_op(&a_scalar, &b_mat).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, true, false]);
    }

    #[test]
    fn test_equal_f64() {
        let t1 = Tensor::new_f64(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::new_f64(vec![1.0, 0.0, 3.0], vec![3]).unwrap();
        let result = equal_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, true]);
    }

    #[test]
    fn test_equal_i32() {
        let t1 = Tensor::new_i32(vec![1, 2, 3], vec![3]).unwrap();
        let t2 = Tensor::new_i32(vec![1, 0, 3], vec![3]).unwrap();
        let result = equal_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, true]);
    }

    #[test]
    fn test_equal_i64() {
        let t1 = Tensor::new_i64(vec![1, 2, 3], vec![3]).unwrap();
        let t2 = Tensor::new_i64(vec![1, 0, 3], vec![3]).unwrap();
        let result = equal_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, true]);
    }

    #[test]
    fn test_equal_bool() {
        let t1 = Tensor::new_bool(vec![true, false, true], vec![3]).unwrap();
        let t2 = Tensor::new_bool(vec![true, true, false], vec![3]).unwrap();
        let result = equal_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, false]);
    }

    #[test]
    fn test_equal_broadcast_incompatible() {
        let a = Tensor::new(vec![1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let result = equal_op(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_equal_float_epsilon() {
        let a = Tensor::new(vec![1.0f32, 2.0 + 1e-7, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![1.0f32, 2.0, 3.0 - 1e-7], vec![3]).unwrap();
        let result_ab = equal_op(&a, &b).unwrap();
        assert_eq!(result_ab.get_bool_data().unwrap(), vec![true, true, true]);
        let result_ac = equal_op(&a, &a).unwrap();
        assert_eq!(result_ac.get_bool_data().unwrap(), vec![true, true, true]);
    }
} 