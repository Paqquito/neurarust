#[cfg(test)]
mod tests {
    use crate::ops::comparison::ne_op;
    use crate::tensor::Tensor;

    #[test]
    fn test_ne_f32() {
        let t1 = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
        let result = ne_op(&t1, &t2).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, false, true]);
    }

    #[test]
    fn test_ne_f32_broadcast() {
        let t1 = Tensor::new(vec![1.0f32], vec![1]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 2.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let result = ne_op(&t1, &t2).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, false, false]);
    }

    #[test]
    fn test_ne_f64() {
        let t1 = Tensor::new_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::new_f64(vec![0.0, 2.0, 3.0, 5.0], vec![2, 2]).unwrap();
        let result = ne_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, false, true]);
    }

    #[test]
    fn test_ne_i32() {
        let t1 = Tensor::new_i32(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let t2 = Tensor::new_i32(vec![1, 0, 3, 5], vec![2, 2]).unwrap();
        let result = ne_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, false, true]);
    }

    #[test]
    fn test_ne_i64() {
        let t1 = Tensor::new_i64(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let t2 = Tensor::new_i64(vec![1, 0, 3, 5], vec![2, 2]).unwrap();
        let result = ne_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, false, true]);
    }

    #[test]
    fn test_ne_bool() {
        let t1 = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let t2 = Tensor::new_bool(vec![true, true, false, false], vec![2, 2]).unwrap();
        let result = ne_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, true, false]);
    }

    #[test]
    fn test_ne_shape_mismatch() {
        let t1 = Tensor::new(vec![1.0f32, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let result = ne_op(&t1, &t2);
        assert!(result.is_err());
    }
} 