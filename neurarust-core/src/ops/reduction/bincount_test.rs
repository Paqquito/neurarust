#[cfg(test)]
mod tests {
    use crate::tensor::create;
    use crate::ops::reduction::bincount_op;

    #[test]
    fn test_bincount_shape_error() {
        let t = create::from_vec_i32(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let res = bincount_op(&t, None, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_bincount_dtype_error() {
        let t = create::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let res = bincount_op(&t, None, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_bincount_basic() {
        let t = create::from_vec_i32(vec![0, 1, 1, 2, 2, 2], vec![6]).unwrap();
        let out = bincount_op(&t, None, 0).unwrap();
        let data = out.get_i32_data().unwrap();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_bincount_minlength() {
        let t = create::from_vec_i32(vec![1, 1, 2], vec![3]).unwrap();
        let out = bincount_op(&t, None, 5).unwrap();
        let data = out.get_i32_data().unwrap();
        assert_eq!(data, vec![0, 2, 1, 0, 0]);
    }

    #[test]
    fn test_bincount_negative() {
        let t = create::from_vec_i32(vec![0, -1, 2], vec![3]).unwrap();
        let res = bincount_op(&t, None, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_bincount_weights() {
        let t = create::from_vec_i32(vec![0, 1, 1, 2], vec![4]).unwrap();
        let w = create::from_vec_f32(vec![0.5, 1.0, 2.0, 3.0], vec![4]).unwrap();
        let out = bincount_op(&t, Some(&w), 0).unwrap();
        let data = out.get_f32_data().unwrap();
        assert_eq!(data, vec![0.5, 3.0, 3.0]);
    }

    #[test]
    fn test_bincount_large() {
        let n = 100_000;
        let mut values = Vec::with_capacity(n);
        for i in 0..n { values.push((i % 10) as i32); }
        let t = create::from_vec_i32(values, vec![n]).unwrap();
        let out = bincount_op(&t, None, 0).unwrap();
        let data = out.get_i32_data().unwrap();
        assert_eq!(data.len(), 10);
        for &v in &data { assert_eq!(v, n as i32 / 10); }
    }
} 