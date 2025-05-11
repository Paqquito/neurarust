#[cfg(test)]
mod tests {
    use super::super::where_op::where_op;
    use crate::tensor::create;
    use crate::error::NeuraRustError;

    #[test]
    fn test_where_op_basic_f32() {
        let cond = create::from_vec_bool(vec![true, false, true], vec![3]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let out = where_op(&cond, &x, &y);
        assert!(out.is_ok());
        let out = out.unwrap();
        let data = out.get_f32_data().unwrap();
        assert_eq!(data, &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_where_op_basic_i32() {
        let cond = create::from_vec_bool(vec![false, true], vec![2]).unwrap();
        let x = create::from_vec_i32(vec![5, 6], vec![2]).unwrap();
        let y = create::from_vec_i32(vec![7, 8], vec![2]).unwrap();
        let out = where_op(&cond, &x, &y).unwrap();
        let data = out.get_i32_data().unwrap();
        assert_eq!(data, &[7, 6]);
    }

    #[test]
    fn test_where_op_broadcast() {
        let cond = create::from_vec_bool(vec![true, false, true, false], vec![4, 1]).unwrap();
        let x = create::from_vec_f64(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let y = create::from_vec_f64(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], vec![4, 2]).unwrap();
        let out = where_op(&cond, &x, &y).unwrap();
        let data = out.get_f64_data().unwrap();
        assert_eq!(data, &[1.0, 2.0, 30.0, 40.0, 1.0, 2.0, 70.0, 80.0]);
    }

    #[test]
    fn test_where_op_error_dtype_condition() {
        let cond = create::from_vec_f32(vec![1.0, 0.0], vec![2]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let y = create::from_vec_f32(vec![3.0, 4.0], vec![2]).unwrap();
        let out = where_op(&cond, &x, &y);
        assert!(matches!(out, Err(NeuraRustError::DataTypeMismatch { .. })));
    }

    #[test]
    fn test_where_op_error_dtype_x_y() {
        let cond = create::from_vec_bool(vec![true, false], vec![2]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let y = create::from_vec_i32(vec![3, 4], vec![2]).unwrap();
        let out = where_op(&cond, &x, &y);
        assert!(matches!(out, Err(NeuraRustError::DataTypeMismatch { .. })));
    }

    #[test]
    fn test_where_op_error_shape() {
        let cond = create::from_vec_bool(vec![true, false, true], vec![3]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let y = create::from_vec_f32(vec![3.0, 4.0], vec![2]).unwrap();
        let out = where_op(&cond, &x, &y);
        assert!(matches!(out, Err(NeuraRustError::BroadcastError { .. })));
    }

    #[test]
    fn test_where_op_autograd_basic() {
        let cond = create::from_vec_bool(vec![true, false], vec![2]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        x.set_requires_grad(true).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0], vec![2]).unwrap();
        y.set_requires_grad(true).unwrap();
        let out = where_op(&cond, &x, &y).unwrap();
        let loss = out.sum(None, false).unwrap();
        loss.backward(None).unwrap();
        let grad_x = x.grad().unwrap().get_f32_data().unwrap();
        let grad_y = y.grad().unwrap().get_f32_data().unwrap();
        assert_eq!(grad_x, vec![1.0, 0.0]);
        assert_eq!(grad_y, vec![0.0, 1.0]);
    }

    #[test]
    fn test_where_op_autograd_broadcast() {
        let cond = create::from_vec_bool(vec![true, false, true, false], vec![4, 1]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        x.set_requires_grad(true).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], vec![4, 2]).unwrap();
        y.set_requires_grad(true).unwrap();
        let out = where_op(&cond, &x, &y).unwrap();
        let loss = out.sum(None, false).unwrap();
        loss.backward(None).unwrap();
        let grad_x = x.grad().unwrap().get_f32_data().unwrap();
        assert_eq!(grad_x, vec![2.0, 2.0]);
        let grad_y = y.grad().unwrap().get_f32_data().unwrap();
        assert_eq!(grad_y, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_where_op_autograd_condition_not_differentiable() {
        let cond = create::from_vec_bool(vec![true, false], vec![2]).unwrap();
        cond.set_requires_grad(true).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0], vec![2]).unwrap();
        let out = where_op(&cond, &x, &y).unwrap();
        let loss = out.sum(None, false).unwrap();
        loss.backward(None).unwrap();
        assert!(cond.grad().is_none());
    }

    #[test]
    fn test_where_op_autograd_broadcast_extreme() {
        let cond = create::from_vec_bool(vec![true], vec![1, 1]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 4]).unwrap();
        x.set_requires_grad(true).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0], vec![3, 4]).unwrap();
        y.set_requires_grad(true).unwrap();
        let out = where_op(&cond, &x, &y).unwrap();
        let loss = out.sum(None, false).unwrap();
        loss.backward(None).unwrap();
        let grad_x = x.grad().unwrap().get_f32_data().unwrap();
        assert_eq!(grad_x, vec![1.0; 12]);
        let grad_y = y.grad().unwrap().get_f32_data().unwrap();
        assert_eq!(grad_y, vec![0.0; 12]);
    }

    #[test]
    fn test_where_cond_method_basic() {
        let cond = create::from_vec_bool(vec![true, false, true, false], vec![4]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let out = x.where_cond(&cond, &y).unwrap();
        assert_eq!(out.get_f32_data().unwrap(), vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_cond_method_broadcast() {
        let cond = create::from_vec_bool(vec![true, false], vec![2, 1]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let out = x.where_cond(&cond, &y).unwrap();
        assert_eq!(out.get_f32_data().unwrap(), vec![1.0, 2.0, 30.0, 40.0]);
    }

    #[test]
    fn test_where_cond_method_error_dtype() {
        let cond = create::from_vec_bool(vec![true, false], vec![2]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let y = create::from_vec_i32(vec![10, 20], vec![2]).unwrap();
        let res = x.where_cond(&cond, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_where_cond_method_autograd() {
        let cond = create::from_vec_bool(vec![true, false], vec![2]).unwrap();
        let x = create::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        x.set_requires_grad(true).unwrap();
        let y = create::from_vec_f32(vec![10.0, 20.0], vec![2]).unwrap();
        y.set_requires_grad(true).unwrap();
        let out = x.where_cond(&cond, &y).unwrap();
        let loss = out.sum(None, false).unwrap();
        loss.backward(None).unwrap();
        let grad_x = x.grad().unwrap().get_f32_data().unwrap();
        let grad_y = y.grad().unwrap().get_f32_data().unwrap();
        assert_eq!(grad_x, vec![1.0, 0.0]);
        assert_eq!(grad_y, vec![0.0, 1.0]);
    }
} 