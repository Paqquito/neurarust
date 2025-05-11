#[cfg(test)]
mod tests {
    use crate::Tensor;
    use crate::ops::reduction::any::any_op;
    use crate::error::NeuraRustError;

    #[test]
    fn test_any_global_true() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![false, true, false], vec![3])?;
        let result = any_op(&t, None, false)?;
        let data = result.read_data().buffer().try_get_cpu_bool()?.clone();
        assert_eq!(result.shape(), &[]);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], true);
        Ok(())
    }
    #[test]
    fn test_any_global_false() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![false, false, false], vec![3])?;
        let result = any_op(&t, None, false)?;
        let data = result.read_data().buffer().try_get_cpu_bool()?.clone();
        assert_eq!(result.shape(), &[]);
        assert_eq!(data[0], false);
        Ok(())
    }
    #[test]
    fn test_any_axis0() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![false, true, false, false], vec![2, 2])?;
        let result = any_op(&t, Some(&[0]), false)?;
        let data = result.read_data().buffer().try_get_cpu_bool()?.clone();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(data, vec![false, true].into());
        Ok(())
    }
    #[test]
    fn test_any_axis1_keepdims() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![false, true, false, false], vec![2, 2])?;
        let result = any_op(&t, Some(&[1]), true)?;
        let data = result.read_data().buffer().try_get_cpu_bool()?.clone();
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(data, vec![true, false].into());
        Ok(())
    }
    #[test]
    fn test_any_empty() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![], vec![0])?;
        let result = any_op(&t, None, false)?;
        let data = result.read_data().buffer().try_get_cpu_bool()?.clone();
        assert_eq!(result.shape(), &[]);
        assert_eq!(data[0], false); // Par convention
        Ok(())
    }
} 