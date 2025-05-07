#[cfg(test)]
pub mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::ops::view::slice::SliceArg;

    // Helper to create tensors easily in tests
    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new(data, shape)
    }

    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
        Tensor::new_f64(data, shape)
    }

    #[test]
    fn test_clamp_inplace_min_only_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![-1.0, 0.0, 1.0, 2.5, 5.0], vec![5])?;
        t.clamp_(Some(0.5f32), None)?;
        assert_eq!(t.get_f32_data()?, vec![0.5, 0.5, 1.0, 2.5, 5.0]);
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_max_only_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![-1.0, 0.0, 1.0, 2.5, 5.0], vec![5])?;
        t.clamp_(None, Some(2.0f32))?;
        assert_eq!(t.get_f32_data()?, vec![-1.0, 0.0, 1.0, 2.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_min_and_max_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![-1.0, 0.0, 1.0, 2.5, 3.0, 5.0], vec![6])?;
        t.clamp_(Some(0.0f32), Some(2.7f32))?;
        assert_eq!(t.get_f32_data()?, vec![0.0, 0.0, 1.0, 2.5, 2.7, 2.7]);
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_no_clamping_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![0.0, 1.0, 2.0], vec![3])?;
        t.clamp_(Some(-1.0f32), Some(3.0f32))?;
        assert_eq!(t.get_f32_data()?, vec![0.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_min_gt_max_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], vec![6])?;
        t.clamp_(Some(2.0f32), Some(1.0f32))?;
        assert_eq!(t.get_f32_data()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_min_only_f64() -> Result<(), NeuraRustError> {
        let mut t = tensor_f64(vec![-1.0, 0.0, 1.0, 2.5, 5.0], vec![5])?;
        t.clamp_(Some(0.5f64), None)?;
        assert_eq!(t.get_f64_data()?, vec![0.5, 0.5, 1.0, 2.5, 5.0]);
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_autograd_error_f32() -> Result<(), NeuraRustError> {
        let mut t = tensor_f32(vec![1.0], vec![1])?;
        t.set_requires_grad(true)?;
        let result = t.clamp_(Some(0.0f32), Some(2.0f32));
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
        Ok(())
    }

    #[test]
    fn test_clamp_inplace_unsupported_dtype_error() -> Result<(), NeuraRustError> {
        // This test requires creating a tensor of a currently unsupported DType for clamp.
        // As DType only has F32, F64, this test is hard to write unless we mock DType or add one.
        // For now, we assume the panic branch for `other_dtype` in `clamp_` covers this.
        // If integer types were added to DType but not supported by clamp_, this would be testable.
        // Let's skip this specific test for now as it depends on future DType extensions.
        Ok(())
    }

    #[test]
    fn test_clamp_cow_on_view_f32() -> Result<(), NeuraRustError> {
        let base = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4])?; // Contigu [1, 2, 3, 4]
        let data_before_op = base.get_f32_data()?;

        // Créer une vue contiguë (slice simple)
        let slice_args = &[ // Utiliser un slice littéral &[...]
            SliceArg::Slice(1, 3, 1) // Prend les éléments aux indices 1 et 2
        ];
        let mut view = base.slice(slice_args)?;
        // view devrait être [2.0, 3.0] de shape [2]
        assert_eq!(view.shape(), &[2], "Unexpected view shape");
        assert!(view.is_contiguous(), "View created by simple slice should be contiguous");

        // Clamper la vue
        view.clamp_(Some(2.5f32), Some(3.5f32))?;
        // view devrait être [2.5, 3.0]

        // Vérifier la vue
        let view_data = view.get_f32_data()?;
        assert_eq!(view_data, vec![2.5, 3.0], "View data after clamp is incorrect");

        // Vérifier que le tenseur de base n'a pas été modifié
        assert_eq!(base.get_f32_data()?, data_before_op, "Base tensor was modified by clamp_ on view");
        Ok(())
    }

    #[test]
    fn test_clamp_non_contiguous_error_f32() -> Result<(), NeuraRustError> {
        let base = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let mut view = base.transpose(0, 1)?; // Crée une vue non contiguë
        assert!(!view.is_contiguous(), "View should be non-contiguous for this test");

        let result = view.clamp_(Some(0.0f32), Some(10.0f32));
        match result {
            Err(NeuraRustError::UnsupportedOperation(msg)) => {
                assert!(msg.contains("non-contiguous"));
            }
            _ => panic!("Expected UnsupportedOperation for non-contiguous clamp_"),
        }
        Ok(())
    }
} 