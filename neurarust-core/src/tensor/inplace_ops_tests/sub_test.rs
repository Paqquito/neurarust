#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    // use crate::types::DType; // DType n'est pas directement utilisé ici pour les assertions après refactor
    use crate::ops::view::SliceArg;

    // Helper pour créer un tenseur F32 pour les tests
    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).unwrap()
    }

    // Helper pour créer un tenseur F64 pour les tests
    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new_f64(data, shape).unwrap()
    }

    // --- Correctness Tests for sub_ ---
    #[test]
    fn test_sub_inplace_simple_correctness_f32() {
        let mut a = tensor_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let b = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f32_data().unwrap(), &[9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_sub_inplace_simple_correctness_f64() {
        let mut a = tensor_f64(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let b = tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f64_data().unwrap(), &[9.0, 18.0, 27.0, 36.0]);
    }

    // --- Broadcasting Tests for sub_ (F32) ---
    #[test]
    fn test_sub_inplace_broadcasting_rhs_vector_f32() {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]); // Broadcast across rows
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f32_data().unwrap(), &[0.0, 0.0, 0.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_sub_inplace_broadcasting_rhs_row_vector_f32() {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![1.0, 0.0, -1.0], vec![1, 3]); // Broadcast across rows
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f32_data().unwrap(), &[0.0, 2.0, 4.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_sub_inplace_broadcasting_rhs_col_vector_f32() {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![1.0, 3.0], vec![2, 1]); // Broadcast across columns
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f32_data().unwrap(), &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sub_inplace_broadcasting_scalar_f32() {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tensor_f32(vec![10.0], vec![1]); // Scalar
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f32_data().unwrap(), &[-9.0, -8.0, -7.0]);
    }

    // --- Broadcasting Tests for sub_ (F64) ---
    #[test]
    fn test_sub_inplace_broadcasting_rhs_vector_f64() {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![1.0, 2.0, 3.0], vec![3]);
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f64_data().unwrap(), &[0.0, 0.0, 0.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_sub_inplace_broadcasting_rhs_row_vector_f64() {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![1.0, 0.0, -1.0], vec![1, 3]);
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f64_data().unwrap(), &[0.0, 2.0, 4.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_sub_inplace_broadcasting_rhs_col_vector_f64() {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![1.0, 3.0], vec![2, 1]);
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f64_data().unwrap(), &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sub_inplace_broadcasting_scalar_f64() {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tensor_f64(vec![10.0], vec![1]);
        a.sub_(&b).unwrap();
        assert_eq!(a.get_f64_data().unwrap(), &[-9.0, -8.0, -7.0]);
    }

    // --- Error Tests for sub_ ---
    #[test]
    fn test_sub_inplace_autograd_error() {
        let mut a = tensor_f32(vec![1.0], vec![1]);
        a.set_requires_grad(true).unwrap();
        let b = tensor_f32(vec![1.0], vec![1]);
        let result = a.sub_(&b);
        assert!(matches!(result, Err(NeuraRustError::InplaceModificationError { .. })));
    }

    #[test]
    fn test_sub_inplace_dtype_mismatch_error() {
        let mut a = tensor_f32(vec![1.0], vec![1]);
        let b = tensor_f64(vec![1.0], vec![1]);
        let result = a.sub_(&b);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch { .. })));
    }
    
    #[test]
    fn test_sub_inplace_broadcast_error_shape_change() {
        let mut a = tensor_f32(vec![1.0, 2.0], vec![2]);
        let b = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]); // Cannot broadcast b to a's shape
        let result = a.sub_(&b);
        assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
    }

    #[test]
    fn test_sub_inplace_buffer_shared_error_try_get_mut() {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let slice_args = vec![SliceArg::Slice(0, 3, 1)];
        let a_view = a.slice(slice_args.as_slice()).unwrap();
        let b = tensor_f32(vec![1.0, 1.0, 1.0], vec![3]);
        let result = a.sub_(&b);
        assert!(matches!(result, Err(NeuraRustError::BufferSharedError { .. })));
        assert_eq!(a_view.get_f32_data().unwrap(), &[1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_sub_inplace_non_contiguous_self() {
        let base = tensor_f32(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], vec![3, 4]);
        let slice_args1 = vec![SliceArg::Slice(0, 2, 1), SliceArg::Slice(0, 2, 1)];
        let mut a_slice_view = base.slice(slice_args1.as_slice()).unwrap(); 
        let b = tensor_f32(vec![10., 10., 10., 10.], vec![2, 2]);
        
        let result_slice_op = a_slice_view.sub_(&b);
        assert!(matches!(result_slice_op, Err(NeuraRustError::BufferSharedError { .. })),
                "Expected BufferSharedError for sub_ on a view with shared Arc<Buffer>, got {:?}", result_slice_op);
        assert_eq!(base.get_f32_data().unwrap(), &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

        let c_orig = tensor_f32(vec![1., 2., 3., 4.], vec![2, 2]);
        let slice_args2 = vec![SliceArg::Slice(0,1,1), SliceArg::Slice(0,2,1)];
        let mut c_view = c_orig.slice(slice_args2.as_slice()).unwrap(); 
        drop(c_orig); 
        
        let b_for_c = tensor_f32(vec![0.5, 0.5], vec![1,2]);
        let result_c_op = c_view.sub_(&b_for_c);
        
        if result_c_op.is_ok() {
            assert_eq!(c_view.get_f32_data().unwrap(), &[0.5, 1.5]);
            eprintln!("Note: sub_ on dropped-original-view succeeded for c_view.");
        } else {
            assert!(matches!(result_c_op, Err(NeuraRustError::BufferSharedError { .. })),
                    "Expected BufferSharedError for c_view if buffer was still shared, got {:?}", result_c_op);
            eprintln!("Note: sub_ on dropped-original-view failed with BufferSharedError for c_view as expected/possible.");
        }
    }
} 