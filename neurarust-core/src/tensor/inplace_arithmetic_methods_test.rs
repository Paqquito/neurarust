// neurarust-core/src/tensor/inplace_arithmetic_methods_test.rs

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::types::DType; // Pour les vérifications d'erreur
    // use crate::tensor_data::TensorData; // Suppression de l'import inutilisé
    use crate::ops::view::SliceArg; // Importation de SliceArg

    // Helper pour créer un tenseur F32 pour les tests
    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).unwrap()
    }

    // Helper pour créer un tenseur F64 pour les tests
    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new_f64(data, shape).unwrap()
    }

    #[test]
    fn test_add_inplace_simple_correctness() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        a.add_(&b)?;

        let expected_data = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_vector() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![10.0, 20.0, 30.0], vec![3]); // Broadcast [3] to [2,3]
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_row_vector() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f32(vec![10.0, 20.0, 30.0], vec![1, 3]); // Broadcast [1,3] to [2,3]
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_col_vector() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let b = tensor_f32(vec![5.0, 100.0], vec![2, 1]); // Broadcast [2,1] to [2,3]
        
        a.add_(&b)?;

        // ligne0: [1,2,3] + 5   = [6, 7, 8]
        // ligne1: [10,20,30] + 100 = [110, 120, 130]
        let expected_data = vec![6.0, 7.0, 8.0, 110.0, 120.0, 130.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_scalar() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f32(vec![10.0], vec![]); // Scalar
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 12.0, 13.0, 14.0];
        assert_eq!(a.get_f32_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_autograd_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0], vec![1]);
        let b = tensor_f32(vec![1.0], vec![1]);
        a.set_requires_grad(true)?;
        
        let result = a.add_(&b);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::InplaceModificationError { operation, reason: _ } => {
                assert_eq!(operation, "add_");
            }
            _ => panic!("Expected InplaceModificationError"),
        }
        Ok(())
    }

    #[test]
    fn test_add_inplace_dtype_mismatch_error() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0], vec![1]);
        let b_f64 = Tensor::new_f64(vec![1.0], vec![1])?;
        
        let result = a.add_(&b_f64);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::DataTypeMismatch { expected, actual, operation } => {
                assert_eq!(expected, DType::F32);
                assert_eq!(actual, DType::F64);
                assert_eq!(operation, "add_");
            }
            other_err => panic!("Expected DataTypeMismatch, got {:?}", other_err),
        }
        Ok(())
    }
    
    // Test pour BufferSharedError (cas où l'Arc<Buffer> de TensorData est partagé)
    // Ce test est plus complexe car notre `new_view` actuel ne partage pas TensorData.data directement,
    // mais clone l'Arc<Buffer> dans un nouveau TensorData.
    // La vérification de partage d'Arc<Buffer> au niveau de TensorData est donc plus pertinente
    // si deux Tensors pointent vers le *même* Arc<Buffer> et que l'un tente une op en place.
    // Cependant, les opérations en place prennent `&mut self`, donc ce scénario est moins probable.
    // La protection pertinente est sur l`Arc<Vec<T>>` interne au `CpuBuffer`.

    #[test]
    fn test_add_inplace_buffer_shared_error_try_get_mut() -> Result<(), NeuraRustError> {
        let mut a = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tensor_f32(vec![10.0], vec![1]);
        
        let _a_view = a.slice(&[SliceArg::Slice(0, 3, 1)])?; // Crée la vue, partageant le buffer de a

        let result = a.add_(&b);
        
        assert!(result.is_err(), "Expected error for in-place op on shared buffer, got {:?}", result);
        match result.err().unwrap() {
            NeuraRustError::BufferSharedError { operation } => {
                // S'attend à ce que Arc::get_mut sur TensorData.buffer échoue car a_view le partage.
                assert!(operation.contains("add_ (TensorData.buffer is shared)") || operation.contains("try_get_cpu_f32_mut"), "Operation was: {}", operation);
            }
            other_err => panic!("Expected BufferSharedError, got {:?}", other_err),
        }
        
        Ok(())
    }

    // TODO: Test pour le cas non-contiguous de self.
    #[test]
    fn test_add_inplace_non_contiguous_self() -> Result<(), NeuraRustError> {
        let a_orig = tensor_f32(vec![1.,2.,3., 4.,5.,6., 7.,8.,9., 10.,11.,12.], vec![4,3]);
        let b = tensor_f32(vec![10., 20., 30., 40.], vec![4]);

        // Pour que l'opération en place sur a_transposed fonctionne, elle doit être la seule à 
        // "posséder" l'accès mutable au buffer via son TensorData. Ou le buffer ne doit pas être partagé.
        // Ici a_transposed partage son Arc<Buffer> avec a_orig.
        let mut a_transposed = a_orig.transpose(0,1)?;
        
        let result = a_transposed.add_(&b);

        assert!(result.is_err(), "Expected BufferSharedError because a_transposed shares buffer with a_orig");
        match result.err().unwrap() {
            NeuraRustError::BufferSharedError { operation } => {
                // S'attend à ce que Arc::get_mut sur TensorData.buffer de a_transposed échoue
                assert!(operation.contains("add_ (TensorData.buffer is shared)"));
            }
            other_err => panic!("Expected BufferSharedError for non_contiguous test, got {:?}", other_err),
        }

        // Test alternatif : si on veut modifier un tenseur non contigu qui *ne partage pas* son buffer principal de manière problématique
        // On pourrait créer a_transposed, puis s'assurer qu'il est le seul à y accéder (ex: a_orig est drop)
        // ou le cloner en un nouveau buffer s'il est partagé.
        // Pour l'instant, ce test vérifie la protection contre la modification de vues partageant un buffer.

        Ok(())
    }

    // --- Tests F64 ---
    #[test]
    fn test_add_inplace_simple_correctness_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        a.add_(&b)?;

        let expected_data = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![10.0, 20.0, 30.0], vec![3]);
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        assert_eq!(a.shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_row_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tensor_f64(vec![10.0, 20.0, 30.0], vec![1, 3]);
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_rhs_col_vector_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let b = tensor_f64(vec![5.0, 100.0], vec![2, 1]);
        
        a.add_(&b)?;

        let expected_data = vec![6.0, 7.0, 8.0, 110.0, 120.0, 130.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_broadcasting_scalar_f64() -> Result<(), NeuraRustError> {
        let mut a = tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_f64(vec![10.0], vec![]); 
        
        a.add_(&b)?;

        let expected_data = vec![11.0, 12.0, 13.0, 14.0];
        assert_eq!(a.get_f64_data().unwrap(), expected_data);
        Ok(())
    }

    #[test]
    fn test_add_inplace_non_contiguous_self_f64() -> Result<(), NeuraRustError> {
        let a_orig = tensor_f64(vec![1.,2.,3., 4.,5.,6., 7.,8.,9., 10.,11.,12.], vec![4,3]);
        let b = tensor_f64(vec![10., 20., 30., 40.], vec![4]);
        let mut a_transposed = a_orig.transpose(0,1)?;
        
        let result = a_transposed.add_(&b);
        assert!(result.is_err(), "Expected BufferSharedError because a_transposed shares buffer with a_orig (f64)");
        match result.err().unwrap() {
            NeuraRustError::BufferSharedError { operation } => {
                assert!(operation.contains("add_ (TensorData.buffer is shared") || operation.contains("try_get_cpu_f64_mut"), 
                        "Unexpected BufferSharedError operation: {}", operation);
            }
            other_err => panic!("Expected BufferSharedError for non_contiguous_f64 test, got {:?}", other_err),
        }
        Ok(())
    }
    // Test pour BufferSharedError et autograd_error sont génériques par rapport au type de données interne (F32/F64)
    // tant que l'erreur elle-même est testée. Le comportement de partage de buffer ou requires_grad
    // ne dépend pas du type de flottant.

    // Le test `test_add_inplace_buffer_shared_error_try_get_mut` existant (F32) est suffisant pour cette logique.

    // TODO: Test pour le cas non-contiguous de self qui *réussit*.
    // Le test original `test_add_inplace_non_contiguous_self` (et sa version _f64) a été réorienté pour tester BufferSharedError.
    // Un nouveau test pour le succès de l'opération en place sur un non-contiguous (qui ne partage pas son buffer) pourrait être ajouté.
    // Pour cela, il faudrait que `a_transposed` soit le seul propriétaire de son buffer, ou qu'il soit contiguisé avant l'op en place.
    // Exemple: 
    // let a_orig = tensor_f32(vec![1.,2.,3., 4.,5.,6., 7.,8.,9., 10.,11.,12.], vec![4,3]);
    // let mut a_transposed_then_owned = a_orig.transpose(0,1)?.contiguous()?;
    // let b = tensor_f32(vec![10., 20., 30., 40.], vec![4]);
    // a_transposed_then_owned.add_(&b)?;
    // // puis vérifier les données de a_transposed_then_owned ou de son buffer sous-jacent si possible.

} 