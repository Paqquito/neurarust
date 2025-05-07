#[cfg(test)]
mod tests {
    use crate::nn::Parameter;
    use crate::tensor::{zeros, ones, full, from_vec_f32, ones_f64}; // Tensor et ones supprimés
    use crate::types::DType;
    use crate::error::NeuraRustError;

    #[test]
    fn test_parameter_creation_f32_zeros() -> Result<(), NeuraRustError> {
        // Utiliser la fonction zeros() qui retourne un Tensor F32 par défaut
        let tensor = zeros(&[2, 2])?;
        let param = Parameter::new_unnamed(tensor);
        assert!(param.requires_grad());
        assert_eq!(param.shape(), &[2, 2]);
        assert_eq!(param.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn test_parameter_creation_f64_ones() -> Result<(), NeuraRustError> {
        // Pour F64, il faudrait une fonction ones_f64. Supposons qu'elle existe et est importée.
        // Si elle n'existe pas, il faudrait créer Tensor::ones(&[2,3], DType::F64)?
        // Pour l'instant, utilisons ones() et changeons le dtype ensuite si c'est le but du test.
        // Alternative: créer Tensor::from_vec_f64(vec![1.0; 6], vec![2,3])? puis Parameter::new()
        // Simplifions: créons avec DType F64 directement si la fonction existe.
        // MAJ: `ones_f64` existe dans `create.rs` et est ré-exporté.
        let tensor = ones_f64(&[2, 3])?; 
        let param = Parameter::new_unnamed(tensor);
        assert!(param.requires_grad());
        assert_eq!(param.shape(), &[2, 3]);
        assert_eq!(param.dtype(), DType::F64);
        Ok(())
    }

    #[test]
    fn test_parameter_set_requires_grad() -> Result<(), NeuraRustError> {
        let tensor = zeros(&[2])?;
        let param = Parameter::new_unnamed(tensor);
        assert!(param.requires_grad(), "Parameter should require grad by default");
        let _ = param.set_requires_grad(false);
        assert!(!param.requires_grad(), "Parameter should not require grad after setting false");
        let _ = param.set_requires_grad(true);
        assert!(param.requires_grad(), "Parameter should require grad after setting true again");
        Ok(())
    }

    #[test]
    fn test_parameter_data_access() -> Result<(), NeuraRustError> {
        // Utiliser full() pour F32
        let tensor = full(&[2, 2], 42.0f32)?;
        let param = Parameter::new_unnamed(tensor);
        // Utiliser get_f32_data() via Deref<Target=Tensor>
        assert_eq!(param.get_f32_data()?, vec![42.0, 42.0, 42.0, 42.0]);
        Ok(())
    }

    #[test]
    fn test_parameter_grad_access_and_manipulation() -> Result<(), NeuraRustError> {
        let tensor_orig = ones(&[2, 2])?;
        let param = Parameter::new_unnamed(tensor_orig);
        param.set_requires_grad(true)?;

        assert!(param.grad().is_none());

        let grad_data_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let grad_tensor = from_vec_f32(grad_data_vec.clone(), vec![1, 5])?;
        
        // Modifier le grad via write_data() sur le Tensor interne
        {
            let mut tensor_data_guard = param.write_data(); // param déréférence vers Tensor, qui a write_data()
            tensor_data_guard.grad = Some(grad_tensor);
        }

        assert!(param.grad().is_some());
        let retrieved_grad = param.grad().unwrap();
        // Utiliser get_f32_data() sur le Tensor grad
        assert_eq!(retrieved_grad.get_f32_data()?, grad_data_vec);
        assert_eq!(retrieved_grad.shape(), &[1, 5]);

        // Remettre le grad à None
        {
            let mut tensor_data_guard = param.write_data();
            tensor_data_guard.grad = None;
        }
        assert!(param.grad().is_none());
        Ok(())
    }

    // TODO: Ajouter des tests pour to_device quand Parameter les implémentera.

    #[test]
    fn test_parameter_creation_with_name() -> Result<(), NeuraRustError> {
        let tensor = zeros(&[2, 2])?;
        let param_name = "test_weight".to_string();
        let param = Parameter::new(tensor, Some(param_name.clone()));
        
        assert!(param.requires_grad());
        assert_eq!(param.shape(), &[2, 2]);
        assert_eq!(param.name(), Some(param_name.as_str()));
        Ok(())
    }

    #[test]
    fn test_parameter_creation_without_name_new() -> Result<(), NeuraRustError> {
        let tensor = zeros(&[2, 2])?;
        let param = Parameter::new(tensor, None);
        
        assert!(param.requires_grad());
        assert_eq!(param.shape(), &[2, 2]);
        assert_eq!(param.name(), None);
        Ok(())
    }

    #[test]
    fn test_parameter_creation_unnamed_constructor() -> Result<(), NeuraRustError> {
        let tensor = ones(&[3, 1])?;
        let param = Parameter::new_unnamed(tensor);
        
        assert!(param.requires_grad());
        assert_eq!(param.shape(), &[3, 1]);
        assert_eq!(param.name(), None);
        Ok(())
    }

    #[test]
    fn test_parameter_name_method() -> Result<(), NeuraRustError> {
        let tensor = zeros(&[1])?;
        let name1 = "bias".to_string();
        let param1 = Parameter::new(tensor.clone(), Some(name1.clone()));
        assert_eq!(param1.name(), Some(name1.as_str()));

        let param2 = Parameter::new_unnamed(tensor);
        assert_eq!(param2.name(), None);
        Ok(())
    }

    #[test]
    fn test_parameter_to_dtype_preserves_name() -> Result<(), NeuraRustError> {
        let tensor_f32 = zeros(&[2, 2])?;
        let param_name = "dtype_param".to_string();
        let mut param = Parameter::new(tensor_f32, Some(param_name.clone()));
        
        param.to_dtype(DType::F64)?;
        
        assert_eq!(param.dtype(), DType::F64);
        assert_eq!(param.name(), Some(param_name.as_str()));
        assert!(param.requires_grad()); // Should still require grad

        let mut param_unnamed = Parameter::new_unnamed(zeros(&[1,1])?);
        param_unnamed.to_dtype(DType::F64)?;
        assert_eq!(param_unnamed.name(), None);
        assert_eq!(param_unnamed.dtype(), DType::F64);

        Ok(())
    }
} 