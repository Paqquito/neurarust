#[cfg(test)]
mod tests {
    use crate::nn::parameter::Parameter; // Ajustement de l'import pour Parameter
    use crate::tensor::Tensor;
    use crate::types::DType;

    #[test]
    fn test_parameter_creation_requires_grad() {
        let tensor = Tensor::<f32>::zeros(vec![2, 2], DType::F32);
        assert!(!tensor.requires_grad(), "Tensor should not require grad initially");
        let param = Parameter::new(tensor);
        assert!(param.0.requires_grad(), "Parameter's tensor should require grad after creation");
    }

    #[test]
    fn test_parameter_deref() {
        let tensor = Tensor::<f32>::ones(vec![2, 3], DType::F32);
        let param = Parameter::new(tensor);
        // Accès via Deref
        assert_eq!(param.shape(), &vec![2, 3]);
        assert_eq!(param.dtype(), DType::F32);
    }

    #[test]
    fn test_parameter_deref_mut() {
        let tensor = Tensor::<f32>::zeros(vec![1, 1], DType::F32);
        let mut param = Parameter::new(tensor);
        
        param.set_requires_grad(false); 
        assert!(!param.0.requires_grad(), "Should be able to modify tensor through DerefMut");
        
        param.set_requires_grad(true);
        assert!(param.0.requires_grad(), "Should be true after setting back");
    }

    #[test]
    fn test_parameter_clone() {
        let tensor = Tensor::<f32>::full(vec![2, 2], 42.0, DType::F32);
        let param = Parameter::new(tensor);
        let param_cloned = param.clone();

        assert_eq!(param.0.shape(), param_cloned.0.shape());
        assert_eq!(param.0.dtype(), param_cloned.0.dtype());
        assert_eq!(param.0.requires_grad(), param_cloned.0.requires_grad());
        assert!(param_cloned.0.requires_grad(), "Cloned parameter's tensor should require grad");
    }
    
    #[test]
    fn test_parameter_into_inner() {
        let tensor_orig = Tensor::<f32>::full(vec![1, 5], 10.0, DType::F32);
        let tensor_for_comparison = tensor_orig.clone(); 
        
        let param = Parameter::new(tensor_orig);
        assert!(param.requires_grad(), "Parameter's tensor should require grad before into_inner");
        
        let tensor_inner = param.into_inner();
        
        assert_eq!(tensor_inner.shape(), tensor_for_comparison.shape());
        assert_eq!(tensor_inner.dtype(), tensor_for_comparison.dtype());
        assert!(tensor_inner.requires_grad(), "Inner tensor should still require grad after into_inner"); 
        assert_eq!(tensor_inner, tensor_for_comparison, "Inner tensor should be equal to the original tensor (after it was set to requires_grad).");
    }
} 