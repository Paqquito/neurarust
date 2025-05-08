// This file will contain tests for the AdamOptimizer.

#[cfg(test)]
mod tests {
    use crate::error::NeuraRustError;
    use crate::nn::parameter::Parameter; 
    use crate::optim::adam::AdamOptimizer; // Adjusted to specific import
    use crate::optim::Optimizer;
    use crate::tensor::Tensor; 
    use std::sync::{Arc, RwLock};

    // Helper to create a named parameter for testing
    fn create_named_param(name: &str, data: Vec<f32>, shape: Vec<usize>) -> Arc<RwLock<Parameter>> {
        let tensor = Tensor::new(data, shape).unwrap();
        let param = Parameter::new(tensor, Some(name.to_string()));
        Arc::new(RwLock::new(param))
    }
    
    // Helper to create an unnamed parameter
    fn create_unnamed_param(data: Vec<f32>, shape: Vec<usize>) -> Arc<RwLock<Parameter>> {
        let tensor = Tensor::new(data, shape).unwrap();
        let param = Parameter::new_unnamed(tensor); 
        Arc::new(RwLock::new(param))
    }

    #[test]
    fn test_adam_optimizer_new() {
        let param1 = create_named_param("p1", vec![1.0, 2.0], vec![2]);
        let params = vec![param1];
        let optimizer = AdamOptimizer::new(params, 0.001, 0.9, 0.999, 1e-8, 0.0, false);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_adam_invalid_lr() {
        let params = vec![create_named_param("p1", vec![1.0], vec![1])];
        let optimizer = AdamOptimizer::new(params, -0.001, 0.9, 0.999, 1e-8, 0.0, false);
        assert!(matches!(optimizer, Err(NeuraRustError::ConfigurationError(_))));
    }


    #[test]
    fn test_adam_basic_step_named_param() -> Result<(), NeuraRustError> {
        let initial_value = 10.0f32;
        let param_data_vec = vec![initial_value; 2];
        let param_shape = vec![2];
        let param1 = create_named_param("param_test", param_data_vec.clone(), param_shape.clone());
        
        let mut optimizer = AdamOptimizer::new(vec![param1.clone()], 0.1, 0.9, 0.999, 1e-8, 0.0, false)?;

        {
            let mut p_locked = param1.write().unwrap();
            let grad_tensor = Tensor::new(vec![1.0, 1.0], vec![2])?;
            p_locked.tensor_mut().clear_grad(); 
            p_locked.tensor_mut().acc_grad(grad_tensor)?; 
        }

        optimizer.step()?;

        let p_locked_after = param1.read().unwrap();
        let tensor_after_step = p_locked_after.tensor(); 
        let data_after_step = tensor_after_step.get_f32_data()?;
        
        assert_ne!(data_after_step[0], initial_value);
        assert_ne!(data_after_step[1], initial_value);
        assert!(data_after_step[0] < initial_value);
        Ok(())
    }
    
    #[test]
    fn test_adam_basic_step_unnamed_param() -> Result<(), NeuraRustError> {
        let initial_value = 5.0f32;
        let param_data_vec = vec![initial_value]; 
        let param_shape = vec![1];
        let param_unnamed = create_unnamed_param(param_data_vec.clone(), param_shape.clone());
        
        let mut optimizer = AdamOptimizer::new(vec![param_unnamed.clone()], 0.01, 0.9, 0.999, 1e-8, 0.0, false)?;

        {
            let mut p_locked = param_unnamed.write().unwrap();
            let grad_tensor = Tensor::new(vec![-2.0], vec![1])?;
            p_locked.tensor_mut().clear_grad();
            p_locked.tensor_mut().acc_grad(grad_tensor)?;
        }

        optimizer.step()?;

        let p_locked_after = param_unnamed.read().unwrap();
        let tensor_after_step = p_locked_after.tensor();
        let data_after_step = tensor_after_step.get_f32_data()?;
        
        assert_ne!(data_after_step[0], initial_value);
        assert!(data_after_step[0] > initial_value);
        
        let ptr_address = Arc::as_ptr(&param_unnamed);
        let temp_id = format!("unnamed_param_at_{:?}", ptr_address);
        assert!(optimizer.state.contains_key(&temp_id));
        Ok(())
    }


    #[test]
    fn test_adam_weight_decay() -> Result<(), NeuraRustError> {
        let initial_value = 1.0f32;
        let lr = 0.1f32;
        let weight_decay = 0.1f32; 
        
        let param = create_named_param("p_wd", vec![initial_value], vec![1]);
        let mut optimizer = AdamOptimizer::new(vec![param.clone()], lr, 0.9, 0.999, 1e-8, weight_decay, false)?;

        {
            let mut p_locked = param.write().unwrap();
            let grad_tensor = Tensor::new(vec![0.0], vec![1])?;
            p_locked.tensor_mut().clear_grad();
            p_locked.tensor_mut().acc_grad(grad_tensor)?;
        }

        optimizer.step()?;

        let p_locked_after = param.read().unwrap();
        let tensor_after_step = p_locked_after.tensor();
        let data_after_step = tensor_after_step.get_f32_data()?;

        let expected_value = initial_value * (1.0 - lr * weight_decay); 
        assert!((data_after_step[0] - expected_value).abs() < 1e-6);
        Ok(())
    }
    
    #[test]
    fn test_adam_no_grad_param() -> Result<(), NeuraRustError> {
        let param_no_grad_val = 7.0;
        let param_no_grad = create_named_param("p_no_grad", vec![param_no_grad_val], vec![1]);

        // Explicitly set requires_grad to false for this parameter and handle Result
        param_no_grad.write().unwrap().set_requires_grad(false)?;

        let mut optimizer = AdamOptimizer::new(vec![param_no_grad.clone()], 0.1, 0.9, 0.999, 1e-8, 0.0, false)?;
        optimizer.step()?;

        let p_locked_after = param_no_grad.read().unwrap();
        let data_after_step = p_locked_after.tensor().get_f32_data()?;
        assert_eq!(data_after_step[0], param_no_grad_val);
        assert!(!optimizer.state.contains_key("p_no_grad"));
        Ok(())
    }

    #[test]
    fn test_zero_grad_clears_gradient() -> Result<(), NeuraRustError> {
        let param = create_named_param("p_zero", vec![1.0], vec![1]);
        {
            let mut p_locked = param.write().unwrap();
            let grad_tensor = Tensor::new(vec![10.0], vec![1])?;
            p_locked.tensor_mut().clear_grad();
            p_locked.tensor_mut().acc_grad(grad_tensor)?;
            assert!(p_locked.grad().is_some()); 
        }

        let mut optimizer = AdamOptimizer::new(vec![param.clone()], 0.1, 0.9, 0.999, 1e-8, 0.0, false)?;
        optimizer.zero_grad();

        let p_locked_after = param.read().unwrap();
        assert!(p_locked_after.grad().is_none()); 
        Ok(())
    }
} 