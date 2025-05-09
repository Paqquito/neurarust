// This file will contain tests for the AdamOptimizer.

#[cfg(test)]
mod tests {
    use crate::error::NeuraRustError;
    use crate::nn::parameter::Parameter; 
    use crate::optim::adam::AdamOptimizer; // Adjusted to specific import
    use crate::optim::Optimizer;
    use crate::tensor::Tensor; 
    use std::sync::{Arc, RwLock};
    use approx::assert_relative_eq;

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
        
        Ok(())
    }


    #[test]
    fn test_adam_weight_decay() -> Result<(), NeuraRustError> {
        let initial_value = 1.0f32;
        let lr = 0.1f32;
        let weight_decay = 0.1f32;
        let beta1 = 0.9f32; // Test utilise 0.9
        let beta2 = 0.999f32; // Test utilise 0.999
        let eps = 1e-8f32;
        
        let param = create_named_param("p_wd", vec![initial_value], vec![1]);
        // Utiliser les betas du test pour la nouvelle instance d'AdamOptimizer
        let mut optimizer = AdamOptimizer::new(vec![param.clone()], lr, beta1, beta2, eps, weight_decay, false)?;

        {
            let mut p_locked = param.write().unwrap();
            let grad_tensor = Tensor::new(vec![0.0], vec![1])?; // Gradient est NUL
            p_locked.tensor_mut().clear_grad();
            p_locked.tensor_mut().acc_grad(grad_tensor)?;
        }

        optimizer.step()?;

        let p_locked_after = param.read().unwrap();
        let tensor_after_step = p_locked_after.tensor();
        let data_after_step = tensor_after_step.get_f32_data()?;

        // Calcul basé sur l'implémentation actuelle
        let grad_original = 0.0f32;
        let grad_eff = grad_original + initial_value * weight_decay; // grad_eff = 0.1

        // Pour le premier pas (optimizer.iterations sera 1 après le premier step)
        let m_t = (1.0 - beta1) * grad_eff; // (1-0.9)*0.1 = 0.01
        let v_t = (1.0 - beta2) * grad_eff.powi(2); // (1-0.999)*(0.1)^2 = 0.001 * 0.01 = 0.00001

        let bias_correction1 = 1.0 - beta1.powi(1);
        let bias_correction2 = 1.0 - beta2.powi(1);

        let m_hat = m_t / bias_correction1; // 0.01 / (1-0.9) = 0.01 / 0.1 = 0.1
        let v_hat = v_t / bias_correction2; // 0.00001 / (1-0.999) = 0.00001 / 0.001 = 0.01

        let update_val = lr * m_hat / (v_hat.sqrt() + eps);
        // update_val = 0.1 * 0.1 / ( (0.01).sqrt() + 1e-8 ) = 0.01 / (0.1 + 1e-8)
        // update_val = 0.01 / 0.10000001 = 0.09999999
        
        let expected_value = initial_value - update_val;
        
        assert_relative_eq!(data_after_step[0], expected_value, epsilon = 1e-7);
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

    #[test]
    fn test_amsgrad_logic() -> Result<(), NeuraRustError> {
        let initial_value = 1.0f32;
        let param_arc = create_named_param("p_amsgrad", vec![initial_value], vec![1]);

        // Optimizer instances (these will maintain their own internal state across steps)
        let mut optim_adam = AdamOptimizer::new(vec![param_arc.clone()], 0.1, 0.9, 0.9, 1e-8, 0.0, false)?;
        let mut optim_amsgrad = AdamOptimizer::new(vec![param_arc.clone()], 0.1, 0.9, 0.9, 1e-8, 0.0, true)?; // amsgrad=true

        // --- Helper to set param value ---
        let set_param_value = |p_arc: &Arc<RwLock<Parameter>>, value: f32| -> Result<(), NeuraRustError> {
            let mut p_guard = p_arc.write().unwrap();
            let mut tensor_guard = p_guard.tensor_mut().write_data();
            let buffer_mut = Arc::make_mut(&mut tensor_guard.buffer);
            let data_slice_mut = buffer_mut.try_get_cpu_f32_mut()?;
            if data_slice_mut.is_empty() {
                return Err(NeuraRustError::ShapeMismatch { operation: "set_param_value".to_string(), expected: "non-empty slice".to_string(), actual: "empty slice".to_string() });
            }
            data_slice_mut[0] = value;
            Ok(())
        };
        
        // --- Helper to set grad ---
        let set_grad_val = |p_arc: &Arc<RwLock<Parameter>>, grad_val: f32| -> Result<(), NeuraRustError> {
            let mut p_locked = p_arc.write().unwrap();
            let grad_tensor = Tensor::new(vec![grad_val], vec![1])?;
            p_locked.tensor_mut().clear_grad();
            p_locked.tensor_mut().acc_grad(grad_tensor)?;
            Ok(())
        };

        // --- Step 1: High gradient ---
        // Adam's Step 1
        set_param_value(&param_arc, initial_value)?;
        set_grad_val(&param_arc, 10.0)?;
        optim_adam.step()?;
        let val_after_adam_s1 = param_arc.read().unwrap().tensor().item_f32()?;

        // AMSGrad's Step 1
        set_param_value(&param_arc, initial_value)?;
        set_grad_val(&param_arc, 10.0)?;
        optim_amsgrad.step()?;
        let val_after_amsgrad_s1 = param_arc.read().unwrap().tensor().item_f32()?;

        assert_relative_eq!(val_after_adam_s1, val_after_amsgrad_s1, epsilon = 1e-6);
        assert!(val_after_adam_s1 < initial_value, "Adam S1 should decrease value. Got: {}, Initial: {}", val_after_adam_s1, initial_value);

        // --- Step 2: Low gradient ---
        // Adam's Step 2 (starts from val_after_adam_s1)
        set_param_value(&param_arc, val_after_adam_s1)?;
        set_grad_val(&param_arc, 0.1)?;
        optim_adam.step()?; // optim_adam uses its state from its step 1
        let val_after_adam_s2 = param_arc.read().unwrap().tensor().item_f32()?;

        // AMSGrad's Step 2 (starts from val_after_amsgrad_s1)
        set_param_value(&param_arc, val_after_amsgrad_s1)?;
        set_grad_val(&param_arc, 0.1)?;
        optim_amsgrad.step()?; // optim_amsgrad uses its state from its step 1 (with v_max set)
        let val_after_amsgrad_s2 = param_arc.read().unwrap().tensor().item_f32()?;

        // --- Verification ---
        let change_adam = val_after_adam_s2 - val_after_adam_s1;
        let change_amsgrad = val_after_amsgrad_s2 - val_after_amsgrad_s1;

        assert!(change_amsgrad > change_adam,
                "AMSGrad change ({}) should be greater (less negative) than Adam change ({}). Adam S1: {}, AMSGrad S1: {}, Adam S2: {}, AMSGrad S2: {}",
                change_amsgrad, change_adam, val_after_adam_s1, val_after_amsgrad_s1, val_after_adam_s2, val_after_amsgrad_s2);

        assert!(val_after_amsgrad_s2 > val_after_adam_s2,
                "AMSGrad final value ({}) should be greater than Adam final value ({}).",
                val_after_amsgrad_s2, val_after_adam_s2);

        Ok(())
    }
} 