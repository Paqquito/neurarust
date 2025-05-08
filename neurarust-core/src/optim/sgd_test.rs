#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use crate::{
        tensor::Tensor,
        nn::parameter::Parameter,
        optim::sgd::SgdOptimizer,
        optim::optimizer_trait::Optimizer,
        optim::optimizer_state::OptimizerState,
        error::NeuraRustError,
    };

    // Helper to create a simple parameter
    fn create_param(data: Vec<f32>, shape: Vec<usize>) -> Result<Arc<Mutex<Parameter>>, NeuraRustError> {
        let tensor = Tensor::new(data, shape)?; // Parameter::new will handle requires_grad
        Ok(Arc::new(Mutex::new(Parameter::new(tensor, None))))
    }

    fn assert_vec_f32_eq(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ: A has {}, B has {}", a.len(), b.len());
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            // Using if condition for custom panic message with index
            if (val_a - val_b).abs() > epsilon {
                panic!(
                    "Mismatch at index {}: left = {}, right = {}, diff = {}, epsilon = {}",
                    i, val_a, val_b, (val_a - val_b).abs(), epsilon
                );
            }
            // Or simply use the macro if direct message formatting is not the issue
            // assert_relative_eq!(val_a, val_b, epsilon = epsilon as f64);
        }
    }

    #[test]
    fn test_sgd_basic_step() -> Result<(), NeuraRustError> {
        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let param_shape = vec![2, 2];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4];
        let lr = 0.1;

        let param_arc = create_param(param_data.clone(), param_shape.clone())?;
        
        {
            let param_locked = param_arc.lock().unwrap();
            let grad_tensor = Tensor::new(grad_data.clone(), param_shape.clone())?;
            param_locked.acc_grad(grad_tensor)?; // Use acc_grad to set initial gradient
        }

        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, 0.0, 0.0, false);

        optimizer.step()?;

        let expected_data: Vec<f32> = param_data
            .iter()
            .enumerate()
            .map(|(i, p)| p - lr * grad_data[i])
            .collect();
        
        let param_locked_after = param_arc.lock().unwrap();
        let updated_data = param_locked_after.get_f32_data()?;

        assert_vec_f32_eq(&updated_data, &expected_data, 1e-6);

        Ok(())
    }

    #[test]
    fn test_sgd_weight_decay() -> Result<(), NeuraRustError> {
        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let param_shape = vec![2, 2];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4];
        let lr = 0.1;
        let weight_decay = 0.01;

        let param_arc = create_param(param_data.clone(), param_shape.clone())?;
        
        {
            let param_locked = param_arc.lock().unwrap();
            let grad_tensor = Tensor::new(grad_data.clone(), param_shape.clone())?;
            param_locked.acc_grad(grad_tensor)?;
        }

        // SgdOptimizer::new(params, lr, momentum, weight_decay, nesterov)
        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, 0.0, weight_decay, false);

        optimizer.step()?;

        // Expected: p_new = p_old - lr * (grad + weight_decay * p_old)
        let expected_data: Vec<f32> = param_data
            .iter()
            .zip(grad_data.iter())
            .map(|(p_old, g)| {
                let effective_grad = g + weight_decay * p_old;
                p_old - lr * effective_grad
            })
            .collect();
        
        let param_locked_after = param_arc.lock().unwrap();
        let updated_data = param_locked_after.get_f32_data()?;

        assert_vec_f32_eq(&updated_data, &expected_data, 1e-6);

        Ok(())
    }

    #[test]
    fn test_sgd_momentum() -> Result<(), NeuraRustError> {
        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let param_shape = vec![2, 2];
        let grad1_data = vec![0.1, 0.2, 0.3, 0.4];
        let grad2_data = vec![0.2, 0.1, 0.4, 0.3]; // Different gradient for step 2
        let lr = 0.1;
        let momentum = 0.9;

        let param_arc = create_param(param_data.clone(), param_shape.clone())?;
        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, momentum, 0.0, false);

        // --- Step 1 --- 
        let mut buf1 = vec![0.0f32; grad1_data.len()]; // Initial buffer is zeros
        let expected_param1: Vec<f32>;
        {
            let param_locked = param_arc.lock().unwrap();
            let grad_tensor = Tensor::new(grad1_data.clone(), param_shape.clone())?;
            param_locked.acc_grad(grad_tensor)?; // Set grad for step 1
            
            // Calculate expected update for step 1
            expected_param1 = param_data
                .iter()
                .zip(grad1_data.iter())
                .enumerate()
                .map(|(i, (p_old, g))| {
                    buf1[i] = momentum * buf1[i] + g; // Update buffer: buf = 0.9*0 + g = g
                    p_old - lr * buf1[i] // Update param: p = p - lr*g
                })
                .collect();
        }
        
        optimizer.step()?;

        // Check param after step 1
        {
            let param_locked_after1 = param_arc.lock().unwrap();
            let updated_data1 = param_locked_after1.get_f32_data()?;
            assert_vec_f32_eq(&updated_data1, &expected_param1, 1e-6);
            // Check buffer state (Optional but good for verification)
            // This requires accessing optimizer state, might need state_dict() call
        }
        
        // --- Step 2 --- 
        let mut buf2 = buf1.clone(); // Buffer from step 1
        let expected_param2: Vec<f32>;
        {
            let param_locked = param_arc.lock().unwrap();
            param_locked.clear_grad(); // Clear old gradient
            let grad_tensor = Tensor::new(grad2_data.clone(), param_shape.clone())?;
            param_locked.acc_grad(grad_tensor)?; // Set grad for step 2
            
            let current_param_data = param_locked.get_f32_data()?; // Params after step 1
            
            // Calculate expected update for step 2
            expected_param2 = current_param_data
                .iter()
                .zip(grad2_data.iter())
                .enumerate()
                .map(|(i, (p_current, g))| {
                    buf2[i] = momentum * buf1[i] + g; // Update buffer: buf = 0.9*buf1 + g2
                    p_current - lr * buf2[i] // Update param: p = p_current - lr*buf2
                })
                .collect();
        }
        
        optimizer.step()?;

        // Check param after step 2
        {
            let param_locked_after2 = param_arc.lock().unwrap();
            let updated_data2 = param_locked_after2.get_f32_data()?;
            assert_vec_f32_eq(&updated_data2, &expected_param2, 1e-6);
        }

        Ok(())
    }

    // TODO: Potentially add tests for multiple parameter groups with different settings.
    // TODO: Add tests for CUDA tensors if/when SGD supports them.
    // TODO: Add tests for sparse gradients if/when SGD supports them.
    // TODO: Ensure all optimizer variants (momentum, nesterov, weight_decay) are tested with state_dict.

    #[test]
    fn test_sgd_nesterov_momentum() {
        let param_data_vec = vec![1.0f32];
        let tensor = Tensor::new(param_data_vec.clone(), vec![1]).unwrap();
        let param_object = Parameter::new(tensor, None); 
        let param_arc = Arc::new(Mutex::new(param_object));

        let params_vec = vec![Arc::clone(&param_arc)];
        let lr_val = 0.1;
        let momentum_val = 0.9;
        let group_weight_decay_val = 0.0;
        let nesterov_val = true;

        let mut optimizer = SgdOptimizer::new(
            params_vec,
            lr_val,
            momentum_val,
            group_weight_decay_val,
            nesterov_val,
        );

        // Step 1
        let grad1_data = vec![0.1f32];
        {
            let param_locked = param_arc.lock().unwrap();
            param_locked.acc_grad(Tensor::new(grad1_data.clone(), vec![1]).unwrap()).unwrap();
        }
        optimizer.step().unwrap(); 

        let p1_expected = 1.0 - 0.1 * (0.1 + 0.9 * (0.9 * 0.0 + 0.1)); 
        let p1_actual = param_arc.lock().unwrap().get_f32_data().unwrap()[0];
        assert!((p1_actual - p1_expected).abs() < 1e-6, "Step 1: Expected {}, got {}", p1_expected, p1_actual);

        // Check buffer after step 1
        let state_after_step1 = optimizer.state_dict().unwrap();
        if let OptimizerState::Sgd { momentum_buffers } = state_after_step1 {
            let param_id = Arc::as_ptr(&param_arc) as usize; 
            let buffer_step1 = momentum_buffers.get(&param_id).unwrap().get_f32_data().unwrap()[0];
            assert!((buffer_step1 - 0.1).abs() < 1e-6, "Buffer after step 1: Expected 0.1, got {}", buffer_step1);
        } else {
            panic!("Optimizer state after step 1 is not Sgd or state_dict() failed");
        }

        // Step 2
        let grad2_data = vec![0.2f32];
        {
            let param_locked = param_arc.lock().unwrap();
            param_locked.clear_grad(); 
            param_locked.acc_grad(Tensor::new(grad2_data.clone(), vec![1]).unwrap()).unwrap();
        }
        optimizer.step().unwrap(); 

        let p2_expected = p1_expected - 0.1 * (0.2 + 0.9 * (0.9 * 0.1 + 0.2));
        let p2_actual = param_arc.lock().unwrap().get_f32_data().unwrap()[0];
        assert!((p2_actual - p2_expected).abs() < 1e-6, "Step 2: Expected {}, got {}", p2_expected, p2_actual);
        
        // Check buffer after step 2
        let state_after_step2 = optimizer.state_dict().unwrap();
        if let OptimizerState::Sgd { momentum_buffers } = state_after_step2 {
            let param_id = Arc::as_ptr(&param_arc) as usize; 
            let buffer_step2 = momentum_buffers.get(&param_id).unwrap().get_f32_data().unwrap()[0];
            assert!((buffer_step2 - 0.29).abs() < 1e-6, "Buffer after step 2: Expected 0.29, got {}", buffer_step2);
        } else {
            panic!("Optimizer state after step 2 is not Sgd or state_dict() failed");
        }
    }

    #[test]
    fn test_sgd_nesterov_momentum_with_weight_decay() {
        let param_data_vec = vec![1.0f32];
        let tensor = Tensor::new(param_data_vec.clone(), vec![1]).unwrap();
        let param_object = Parameter::new(tensor, None); 
        let param_arc = Arc::new(Mutex::new(param_object));

        let params_vec = vec![Arc::clone(&param_arc)];
        let lr_val = 0.1;
        let momentum_val = 0.9;
        let wd_val = 0.01;
        let nesterov_val = true;

        let mut optimizer = SgdOptimizer::new(
            params_vec,
            lr_val,
            momentum_val,
            wd_val, 
            nesterov_val,
        );

        // --- Step 1 ---
        let grad1_data = vec![0.1f32];
        let p_initial = param_data_vec[0];
        {
            let param_locked = param_arc.lock().unwrap();
            param_locked.acc_grad(Tensor::new(grad1_data.clone(), vec![1]).unwrap()).unwrap();
        }
        optimizer.step().unwrap(); 

        // Calculations for step 1
        // d_p1 = grad1 + wd * p_initial = 0.1 + 0.01 * 1.0 = 0.11
        let d_p1 = grad1_data[0] + wd_val * p_initial;
        // buf1 = momentum * buf_prev (0.0) + d_p1 = 0.9 * 0.0 + 0.11 = 0.11
        let buf1_expected = momentum_val * 0.0 + d_p1;
        // update_val1 = d_p1 + momentum * buf1 = 0.11 + 0.9 * 0.11 = 0.11 + 0.099 = 0.209
        let update_val1 = d_p1 + momentum_val * buf1_expected;
        // p1 = p_initial - lr * update_val1 = 1.0 - 0.1 * 0.209 = 1.0 - 0.0209 = 0.9791
        let p1_expected = p_initial - lr_val * update_val1;

        let p1_actual = param_arc.lock().unwrap().get_f32_data().unwrap()[0];
        assert!((p1_actual - p1_expected).abs() < 1e-6, "Step 1 Param: Expected {}, got {}", p1_expected, p1_actual);

        let state_after_step1 = optimizer.state_dict().unwrap();
        if let OptimizerState::Sgd { momentum_buffers } = state_after_step1 {
            let param_id = Arc::as_ptr(&param_arc) as usize; 
            let buffer_s1_actual = momentum_buffers.get(&param_id).unwrap().get_f32_data().unwrap()[0];
            assert!((buffer_s1_actual - buf1_expected).abs() < 1e-6, "Step 1 Buffer: Expected {}, got {}", buf1_expected, buffer_s1_actual);
        } else {
            panic!("Optimizer state after step 1 is not Sgd");
        }

        // --- Step 2 ---
        let grad2_data = vec![0.2f32];
        let p_after_step1 = p1_actual; // Use actual p1 for next step's p_old
        {
            let param_locked = param_arc.lock().unwrap();
            param_locked.clear_grad(); 
            param_locked.acc_grad(Tensor::new(grad2_data.clone(), vec![1]).unwrap()).unwrap();
        }
        optimizer.step().unwrap(); 

        // Calculations for step 2
        // d_p2 = grad2 + wd * p_after_step1 = 0.2 + 0.01 * 0.9791 = 0.2 + 0.009791 = 0.209791
        let d_p2 = grad2_data[0] + wd_val * p_after_step1;
        // buf2 = momentum * buf1_expected + d_p2 = 0.9 * 0.11 + 0.209791 = 0.099 + 0.209791 = 0.308791
        let buf2_expected = momentum_val * buf1_expected + d_p2;
        // update_val2 = d_p2 + momentum * buf2 = 0.209791 + 0.9 * 0.308791 = 0.209791 + 0.2779119 = 0.4877029
        let update_val2 = d_p2 + momentum_val * buf2_expected;
        // p2 = p_after_step1 - lr * update_val2 = 0.9791 - 0.1 * 0.4877029 = 0.9791 - 0.04877029 = 0.93032971
        let p2_expected = p_after_step1 - lr_val * update_val2;

        let p2_actual = param_arc.lock().unwrap().get_f32_data().unwrap()[0];
        assert!((p2_actual - p2_expected).abs() < 1e-6, "Step 2 Param: Expected {}, got {}", p2_expected, p2_actual);

        let state_after_step2 = optimizer.state_dict().unwrap();
        if let OptimizerState::Sgd { momentum_buffers } = state_after_step2 {
            let param_id = Arc::as_ptr(&param_arc) as usize; 
            let buffer_s2_actual = momentum_buffers.get(&param_id).unwrap().get_f32_data().unwrap()[0];
            assert!((buffer_s2_actual - buf2_expected).abs() < 1e-6, "Step 2 Buffer: Expected {}, got {}", buf2_expected, buffer_s2_actual);
        } else {
            panic!("Optimizer state after step 2 is not Sgd");
        }
    }

    #[test]
    fn test_sgd_state_dict() -> Result<(), NeuraRustError> {
        // --- Setup ---
        let param_data1 = vec![1.0, 2.0];
        let param_data2 = vec![3.0, 4.0];
        let grad_data_s1 = vec![0.1, 0.1]; // Grads for step 1
        let grad_data_s2 = vec![0.2, 0.2]; // Grads for step 2
        let lr = 0.1;
        let momentum = 0.9;

        // --- Run Original Optimizer ---
        let param1_orig = create_param(param_data1.clone(), vec![2])?;
        let param2_orig = create_param(param_data2.clone(), vec![2])?;
        let mut optimizer1 = SgdOptimizer::new(
            vec![param1_orig.clone(), param2_orig.clone()], 
            lr, momentum, 0.0, false
        );

        // Step 1
        {
            param1_orig.lock().unwrap().acc_grad(Tensor::new(grad_data_s1.clone(), vec![2])?)?;
            param2_orig.lock().unwrap().acc_grad(Tensor::new(grad_data_s1.clone(), vec![2])?)?;
        }
        optimizer1.step()?;

        // Step 2
        {
            param1_orig.lock().unwrap().clear_grad();
            param2_orig.lock().unwrap().clear_grad();
            param1_orig.lock().unwrap().acc_grad(Tensor::new(grad_data_s2.clone(), vec![2])?)?;
            param2_orig.lock().unwrap().acc_grad(Tensor::new(grad_data_s2.clone(), vec![2])?)?;
        }
        optimizer1.step()?;
        
        let params1_final_orig = param1_orig.lock().unwrap().get_f32_data()?;
        let params2_final_orig = param2_orig.lock().unwrap().get_f32_data()?;
        let state_dict_orig = optimizer1.state_dict()?;


        // --- Create New Optimizer, Load State, and Run ---
        let param1_new = create_param(param_data1.clone(), vec![2])?; // Start with SAME initial data
        let param2_new = create_param(param_data2.clone(), vec![2])?;
        let mut optimizer2 = SgdOptimizer::new(
            vec![param1_new.clone(), param2_new.clone()], 
            lr, momentum, 0.0, false
        );

        // Load state BEFORE any steps
        optimizer2.load_state_dict(&state_dict_orig)?; 

        // Step 1 (New optimizer with loaded state)
        {
            param1_new.lock().unwrap().acc_grad(Tensor::new(grad_data_s1.clone(), vec![2])?)?;
            param2_new.lock().unwrap().acc_grad(Tensor::new(grad_data_s1.clone(), vec![2])?)?;
        }
        optimizer2.step()?;

        // Step 2 (New optimizer with loaded state)
        {
            param1_new.lock().unwrap().clear_grad();
            param2_new.lock().unwrap().clear_grad();
            param1_new.lock().unwrap().acc_grad(Tensor::new(grad_data_s2.clone(), vec![2])?)?;
            param2_new.lock().unwrap().acc_grad(Tensor::new(grad_data_s2.clone(), vec![2])?)?;
        }
        optimizer2.step()?;

        // Check final parameters
        let params1_final_new = param1_new.lock().unwrap().get_f32_data()?;
        let params2_final_new = param2_new.lock().unwrap().get_f32_data()?;

        assert_vec_f32_eq(&params1_final_new, &params1_final_orig, 1e-6);
        assert_vec_f32_eq(&params2_final_new, &params2_final_orig, 1e-6);

        Ok(())
    }
} 