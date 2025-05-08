//! Tests for the RMSprop optimizer implementation.
//!
//! This module ensures the correct behavior of `RmsPropOptimizer`,
//! including its instantiation, hyperparameter validation, optimization steps
//! with various features (weight decay, momentum, centered), and state management.

use crate::optim::Optimizer;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use std::sync::{Arc, RwLock};

use super::RmsPropOptimizer;

/// Helper function to create a test parameter wrapped in `Arc<RwLock<Parameter>>`.
///
/// This simplifies the setup for tests requiring `Parameter` instances.
/// The tensor created within the parameter will have `requires_grad` set to `false` by default
/// by `Tensor::new`, but `Parameter::new` then sets it to `true`.
///
/// # Arguments
/// * `data`: A `Vec<f32>` containing the tensor data.
/// * `shape`: A `Vec<usize>` defining the tensor dimensions.
/// * `name`: An optional `String` for the parameter's name.
///
/// # Returns
/// An `Arc<RwLock<Parameter>>` suitable for use in optimizer tests.
fn create_test_param(
    data: Vec<f32>,
    shape: Vec<usize>,
    name: Option<String>,
) -> Arc<RwLock<Parameter>> {
    let tensor = Tensor::new(data, shape).unwrap();
    let param = Parameter::new(tensor, name);
    Arc::new(RwLock::new(param))
}

#[cfg(test)]
/// Contains all unit tests for the `RmsPropOptimizer`.
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::NeuraRustError;
    use crate::nn::parameter::Parameter;
    use crate::optim::optimizer_state::OptimizerState;

    /// Tests the successful creation of an `RmsPropOptimizer` with valid hyperparameters.
    #[test]
    fn test_rmsprop_optimizer_new() -> Result<(), NeuraRustError> {
        let param = create_test_param(vec![1.0, 2.0, 3.0], vec![3], Some("param1".to_string()));
        let params = vec![param];
        let _optimizer = RmsPropOptimizer::new(params, 0.01, 0.99, 1e-8, 0.0, 0.0, false)?;
        Ok(())
    }

    /// Tests that `RmsPropOptimizer::new` returns an error for various invalid hyperparameters.
    /// This includes negative learning rate, alpha outside [0,1], negative epsilon,
    /// negative weight decay, and negative momentum.
    #[test]
    fn test_rmsprop_invalid_hyperparams() {
        let param = create_test_param(vec![1.0, 2.0, 3.0], vec![3], None);
        let params = vec![param.clone()];

        assert!(RmsPropOptimizer::new(params.clone(), -0.01, 0.99, 1e-8, 0.0, 0.0, false).is_err());
        assert!(RmsPropOptimizer::new(params.clone(), 0.01, -0.99, 1e-8, 0.0, 0.0, false).is_err());
        assert!(RmsPropOptimizer::new(params.clone(), 0.01, 1.1, 1e-8, 0.0, 0.0, false).is_err());
        assert!(RmsPropOptimizer::new(params.clone(), 0.01, 0.99, -1e-8, 0.0, 0.0, false).is_err());
        assert!(RmsPropOptimizer::new(params.clone(), 0.01, 0.99, 1e-8, -0.1, 0.0, false).is_err());
        assert!(RmsPropOptimizer::new(params, 0.01, 0.99, 1e-8, 0.0, -0.1, false).is_err());
    }

    /// Tests a basic optimization step of `RmsPropOptimizer`.
    /// It verifies that the parameter's data is updated after a step.
    #[test]
    fn test_rmsprop_basic_step() -> Result<(), NeuraRustError> {
        let initial_data = vec![1.0, 2.0, 3.0];
        let param_arc_rwlock = create_test_param(initial_data.clone(), vec![3], Some("param_step".to_string()));
        let params = vec![param_arc_rwlock.clone()];

        let mut optimizer = RmsPropOptimizer::new(params, 0.1, 0.9, 1e-8, 0.0, 0.0, false)?;

        {
            let p_lock = param_arc_rwlock.try_write().expect("Failed to lock param for grad");
            let grad_data = vec![0.1, 0.2, 0.3];
            let grad_tensor = Tensor::new(grad_data, vec![3])?;
            p_lock.tensor.write_data().grad = Some(grad_tensor);
        }

        optimizer.step()?;

        let p_lock_after = param_arc_rwlock.try_read().expect("Failed to lock param after step");
        let data_after_step = p_lock_after.tensor().get_f32_data()?;

        for i in 0..initial_data.len() {
            assert_ne!(data_after_step[i], initial_data[i], "Parameter at index {} was not updated.", i);
        }
        Ok(())
    }

    /// Tests the weight decay functionality of `RmsPropOptimizer`.
    /// With zero gradient, parameters should decrease if weight decay is positive.
    #[test]
    fn test_rmsprop_step_with_weight_decay() -> Result<(), NeuraRustError> {
        let initial_value = 1.0;
        let param_data = vec![initial_value; 3];
        let param = create_test_param(param_data.clone(), vec![3], Some("param_wd".to_string()));
        param.write().unwrap().set_requires_grad(true)?;
        let params = vec![param.clone()];

        let learning_rate = 0.1;
        let weight_decay = 0.1;
        let mut optimizer = RmsPropOptimizer::new(params, learning_rate, 0.9, 1e-8, weight_decay, 0.0, false)?;

        {
            let p_lock = param.try_write().expect("Failed to lock param for grad");
            let grad_tensor = Tensor::new(vec![0.0, 0.0, 0.0], vec![3])?;
            p_lock.tensor.write_data().grad = Some(grad_tensor);
        }

        optimizer.step()?;

        let p_lock_after = param.try_read().expect("Failed to lock param after step");
        let data_after_step = p_lock_after.tensor().get_f32_data()?;

        for val_after in data_after_step {
            assert!(val_after < initial_value, "Parameter should decrease with weight decay and zero grad. Got {}, expected < {}", val_after, initial_value);
        }
        Ok(())
    }

    /// Tests the momentum functionality of `RmsPropOptimizer`.
    /// It verifies that the update in the second step is larger than the first
    /// when applying the same gradient, due to momentum accumulation.
    #[test]
    fn test_rmsprop_step_with_momentum() -> Result<(), NeuraRustError> {
        let initial_value = 1.0;
        let param_data = vec![initial_value; 1];
        let param = create_test_param(param_data.clone(), vec![1], Some("param_momentum".to_string()));
        param.write().unwrap().set_requires_grad(true)?;
        let params = vec![param.clone()];

        let learning_rate = 0.1;
        let momentum_param = 0.9;
        let mut optimizer = RmsPropOptimizer::new(params, learning_rate, 0.9, 1e-8, 0.0, momentum_param, false)?;

        let grad_val = 0.1;
        let grad_tensor1 = Tensor::new(vec![grad_val], vec![1])?;

        // First step
        {
            let p_lock = param.try_write().expect("Failed to lock param for grad step 1");
            p_lock.tensor.write_data().grad = Some(grad_tensor1.clone());
        }
        optimizer.step()?;
        let val_after_step1 = param.try_read().unwrap().tensor().get_f32_data()?[0];
        let update_step1 = initial_value - val_after_step1;
        assert_ne!(update_step1, 0.0, "Parameter should be updated in step 1");

        // Second step with the same gradient
        {
            let p_lock = param.try_write().expect("Failed to lock param for grad step 2");
            let grad_tensor2 = Tensor::new(vec![grad_val], vec![1])?;
            p_lock.tensor.write_data().grad = Some(grad_tensor2);
        }
        optimizer.step()?;
        let val_after_step2 = param.try_read().unwrap().tensor().get_f32_data()?[0];
        let update_step2 = val_after_step1 - val_after_step2;

        assert!(update_step2.abs() > update_step1.abs(), 
                "Update in step 2 (abs value: {}) should be greater than in step 1 (abs value: {}) due to momentum.", 
                update_step2.abs(), update_step1.abs());

        Ok(())
    }

    /// Tests the `centered` variant of `RmsPropOptimizer`.
    /// It compares the parameter updates of a centered RMSprop optimizer against a
    /// non-centered one, expecting different results after a few steps.
    #[test]
    fn test_rmsprop_step_centered() -> Result<(), NeuraRustError> {
        let initial_value = 1.0;
        let param_data = vec![initial_value; 1];
        let param_non_centered = create_test_param(param_data.clone(), vec![1], Some("p_non_centered".to_string()));
        param_non_centered.write().unwrap().set_requires_grad(true)?;
        let param_centered = create_test_param(param_data.clone(), vec![1], Some("p_centered".to_string()));
        param_centered.write().unwrap().set_requires_grad(true)?;

        let params_non_centered = vec![param_non_centered.clone()];
        let params_centered = vec![param_centered.clone()];

        let lr = 0.01;
        let alpha = 0.9;
        let eps = 1e-8;
        
        let mut optimizer_non_centered = RmsPropOptimizer::new(params_non_centered, lr, alpha, eps, 0.0, 0.0, false)?;
        let mut optimizer_centered = RmsPropOptimizer::new(params_centered, lr, alpha, eps, 0.0, 0.0, true)?;

        let grad_val = 0.1;
        let grad_tensor = Tensor::new(vec![grad_val], vec![1])?;

        // Step 1
        {
            let p_nc_lock = param_non_centered.try_write().unwrap();
            p_nc_lock.tensor.write_data().grad = Some(grad_tensor.clone());
            
            let p_c_lock = param_centered.try_write().unwrap();
            p_c_lock.tensor.write_data().grad = Some(grad_tensor.clone());
        }
        optimizer_non_centered.step()?;
        optimizer_centered.step()?;

        // Step 2 (to allow grad_avg in centered to become non-zero)
         {
            let p_nc_lock = param_non_centered.try_write().unwrap();
            p_nc_lock.tensor.write_data().grad = Some(grad_tensor.clone());
            
            let p_c_lock = param_centered.try_write().unwrap();
            p_c_lock.tensor.write_data().grad = Some(grad_tensor.clone());
        }
        optimizer_non_centered.step()?;
        optimizer_centered.step()?;

        let val_non_centered = param_non_centered.try_read().unwrap().tensor().get_f32_data()?[0];
        let val_centered = param_centered.try_read().unwrap().tensor().get_f32_data()?[0];

        assert_ne!(val_non_centered, val_centered, 
                   "Centered ({}) and non-centered ({}) RMSprop should yield different results after a few steps.",
                   val_centered, val_non_centered);

        Ok(())
    }

    /// Tests the state saving (`state_dict`) and loading (`load_state_dict`)
    /// functionality of `RmsPropOptimizer`.
    ///
    /// This test ensures that:
    /// 1. An optimizer's state (iterations, hyperparameters, and internal buffers like
    ///    `square_avg`, `grad_avg`, `momentum_buffer`) can be captured.
    /// 2. This state can be loaded into a new optimizer instance.
    /// 3. The new optimizer, after loading the state and having its parameter data
    ///    set to match the first optimizer's parameter data at the point of state capture,
    ///    produces the same parameter update as the original optimizer would if it continued.
    /// 4. Internal buffers are deeply cloned and correctly restored.
    #[test]
    fn test_rmsprop_state_dict() -> Result<(), NeuraRustError> {
        // 1. Create optimizer and run some steps
        let param1 = create_test_param(vec![3.0f32], vec![], Some("p1_state".to_string()));
        param1.write().unwrap().set_requires_grad(true)?;

        let params1 = vec![param1.clone()];
        let opt1_lr = 0.01;
        let opt1_alpha = 0.9;
        let opt1_eps = 1e-7;
        let opt1_weight_decay = 0.05;
        let opt1_momentum = 0.8;
        let opt1_centered = true;

        let mut optimizer1 = RmsPropOptimizer::new(
            params1, 
            opt1_lr, 
            opt1_alpha, 
            opt1_eps, 
            opt1_weight_decay, 
            opt1_momentum, 
            opt1_centered
        )?;

        for i in 0..3 {
            let p1_lock = param1.try_write().unwrap();
            let grad_val = (i + 1) as f32 * 0.5;
            let grad_for_p1 = Tensor::new(vec![grad_val], vec![])?;
            p1_lock.tensor.write_data().grad = Some(grad_for_p1);
            drop(p1_lock);
            optimizer1.step()?;
        }

        let state_dict_opt1 = optimizer1.state_dict()?;

        if let OptimizerState::RmsProp { iterations, param_states, .. } = &state_dict_opt1 {
            assert_eq!(*iterations, 3, "Iterations after 3 steps on optimizer1");
            assert_eq!(param_states.len(), 1, "Param states should contain one entry for param1");
            assert!(param_states.contains_key("p1_state"));
            if let Some(p_state) = param_states.get("p1_state") {
                assert!(p_state.square_avg.numel() > 0); 
            }
        } else {
            panic!("Expected RmsProp state from optimizer1");
        }

        let param1_current_data_vec = param1.read().unwrap().tensor.get_f32_data()?;
        
        let param2_tensor_intermediate = Tensor::new(param1_current_data_vec.clone(), vec![])?;
        param2_tensor_intermediate.set_requires_grad(true)?;
        let param2_tensor = param2_tensor_intermediate;

        let param2 = Arc::new(RwLock::new(
            Parameter::new(param2_tensor, Some("p1_state".to_string()))
        ));

        let mut optimizer2 = RmsPropOptimizer::new(
            vec![param2.clone()],
            0.001,
            0.5,
            1e-5,
            0.5,
            0.5,
            false,
        )?;

        optimizer2.load_state_dict(&state_dict_opt1)?;

        let state_dict_opt2_after_load = optimizer2.state_dict()?;

        if let (OptimizerState::RmsProp { iterations: iter1, param_states: ps1, lr: lr1, alpha: al1, eps: ep1, weight_decay: wd1, momentum: mo1, centered: ce1 },
                OptimizerState::RmsProp { iterations: iter2, param_states: ps2, lr: lr2, alpha: al2, eps: ep2, weight_decay: wd2, momentum: mo2, centered: ce2 })
                = (&state_dict_opt1, &state_dict_opt2_after_load) {
            assert_eq!(iter1, iter2, "Iterations should match after load");
            assert_eq!(lr1, lr2, "LR should match");
            assert_eq!(al1, al2, "Alpha should match");
            assert_eq!(ep1, ep2, "Epsilon should match");
            assert_eq!(wd1, wd2, "Weight decay should match");
            assert_eq!(mo1, mo2, "Momentum should match");
            assert_eq!(ce1, ce2, "Centered flag should match");
            assert_eq!(ps1.len(), ps2.len(), "Number of param states should match");

            let p_state1 = ps1.get("p1_state").expect("p1_state missing in opt1");
            let p_state2 = ps2.get("p1_state").expect("p1_state missing in opt2");

            let sq_avg1_data = p_state1.square_avg.get_f32_data()?;
            let sq_avg2_data = p_state2.square_avg.get_f32_data()?;
            assert_eq!(sq_avg1_data.len(), sq_avg2_data.len(), "Square avg length mismatch");
            for i in 0..sq_avg1_data.len() {
                assert!((sq_avg1_data[i] - sq_avg2_data[i]).abs() < 1e-7, "Square avg data mismatch at index {}", i);
            }

            if let (Some(ga1_tensor), Some(ga2_tensor)) = (&p_state1.grad_avg, &p_state2.grad_avg) {
                let ga1_data = ga1_tensor.get_f32_data()?;
                let ga2_data = ga2_tensor.get_f32_data()?;
                assert_eq!(ga1_data.len(), ga2_data.len(), "Grad avg length mismatch");
                for i in 0..ga1_data.len() {
                    assert!((ga1_data[i] - ga2_data[i]).abs() < 1e-7, "Grad avg data mismatch at index {}", i);
                }
            } else if p_state1.grad_avg.is_some() != p_state2.grad_avg.is_some() {
                panic!("Grad avg presence mismatch. Opt1: {:?}, Opt2: {:?}", p_state1.grad_avg.is_some(), p_state2.grad_avg.is_some());
            }

            if let (Some(mb1_tensor), Some(mb2_tensor)) = (&p_state1.momentum_buffer, &p_state2.momentum_buffer) {
                 let mb1_data = mb1_tensor.get_f32_data()?;
                 let mb2_data = mb2_tensor.get_f32_data()?;
                 assert_eq!(mb1_data.len(), mb2_data.len(), "Momentum buffer length mismatch");
                 for i in 0..mb1_data.len() {
                    assert!((mb1_data[i] - mb2_data[i]).abs() < 1e-7, "Momentum buffer data mismatch at index {}", i);
                 }
            } else if p_state1.momentum_buffer.is_some() != p_state2.momentum_buffer.is_some() {
                panic!("Momentum buffer presence mismatch. Opt1: {:?}, Opt2: {:?}", p_state1.momentum_buffer.is_some(), p_state2.momentum_buffer.is_some());
            }
        } else {
            panic!("Expected RmsProp state from both optimizers after load");
        }

        let final_grad_val = 0.75;
        let grad_tensor = Tensor::new(vec![final_grad_val], vec![])?;

        {
            let p1_lock = param1.try_write().unwrap();
            p1_lock.tensor.write_data().grad = Some(grad_tensor.clone());
        }
        optimizer1.step()?;

        {
            let p2_lock = param2.try_write().unwrap();
            p2_lock.tensor.write_data().grad = Some(grad_tensor.clone());
        }
        optimizer2.step()?;

        let p1_final_val = param1.read().unwrap().tensor.item_f32()?;
        let p2_final_val = param2.read().unwrap().tensor.item_f32()?;

        let tolerance = 1e-6; 
        assert!(
            (p1_final_val - p2_final_val).abs() < tolerance,
            "Parameters differ after loading state dict and stepping. Opt1: {}, Opt2: {}",
            p1_final_val,
            p2_final_val
        );

        if let OptimizerState::RmsProp { iterations, .. } = optimizer1.state_dict()? {
            assert_eq!(iterations, 4, "Optimizer1 iterations after final step");
        }
        if let OptimizerState::RmsProp { iterations, .. } = optimizer2.state_dict()? {
            assert_eq!(iterations, 4, "Optimizer2 iterations after final step (should be 3 from load + 1 step)");
        }
        
        Ok(())
    }

    // TODO: Add more tests:
    // - Test with multiple parameters (named and unnamed)
    // - Test with no_grad parameters
    // - Test clearing gradients with zero_grad()
    // - Test step behavior over multiple iterations (convergence, stability)
    // - Test state_dict and load_state_dict
} 