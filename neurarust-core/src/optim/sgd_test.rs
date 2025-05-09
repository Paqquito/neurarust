#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};
    use crate::{
        tensor::Tensor,
        nn::parameter::Parameter,
        optim::sgd::SgdOptimizer,
        optim::optimizer_trait::Optimizer,
        optim::optimizer_state::OptimizerState,
        error::NeuraRustError,
    };
    use std::collections::HashMap;

    // Helper pour créer un paramètre mock avec RwLock et un gradient initial
    fn mock_param_rwlock(initial_value: f32, grad_value: f32, name_suffix: &str) -> Arc<RwLock<Parameter>> {
        let tensor = Tensor::new(vec![initial_value], vec![1]).unwrap();
        let param = Parameter::new(tensor.clone(), Some(format!("mock_param_{}", name_suffix)));
        param.tensor.set_requires_grad(true).unwrap();
        
        let grad_tensor = Tensor::new(vec![grad_value], vec![1]).unwrap();
        param.tensor.data.write().unwrap().grad = Some(grad_tensor);
        
        Arc::new(RwLock::new(param))
    }

    #[test]
    fn test_sgd_basic() -> Result<(), NeuraRustError> {
        let lr = 0.1;
        let param_arc = mock_param_rwlock(1.0, 0.1, "basic");
        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, 0.0, 0.0, 0.0, false)?;

        let initial_value = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        assert_eq!(initial_value, 1.0);

        optimizer.step()?;

        let value_after_step = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        // Expected: param = 1.0 - 0.1 * 0.1 = 0.99
        assert!((value_after_step - 0.99).abs() < 1e-6, "Expected 0.99, got {}", value_after_step);
        Ok(())
    }

    #[test]
    fn test_sgd_weight_decay() -> Result<(), NeuraRustError> {
        let lr = 0.1;
        let weight_decay = 0.01;
        let param_arc = mock_param_rwlock(1.0, 0.1, "wd"); 
        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, 0.0, 0.0, weight_decay, false)?;

        optimizer.step()?;

        // Expected: grad_wd = 0.1 + 1.0 * 0.01 = 0.11. param = 1.0 - 0.1 * 0.11 = 0.989
        let value_after_step = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        assert!((value_after_step - 0.989).abs() < 1e-6, "Expected 0.989, got {}", value_after_step);
        Ok(())
    }

    #[test]
    fn test_sgd_momentum() -> Result<(), NeuraRustError> {
        let lr = 0.1;
        let momentum = 0.9;
        let param_arc = mock_param_rwlock(1.0, 0.1, "mom");
        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, momentum, 0.0, 0.0, false)?;

        // Step 1
        optimizer.step()?;
        // buf_1 = 0.1. param_1 = 1.0 - 0.1 * 0.1 = 0.99
        let val_step1 = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        assert!((val_step1 - 0.99).abs() < 1e-6, "Step 1: Expected 0.99, got {}", val_step1);

        // Mettre à jour le gradient du paramètre pour le step suivant
        param_arc.read().unwrap().tensor.data.write().unwrap().grad = Some(Tensor::new(vec![0.1], vec![1]).unwrap());

        // Step 2
        optimizer.step()?;
        // buf_2 = 0.9 * 0.1 (buf_1) + 0.1 (grad) = 0.19.
        // param_2 = 0.99 - 0.1 * 0.19 = 0.971
        let val_step2 = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        assert!((val_step2 - 0.971).abs() < 1e-6, "Step 2: Expected 0.971, got {}", val_step2);
        Ok(())
    }

    #[test]
    fn test_sgd_nesterov_momentum() -> Result<(), NeuraRustError> {
        let lr = 0.1;
        let momentum = 0.9;
        let param_arc = mock_param_rwlock(1.0, 0.1, "nest");
        let mut optimizer = SgdOptimizer::new(vec![param_arc.clone()], lr, momentum, 0.0, 0.0, true)?;

        // Step 1
        optimizer.step()?;
        // buf_1 = 0.1. d_p = 0.1 + 0.9 * 0.1 = 0.19. param_1 = 1.0 - 0.1 * 0.19 = 0.981
        let val_step1 = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        assert!((val_step1 - 0.981).abs() < 1e-6, "Nesterov Step 1: Got {}", val_step1);

        // Mettre à jour le gradient du paramètre pour le step suivant
        param_arc.read().unwrap().tensor.data.write().unwrap().grad = Some(Tensor::new(vec![0.1], vec![1]).unwrap());

        // Step 2
        optimizer.step()?;
        // buf_2 = 0.9 * 0.1 (buf_1) + 0.1 (grad) = 0.19.
        // d_p = 0.1 (grad) + 0.9 * 0.19 (buf_2) = 0.1 + 0.171 = 0.271
        // param_2 = 0.981 - 0.1 * 0.271 = 0.9539
        let val_step2 = param_arc.read().unwrap().tensor.get_f32_data().unwrap()[0];
        assert!((val_step2 - 0.9539).abs() < 1e-6, "Nesterov Step 2: Got {}", val_step2);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "SGD state_dict not implemented yet.")]
    fn test_sgd_state_dict_panics() {
        let param = mock_param_rwlock(1.0, 0.1, "panic_state");
        let optimizer = SgdOptimizer::new(vec![param], 0.1, 0.0, 0.0, 0.0, false).unwrap();
        let _ = optimizer.state_dict(); 
    }

    #[test]
    #[should_panic(expected = "SGD load_state_dict not implemented yet.")]
    fn test_sgd_load_state_dict_panics() {
        let param = mock_param_rwlock(1.0, 0.1, "panic_load");
        let mut optimizer = SgdOptimizer::new(vec![param.clone()], 0.1, 0.0, 0.0, 0.0, false).unwrap();
        let dummy_momentum_buffers: HashMap<usize, Tensor> = HashMap::new();
        let dummy_optimizer_state = OptimizerState::Sgd { momentum_buffers: dummy_momentum_buffers };
        optimizer.load_state_dict(&dummy_optimizer_state).unwrap();
    }

    #[test]
    fn test_sgd_add_param_group() -> Result<(), NeuraRustError> {
        let param1 = mock_param_rwlock(1.0, 0.1, "g1p1");
        let mut optimizer = SgdOptimizer::new(vec![param1.clone()], 0.1, 0.0, 0.0, 0.0, false)?;
        
        let param2 = mock_param_rwlock(2.0, 0.1, "g2p1");
        let mut opts_g2 = crate::optim::param_group::ParamGroupOptions::default();
        opts_g2.lr = Some(0.01);
        opts_g2.momentum = Some(0.5);
        let mut group2 = crate::optim::param_group::ParamGroup::new(vec![param2.clone()]);
        group2.options = opts_g2;
        optimizer.add_param_group(group2);

        assert_eq!(optimizer.param_groups().len(), 2);
        optimizer.step()?;

        let val_p1 = param1.read().unwrap().tensor.get_f32_data().unwrap()[0];
        let val_p2 = param2.read().unwrap().tensor.get_f32_data().unwrap()[0];

        // P1: lr=0.1, mom=0. grad=0.1 -> 1.0 - 0.1*0.1 = 0.99
        assert!((val_p1 - 0.99).abs() < 1e-6, "P1: {}", val_p1);
        // P2: lr=0.01, mom=0.5. grad=0.1.
        // buf1_g2 = 0.1. p2_val = 2.0 - 0.01 * 0.1 = 1.999
        assert!((val_p2 - 1.999).abs() < 1e-6, "P2: {}", val_p2);
        Ok(())
    }

    // TODO: Potentially add tests for multiple parameter groups with different settings.
    // TODO: Add tests for CUDA tensors if/when SGD supports them.
    // TODO: Add tests for sparse gradients if/when SGD supports them.
    // TODO: Ensure all optimizer variants (momentum, nesterov, weight_decay) are tested with state_dict.
} 