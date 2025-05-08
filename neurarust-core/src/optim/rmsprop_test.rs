use crate::optim::Optimizer;
use crate::types::DType as DataType;
use crate::device::StorageDevice as Device;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use crate::NeuraRustError;
use std::sync::{Arc, RwLock};

use super::RmsPropOptimizer;

// Helper function to create a test parameter
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
mod tests {
    use super::*;
    use crate::types::DType as DataType;
    use crate::device::StorageDevice as Device;
    use crate::tensor::Tensor;
    use crate::NeuraRustError;
    use crate::nn::parameter::Parameter;
    use crate::optim::Optimizer as _;

    #[test]
    fn test_rmsprop_optimizer_new() -> Result<(), NeuraRustError> {
        let param = create_test_param(vec![1.0, 2.0, 3.0], vec![3], Some("param1".to_string()));
        let params = vec![param];
        let _optimizer = RmsPropOptimizer::new(params, 0.01, 0.99, 1e-8, 0.0, 0.0, false)?;
        Ok(())
    }

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

    #[test]
    fn test_rmsprop_basic_step() -> Result<(), NeuraRustError> {
        let initial_data = vec![1.0, 2.0, 3.0];
        let param_arc_rwlock = create_test_param(initial_data.clone(), vec![3], Some("param_step".to_string()));
        let params = vec![param_arc_rwlock.clone()];

        let mut optimizer = RmsPropOptimizer::new(params, 0.1, 0.9, 1e-8, 0.0, 0.0, false)?;

        {
            let mut p_lock = param_arc_rwlock.try_write().expect("Failed to lock param for grad");
            let grad_data = vec![0.1, 0.2, 0.3];
            let grad_tensor = Tensor::new(grad_data, vec![3])?;
            let mut tensor_data_guard = p_lock.write_data();
            tensor_data_guard.grad = Some(grad_tensor);
        }

        optimizer.step()?;

        let p_lock_after = param_arc_rwlock.try_read().expect("Failed to lock param after step");
        let data_after_step = p_lock_after.tensor().get_f32_data()?;

        for i in 0..initial_data.len() {
            assert_ne!(data_after_step[i], initial_data[i], "Parameter at index {} was not updated.", i);
        }
        Ok(())
    }

    // TODO: Add more tests:
    // - Test with weight decay
    // - Test with momentum
    // - Test with centered RMSprop
    // - Test with multiple parameters (named and unnamed)
    // - Test with no_grad parameters
    // - Test clearing gradients with zero_grad()
    // - Test step behavior over multiple iterations (convergence, stability)
    // - Test state_dict and load_state_dict
} 