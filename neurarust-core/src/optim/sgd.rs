use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use crate::nn::parameter::Parameter;
use crate::optim::optimizer_trait::Optimizer;
use crate::optim::param_group::ParamGroup;
use crate::optim::optimizer_state::OptimizerState;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};
use crate::types::DType;
use std::sync::RwLock;
use log;

/// Implements the Stochastic Gradient Descent (SGD) optimizer.
///
/// Supports momentum, weight decay, and Nesterov momentum.
#[derive(Debug)]
pub struct SgdOptimizer {
    param_groups: Vec<ParamGroup>,
    momentum: f32,
    nesterov: bool,
    // Les buffers suivants sont la source de vérité de l'état interne.
    momentum_buffers: Arc<RwLock<HashMap<usize, Tensor>>>,
}

impl SgdOptimizer {
    /// Creates a new `SgdOptimizer`.
    ///
    /// # Arguments
    ///
    /// * `params`: An iterator of parameters (`Arc<Mutex<Parameter>>`) to optimize.
    ///   These parameters will be placed into a default parameter group.
    /// * `lr`: The learning rate.
    /// * `momentum`: Momentum factor (default: 0.0).
    /// * `group_weight_decay`: Weight decay (L2 penalty) factor for the default group.
    /// * `nesterov`: Enables Nesterov momentum (default: false).
    ///
    /// To use multiple parameter groups with different learning rates or other
    /// hyperparameters, first create the optimizer with an empty iterator or an
    /// initial set of parameters, then use `add_param_group`.
    pub fn new(
        params: impl IntoIterator<Item = Arc<Mutex<Parameter>>>,
        lr: f32,
        momentum: f32,
        group_weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        let params_vec: Vec<Arc<Mutex<Parameter>>> = params.into_iter().collect();
        let default_param_group = ParamGroup::new(params_vec, lr, group_weight_decay);
        
        SgdOptimizer {
            param_groups: vec![default_param_group],
            momentum,
            nesterov,
            momentum_buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Optimizer for SgdOptimizer {
    fn step(&mut self) -> Result<(), NeuraRustError> {
        for group in self.param_groups.iter() {
            let lr = group.lr;
            let weight_decay = group.weight_decay;
            let momentum_val = self.momentum;
            let nesterov = self.nesterov;

            for param_arc in group.params.iter() {
                let param_id = Arc::as_ptr(param_arc) as usize;
                let mut param = param_arc.lock().map_err(|e| {
                    NeuraRustError::LockError {
                        lock_type: "write".to_string(),
                        reason: format!("Failed to lock parameter mutex in SGD step: {}", e),
                    }
                })?;

                if !param.requires_grad() {
                    continue;
                }

                let base_grad_opt = param.grad();
                let base_grad = match base_grad_opt {
                    Some(g) => {
                        g
                    }
                    None => {
                        continue;
                    }
                };

                let param_dtype = param.dtype();
                if base_grad.dtype() != param_dtype {
                    return Err(NeuraRustError::DataTypeMismatch {
                        expected: param_dtype,
                        actual: base_grad.dtype(),
                        operation: "Parameter and Gradient in SGD step".to_string(),
                    });
                }

                let mut d_p = match base_grad.dtype() {
                    DType::F32 => {
                        let data = base_grad.get_f32_data().map_err(|e| {
                            e
                        })?;
                        Tensor::new(data, base_grad.shape().to_vec()).map_err(|e| {
                            e
                        })?
                    }
                    DType::F64 => {
                        let data = base_grad.get_f64_data().map_err(|e| {
                            e
                        })?;
                        Tensor::new_f64(data, base_grad.shape().to_vec()).map_err(|e| {
                            e
                        })?
                    }
                };

                if weight_decay != 0.0 {
                    let param_tensor_ref: &Tensor = param.deref(); 
                    let mut wd_term = match param_dtype {
                        DType::F32 => {
                            let data = param_tensor_ref.get_f32_data()?;
                            Tensor::new(data, param_tensor_ref.shape().to_vec())?
                        }
                        DType::F64 => {
                            let data = param_tensor_ref.get_f64_data()?;
                            Tensor::new_f64(data, param_tensor_ref.shape().to_vec())?
                        }
                    };
                    
                    match param_dtype {
                        DType::F32 => wd_term.direct_mul_scalar_f32_inplace(weight_decay as f32)?,
                        DType::F64 => wd_term.direct_mul_scalar_f64_inplace(weight_decay as f64)?,
                    }
                    d_p.add_(&wd_term)?;
                }

                let final_update_value: Tensor;

                if momentum_val != 0.0 {
                    let final_val_from_momentum_branch: Tensor;
                    {
                        let mut buffers = match self.momentum_buffers.write() {
                            Ok(guard) => {
                                guard
                            }
                            Err(poisoned) => {
                                log::warn!("RwLock for momentum_buffers was poisoned in step. Recovering writer guard.");
                                poisoned.into_inner()
                            }
                        };

                        let buffer = buffers.entry(param_id).or_insert_with(|| {
                            crate::tensor::create::zeros_like(&d_p)
                                .expect("Failed to create momentum buffer")
                        });

                        match param_dtype {
                            DType::F32 => buffer.direct_mul_scalar_f32_inplace(momentum_val as f32)?,
                            DType::F64 => buffer.direct_mul_scalar_f64_inplace(momentum_val as f64)?,
                        }
                        
                        buffer.add_(&d_p)?;

                        if nesterov {
                            let mut nesterov_momentum_term = match buffer.dtype() {
                                DType::F32 => Tensor::new(buffer.get_f32_data()?, buffer.shape().to_vec())?,
                                DType::F64 => Tensor::new_f64(buffer.get_f64_data()?, buffer.shape().to_vec())?,
                            };

                            match param_dtype {
                                DType::F32 => nesterov_momentum_term.direct_mul_scalar_f32_inplace(momentum_val as f32)?,
                                DType::F64 => nesterov_momentum_term.direct_mul_scalar_f64_inplace(momentum_val as f64)?,
                            }
                            
                            let mut nesterov_update = d_p.clone();

                            nesterov_update.add_(&nesterov_momentum_term)?;
                            
                            final_val_from_momentum_branch = nesterov_update;
                        } else {
                            final_val_from_momentum_branch = buffer.clone();
                        }
                    }
                    final_update_value = final_val_from_momentum_branch;
                } else {
                    final_update_value = d_p;
                }
                
                let mut final_delta = match final_update_value.dtype() {
                     DType::F32 => {
                         let data = final_update_value.get_f32_data()?;
                         Tensor::new(data, final_update_value.shape().to_vec())?
                     }
                     DType::F64 => {
                         let data = final_update_value.get_f64_data()?;
                         Tensor::new_f64(data, final_update_value.shape().to_vec())?
                     }
                 };

                match final_delta.dtype() {
                    DType::F32 => final_delta.direct_mul_scalar_f32_inplace(lr as f32)?,
                    DType::F64 => final_delta.direct_mul_scalar_f64_inplace(lr as f64)?,
                }

                let param_tensor = param.deref_mut();
                param_tensor.direct_sub_inplace(&final_delta)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in self.param_groups.iter_mut() {
            for param_arc in group.params.iter_mut() {
                match param_arc.lock() {
                    Ok(param_guard) => {
                        param_guard.clear_grad();
                    }
                    Err(poisoned) => {
                        log::warn!("Mutex for parameter in SgdOptimizer::zero_grad was poisoned. Recovering.");
                        let param_guard = poisoned.into_inner();
                        param_guard.clear_grad();
                    }
                }
            }
        }
    }

    fn add_param_group(&mut self, param_group: ParamGroup) {
        self.param_groups.push(param_group);
    }

    fn load_state_dict(&mut self, state_dict: &OptimizerState) -> Result<(), NeuraRustError> {
        match state_dict {
            OptimizerState::Sgd { momentum_buffers: incoming_buffers } => {
                if self.momentum == 0.0 && !incoming_buffers.is_empty() {
                   log::warn!("Loading SGD state with momentum buffers, but optimizer momentum is 0.");
                }
                 if self.momentum != 0.0 && incoming_buffers.is_empty() {
                    log::warn!("Loading empty SGD state, but optimizer momentum is non-zero. Momentum buffers will be initialized on first step if not present here.");
                 }
                
                let mut buffers_guard = match self.momentum_buffers.write() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        log::warn!("RwLock for momentum_buffers was poisoned during load_state_dict. Recovering write guard.");
                        poisoned.into_inner()
                    }
                };
                *buffers_guard = incoming_buffers.clone();
                Ok(())
            }
            OptimizerState::Placeholder => {
                 if self.momentum != 0.0 {
                     log::warn!("Loading Placeholder state, but optimizer momentum is non-zero. Momentum buffers will be initialized on first step.");
                 }
                 match self.momentum_buffers.write() {
                     Ok(mut guard) => guard.clear(),
                     Err(poisoned) => {
                         log::warn!("RwLock for momentum_buffers was poisoned during clear for Placeholder. Recovering and clearing.");
                         poisoned.into_inner().clear();
                     }
                 }
                 Ok(())
            }
            _ => Err(NeuraRustError::UnsupportedOperation(
                "Attempted to load incompatible state into SgdOptimizer".to_string(),
            )),
        }
    }

    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        let buffers_guard = match self.momentum_buffers.read() { 
            Ok(guard) => guard,
            Err(poisoned) => {
                log::warn!("RwLock for momentum_buffers was poisoned during state_dict. Recovering read guard.");
                poisoned.into_inner()
            }
        };
        let cloned_buffers = buffers_guard.clone();
        Ok(OptimizerState::Sgd { momentum_buffers: cloned_buffers })
    }
} 