use std::sync::Arc;
use std::collections::HashMap;
use crate::nn::parameter::Parameter;
use crate::optim::Optimizer;
use crate::optim::param_group::{ParamGroup /*, ParamGroupOptions*/};
use crate::optim::optimizer_state::OptimizerState;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use std::sync::RwLock;
use crate::ops::arithmetic::add::add_op;
use crate::ops::arithmetic::mul::mul_op_scalar;

/// Represents the state for a single parameter in the SGD optimizer.
#[derive(Default, Clone, Debug)]
struct SgdState {
    /// Momentum buffer.
    momentum_buffer: Option<Tensor>,
}

/// Implements the Stochastic Gradient Descent (SGD) optimizer.
///
/// Supports momentum, weight decay, and Nesterov momentum.
#[derive(Debug)]
pub struct SgdOptimizer {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, SgdState>,
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
    pub fn new(params: Vec<Arc<RwLock<Parameter>>>, lr: f32, momentum: f32, dampening: f32, weight_decay: f32, nesterov: bool) -> Result<Self, NeuraRustError> {
        if lr < 0.0 {
            return Err(NeuraRustError::ConfigurationError("Invalid learning rate: {}".to_string()));
        }
        if momentum < 0.0 {
             return Err(NeuraRustError::ConfigurationError("Invalid momentum value: {}".to_string()));
        }
        if weight_decay < 0.0 {
             return Err(NeuraRustError::ConfigurationError("Invalid weight_decay value: {}".to_string()));
        }

        // Créer le groupe de paramètres unique
        let mut default_param_group = ParamGroup::new(params);
        // Définir les options
        default_param_group.options.lr = Some(lr);
        default_param_group.options.momentum = Some(momentum);
        default_param_group.options.dampening = Some(dampening);
        default_param_group.options.weight_decay = Some(weight_decay);
        default_param_group.options.nesterov = Some(nesterov);

        Ok(SgdOptimizer {
            param_groups: vec![default_param_group],
            state: HashMap::new(),
        })
    }
}

impl Optimizer for SgdOptimizer {
    fn step(&mut self) -> Result<(), NeuraRustError> {
        for group in &mut self.param_groups {
            // Obtenir les options du groupe, avec des valeurs par défaut si non spécifiées
            let lr = group.options.lr.ok_or(NeuraRustError::ConfigurationError("Missing LR in SGD group".to_string()))?;
            let momentum = group.options.momentum.unwrap_or(0.0);
            let dampening = group.options.dampening.unwrap_or(0.0);
            let weight_decay = group.options.weight_decay.unwrap_or(0.0);
            let nesterov = group.options.nesterov.unwrap_or(false);

            for param_arc in &group.params {
                let mut param_locked = param_arc.write().map_err(|e| NeuraRustError::LockError {
                    lock_type: "write".to_string(),
                    reason: format!("Failed to lock param in SGD step: {}", e),
                })?;
                
                if param_locked.grad().is_none() { continue; }
                let grad = param_locked.grad().unwrap().clone(); // grad() ne prend pas d'arg

                let param_name = param_locked.name().map(|n| n.to_string()).unwrap_or_else(|| format!("unnamed_{:?}", Arc::as_ptr(param_arc)));

                let mut grad_processed = grad.clone();

                // Apply weight decay (L2 penalty)
                if weight_decay != 0.0 {
                    grad_processed = add_op(&grad_processed, &mul_op_scalar(&param_locked.tensor, weight_decay)?)?;
                }

                // Apply momentum
                if momentum != 0.0 {
                    let state_entry = self.state.entry(param_name.clone()).or_default();
                    
                    if state_entry.momentum_buffer.is_none() {
                        state_entry.momentum_buffer = Some(grad_processed.clone());
                    } else {
                        let buf = state_entry.momentum_buffer.as_mut().unwrap();
                        // buf = buf * momentum + grad_processed * (1 - dampening)
                        let buf_scaled = mul_op_scalar(buf, momentum)?;
                        let grad_dampened = mul_op_scalar(&grad_processed, 1.0 - dampening)?;
                        *buf = add_op(&buf_scaled, &grad_dampened)?;
                    }

                    if nesterov {
                        // grad_processed = grad_processed + buf * momentum
                        let buf_scaled_nesterov = mul_op_scalar(state_entry.momentum_buffer.as_ref().unwrap(), momentum)?;
                        grad_processed = add_op(&grad_processed, &buf_scaled_nesterov)?;
                    } else {
                        grad_processed = state_entry.momentum_buffer.as_ref().unwrap().clone();
                    }
                }

                // Perform the update: param = param - lr * grad_processed
                let update_val = mul_op_scalar(&grad_processed, -lr)?;
                let current_tensor = param_locked.tensor.clone();
                let updated_tensor = add_op(&current_tensor, &update_val)?;
                param_locked.tensor = updated_tensor;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param_arc in &group.params {
                if let Ok(mut param_locked) = param_arc.write() {
                    param_locked.zero_grad();
                } else {
                     eprintln!("Warning: Could not lock parameter to zero_grad in SGD.");
                }
            }
        }
    }

    fn add_param_group(&mut self, param_group: ParamGroup) {
        self.param_groups.push(param_group);
    }

    fn param_groups(&self) -> &[ParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.param_groups
    }

    fn load_state_dict(&mut self, _state_dict: &OptimizerState) -> Result<(), NeuraRustError> {
        unimplemented!("SGD load_state_dict not implemented yet.")
    }

    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        unimplemented!("SGD state_dict not implemented yet.")
    }
}

#[cfg(test)]
#[path = "sgd_test.rs"]
mod tests; 