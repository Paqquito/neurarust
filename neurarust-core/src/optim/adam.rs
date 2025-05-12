use crate::error::NeuraRustError;
use crate::nn::parameter::Parameter;
use crate::optim::{Optimizer, ParamGroup};
use crate::optim::optimizer_state::OptimizerState; // Pour state_dict/load_state_dict
use crate::tensor::create::{zeros_like, full, full_f64}; // Ensure full/full_f64 are available
use crate::tensor::Tensor;
use crate::types::DType;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Import specific op functions from their confirmed/likely locations
use crate::ops::arithmetic::add::add_op;
use crate::ops::arithmetic::div::div_op;
use crate::ops::arithmetic::mul::{mul_op, mul_op_scalar};
use crate::ops::arithmetic::pow::pow_op;
use crate::ops::arithmetic::max_elemwise::max_elemwise_op; // Import pour AMSGrad

/// Represents the state for a single parameter in the Adam optimizer.
#[derive(Default, Clone, Debug)]
pub struct AdamParamState {
    /// First moment vector (exponential moving average of gradients).
    pub m: Option<Tensor>,
    /// Second moment vector (exponential moving average of squared gradients).
    pub v: Option<Tensor>,
    /// Maximum value of v_hat seen so far (for AMSGrad).
    pub v_max: Option<Tensor>, 
}

/// Adam and AdamW Optimizer.
#[derive(Debug)]
pub struct AdamOptimizer {
    param_groups: Vec<ParamGroup>,
    iterations: u64,
    state: HashMap<String, AdamParamState>,
}

impl AdamOptimizer {
    #[allow(clippy::too_many_arguments)] // Arguments are standard for Adam
    pub fn new(
        params: Vec<Arc<RwLock<Parameter>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Result<Self, NeuraRustError> {
        if lr <= 0.0 {
            return Err(NeuraRustError::ConfigurationError(
                "Learning rate must be positive".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&beta1) {
            return Err(NeuraRustError::ConfigurationError(
                "Beta1 must be in [0, 1)".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&beta2) {
            return Err(NeuraRustError::ConfigurationError(
                "Beta2 must be in [0, 1)".to_string(),
            ));
        }
        if eps <= 0.0 {
            return Err(NeuraRustError::ConfigurationError(
                "Epsilon must be positive".to_string(),
            ));
        }
        if weight_decay < 0.0 {
            return Err(NeuraRustError::ConfigurationError(
                "Weight decay must be non-negative".to_string(),
            ));
        }

        let mut initial_group = ParamGroup::new(params);
        initial_group.options.lr = Some(lr);
        initial_group.options.betas = Some((beta1, beta2));
        initial_group.options.eps = Some(eps);
        initial_group.options.weight_decay = Some(weight_decay);
        initial_group.options.amsgrad = Some(amsgrad);

        Ok(AdamOptimizer {
            param_groups: vec![initial_group],
            iterations: 0,
            state: HashMap::new(),
        })
    }

    /// Returns strong references to the parameters managed by the optimizer.
    pub fn get_params(&self) -> Vec<Arc<RwLock<Parameter>>> {
        self.param_groups.get(0).map_or(Vec::new(), |group| group.params.clone())
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self) -> Result<(), NeuraRustError> {
        if self.param_groups.is_empty() || self.param_groups[0].params.is_empty() {
            return Ok(());
        }
        self.iterations += 1;

        for group in &mut self.param_groups {
            let lr = group.options.lr.ok_or_else(|| NeuraRustError::ConfigurationError("Missing LR".to_string()))?;
            let (beta1, beta2) = group.options.betas.ok_or_else(|| NeuraRustError::ConfigurationError("Missing betas".to_string()))?;
            let eps = group.options.eps.ok_or_else(|| NeuraRustError::ConfigurationError("Missing eps".to_string()))?;
            let weight_decay = group.options.weight_decay.unwrap_or(0.0);
            let amsgrad = group.options.amsgrad.unwrap_or(false);
            
            let bias_correction1 = 1.0 - beta1.powi(self.iterations as i32);
            let bias_correction2 = 1.0 - beta2.powi(self.iterations as i32);
            if bias_correction1.abs() < f32::EPSILON { return Err(NeuraRustError::ArithmeticError("bias_correction1 near zero".to_string())); }
            if bias_correction2.abs() < f32::EPSILON { return Err(NeuraRustError::ArithmeticError("bias_correction2 near zero".to_string())); }
    
            for param_arc in &group.params {
                let mut param_locked = param_arc.write().expect("Lock failed");
                if param_locked.grad().is_none() { continue; }
                let grad = param_locked.grad().as_ref().unwrap().clone(); 
                let param_dtype = grad.dtype();
                
                // Utiliser le nom du paramètre pour l'affichage/debug s'il existe, sinon un ID.
                let _display_name = param_locked.name().map(|n| n.to_string()).unwrap_or_else(|| format!("unnamed_param_at_{:p}", Arc::as_ptr(param_arc)));
                // Utiliser l'adresse du Arc comme clé unique pour l'état de l'optimiseur.
                let state_key = format!("{:p}", Arc::as_ptr(param_arc));
                
                let mut grad_decayed = grad.clone();
                if weight_decay != 0.0 {
                    grad_decayed = add_op(&grad_decayed, &mul_op_scalar(&param_locked.tensor, weight_decay)?)?;
                }
                
                let state_entry = self.state.entry(state_key.clone()).or_insert_with(AdamParamState::default);
    
                let m_prev = state_entry.m.clone().unwrap_or_else(|| zeros_like(&grad_decayed).unwrap());
                let term1_m = mul_op_scalar(&m_prev, beta1)?;
                let term2_m = mul_op_scalar(&grad_decayed, 1.0 - beta1)?;
                let m_t = add_op(&term1_m, &term2_m)?;
                state_entry.m = Some(m_t.clone());
    
                let v_prev = state_entry.v.clone().unwrap_or_else(|| zeros_like(&grad_decayed).unwrap());
                let grad_sq = mul_op(&grad_decayed, &grad_decayed)?;
                let term1_v = mul_op_scalar(&v_prev, beta2)?;
                let term2_v = mul_op_scalar(&grad_sq, 1.0 - beta2)?;
                let v_t = add_op(&term1_v, &term2_v)?;
                state_entry.v = Some(v_t.clone());
    
                let m_hat = match param_dtype {
                    DType::F32 => div_op(&m_t, &full(&[], bias_correction1)?)?,
                    DType::F64 => div_op(&m_t, &full_f64(&[], bias_correction1 as f64)?)?,
                    DType::I32 | DType::I64 | DType::Bool => {
                        return Err(NeuraRustError::UnsupportedOperation(
                            "AdamOptimizer n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                        );
                    },
                };
                let v_hat = match param_dtype {
                    DType::F32 => div_op(&v_t, &full(&[], bias_correction2)?)?,
                    DType::F64 => div_op(&v_t, &full_f64(&[], bias_correction2 as f64)?)?,
                    DType::I32 | DType::I64 | DType::Bool => {
                        return Err(NeuraRustError::UnsupportedOperation(
                            "AdamOptimizer n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                        );
                    },
                };
    
                let v_hat_for_update = if amsgrad {
                    let v_max_prev = state_entry.v_max.clone().unwrap_or_else(|| v_hat.clone());
                    let v_max_t = max_elemwise_op(&v_max_prev, &v_hat)?;
                    state_entry.v_max = Some(v_max_t.clone());
                    v_max_t
                } else {
                    v_hat
                };
    
                let sqrt_v_hat = match param_dtype {
                    DType::F32 => pow_op(&v_hat_for_update, &full(&[], 0.5f32)?)?,
                    DType::F64 => pow_op(&v_hat_for_update, &full_f64(&[], 0.5f64)?)?,
                    DType::I32 | DType::I64 | DType::Bool => {
                        return Err(NeuraRustError::UnsupportedOperation(
                            "AdamOptimizer n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                        );
                    },
                };
                
                let denom = match param_dtype {
                    DType::F32 => add_op(&sqrt_v_hat, &full(&[], eps)?)?,
                    DType::F64 => add_op(&sqrt_v_hat, &full_f64(&[], eps as f64)?)?,
                    DType::I32 | DType::I64 | DType::Bool => {
                        return Err(NeuraRustError::UnsupportedOperation(
                            "AdamOptimizer n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                        );
                    },
                };
                
                let update_num = mul_op_scalar(&m_hat, lr)?;
                let update = div_op(&update_num, &denom)?;

    
                let contiguous_update = update.contiguous()?;

                
                let update_inplace = crate::ops::arithmetic::neg::neg_op(&contiguous_update)?;
                
                let current_tensor = param_locked.tensor.clone();
                let updated_tensor = add_op(&current_tensor, &update_inplace)?;
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
                    eprintln!("Warning: Could not lock parameter to zero_grad.");
                }
            }
        }
    }

    fn param_groups(&self) -> &[ParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.param_groups
    }

    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        unimplemented!("AdamOptimizer::state_dict not implemented yet")
    }

    fn load_state_dict(&mut self, _state_dict: &OptimizerState) -> Result<(), NeuraRustError> {
        unimplemented!("AdamOptimizer::load_state_dict not implemented yet")
    }

    fn add_param_group(&mut self, param_group: ParamGroup) {
        self.param_groups.push(param_group);
        eprintln!("Warning: Added a new parameter group to AdamOptimizer.");
    }
}

#[cfg(test)]
#[path = "adam_test.rs"]
mod tests; 