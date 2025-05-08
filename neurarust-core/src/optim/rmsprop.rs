use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock, Weak};

use crate::tensor::Tensor;
use crate::nn::parameter::Parameter;
use crate::optim::optimizer_trait::Optimizer;
use crate::optim::param_group::ParamGroup;
use crate::optim::optimizer_state::OptimizerState;
use crate::error::NeuraRustError;
use crate::types::DType;
use crate::tensor::create::{zeros_like, full, full_f64};

#[derive(Clone, Debug, PartialEq)]
pub struct RmsPropHyperParams {
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub centered: bool,
}

#[derive(Debug, Clone)]
pub struct RmsPropParamState {
    pub square_avg: Tensor,
    pub grad_avg: Option<Tensor>,
    pub momentum_buffer: Option<Tensor>,
}

#[derive(Debug)]
pub struct RmsPropOptimizer {
    param_refs: Vec<Weak<RwLock<Parameter>>>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    iterations: u64,
    state: HashMap<String, RmsPropParamState>,
}

impl RmsPropOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Parameter>>>,
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
    ) -> Result<Self, NeuraRustError> {
        if lr <= 0.0 {
            return Err(NeuraRustError::ConfigurationError("Learning rate must be positive".to_string()));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(NeuraRustError::ConfigurationError("alpha must be in [0.0, 1.0]".to_string()));
        }
        if eps <= 0.0 {
            return Err(NeuraRustError::ConfigurationError("Epsilon must be positive".to_string()));
        }
        if weight_decay < 0.0 {
            return Err(NeuraRustError::ConfigurationError("Weight decay must be non-negative".to_string()));
        }
        if momentum < 0.0 {
            return Err(NeuraRustError::ConfigurationError("Momentum must be non-negative".to_string()));
        }
        let param_refs = params.iter().map(Arc::downgrade).collect();
        Ok(Self {
            param_refs,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            iterations: 0,
            state: HashMap::new(),
        })
    }
    fn get_strong_params(&self) -> Vec<Arc<RwLock<Parameter>>> {
        self.param_refs.iter().filter_map(Weak::upgrade).collect()
    }
}

// Fonction utilitaire pour créer un tenseur scalaire du même type que `other`
fn full_like_with_val(other: &Tensor, value: f32) -> Result<Tensor, NeuraRustError> {
    match other.dtype() {
        DType::F32 => full(&[], value),
        DType::F64 => full_f64(&[], value as f64),
    }
}

impl Optimizer for RmsPropOptimizer {
    fn step(&mut self) -> Result<(), NeuraRustError> {
        self.iterations += 1;

        use crate::ops::arithmetic::add::add_op;
        use crate::ops::arithmetic::mul::mul_op;
        use crate::ops::arithmetic::div::div_op;
        use crate::ops::arithmetic::sub::sub_op;
        use crate::ops::arithmetic::pow::pow_op;

        for param_arc in self.get_strong_params() {
            let mut param_locked = match param_arc.write() {
                Ok(p) => p,
                Err(_) => {
                    let param_name_display = param_arc.read().ok().and_then(|p_read| p_read.name().map(|n| n.to_string())).unwrap_or_else(|| format!("Unnamed_param_at_{:?}", Arc::as_ptr(&param_arc)));
                    eprintln!("Warning: RmsPropOptimizer::step could not acquire write lock for parameter {:?}. Skipping update.", param_name_display);
                    continue;
                }
            };

            let param_name = match param_locked.name() {
                Some(name) => name.to_string(),
                None => format!("unnamed_param_at_{:?}", Arc::as_ptr(&param_arc)),
            };

            if param_locked.grad().is_none() {
                continue;
            }

            let grad_tensor = param_locked.grad().as_ref().unwrap().clone();
            
            let effective_grad = if self.weight_decay != 0.0 {
                if self.iterations == 1 && param_locked.requires_grad() {
                    eprintln!(
                        "Warning: Weight decay (L2 penalty) for RMSprop is not fully implemented for param '{}'. Original gradient will be used.",
                        param_name
                    );
                }
                grad_tensor.clone()
            } else {
                grad_tensor.clone()
            };

            let state_entry = self.state.entry(param_name.clone()).or_insert_with(|| {
                RmsPropParamState {
                    square_avg: zeros_like(&effective_grad).expect("Failed to init square_avg"),
                    grad_avg: if self.centered { Some(zeros_like(&effective_grad).expect("Failed to init grad_avg")) } else { None },
                    momentum_buffer: if self.momentum > 0.0 { Some(zeros_like(&effective_grad).expect("Failed to init momentum_buffer")) } else { None },
                }
            });

            let alpha_tensor = full_like_with_val(&effective_grad, self.alpha)?;
            let one_minus_alpha_tensor = full_like_with_val(&effective_grad, 1.0 - self.alpha)?;
            let eps_tensor = full_like_with_val(&effective_grad, self.eps)?;
            let two_tensor = full_like_with_val(&effective_grad, 2.0)?;
            let half_tensor = full_like_with_val(&effective_grad, 0.5)?;

            let grad_sq = pow_op(&effective_grad, &two_tensor)?;
            let term1_sq_avg = mul_op(&state_entry.square_avg, &alpha_tensor)?;
            let term2_sq_avg = mul_op(&grad_sq, &one_minus_alpha_tensor)?;
            state_entry.square_avg = add_op(&term1_sq_avg, &term2_sq_avg)?;
            
            let avg_denom = if self.centered {
                if let Some(ga_prev) = &state_entry.grad_avg {
                    let term1_ga = mul_op(ga_prev, &alpha_tensor)?;
                    let term2_ga = mul_op(&effective_grad, &one_minus_alpha_tensor)?;
                    let current_grad_avg = add_op(&term1_ga, &term2_ga)?;
                    state_entry.grad_avg = Some(current_grad_avg.clone());

                    let grad_avg_sq = pow_op(&current_grad_avg, &two_tensor)?;
                    let denom_base = sub_op(&state_entry.square_avg, &grad_avg_sq)?;
                    let denom_base_eps = add_op(&denom_base, &eps_tensor)?;
                    pow_op(&denom_base_eps, &half_tensor)?
                } else {
                    return Err(NeuraRustError::ConfigurationError(format!("RMSprop (centered): grad_avg is None for param '{}', this should not happen if initialized correctly.", param_name)));
                }
            } else {
                let denom_base_eps = add_op(&state_entry.square_avg, &eps_tensor)?;
                pow_op(&denom_base_eps, &half_tensor)?
            };
            
            let update_val_no_lr = div_op(&effective_grad, &avg_denom)?;
            
            let final_update_val = if self.momentum > 0.0 {
                if let Some(mom_buf_prev) = &state_entry.momentum_buffer {
                    let momentum_tensor = full_like_with_val(&effective_grad, self.momentum)?;
                    let buf_updated = add_op(&mul_op(mom_buf_prev, &momentum_tensor)?, &update_val_no_lr)?;
                    state_entry.momentum_buffer = Some(buf_updated.clone());
                    buf_updated
                } else {
                    return Err(NeuraRustError::ConfigurationError(format!("RMSprop (momentum): momentum_buffer is None for param '{}'", param_name)));
                }
            } else {
                update_val_no_lr
            };

            let lr_tensor = full_like_with_val(&effective_grad, self.lr)?;
            let param_delta = mul_op(&final_update_val, &lr_tensor)?;

            let current_param_data = &param_locked.tensor;
            let updated_param_data = sub_op(current_param_data, &param_delta)?;
            
            let original_name = param_locked.name.clone();
            *param_locked = Parameter::new(updated_param_data.detach(), original_name);
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param_arc in self.get_strong_params() {
            if let Ok(mut param) = param_arc.write() {
                if param.requires_grad() {
                    param.zero_grad();
                }
            } else {
                let param_name_display = param_arc.read().ok().and_then(|p| p.name().map(|n| n.to_string())).unwrap_or_else(|| format!("Unnamed_param_at_{:?}", Arc::as_ptr(&param_arc)));
                eprintln!(
                    "Warning: RmsPropOptimizer::zero_grad could not acquire lock for parameter {:?}. Gradients may not be cleared.",
                    param_name_display
                );
            }
        }
    }

    fn add_param_group(&mut self, _param_group: ParamGroup) {
        eprintln!("Warning: RmsPropOptimizer::add_param_group called, but complex per-group hyperparameter management (beyond lr/weight_decay in ParamGroup) is not fully implemented for RMSprop's specific hyperparams. Default RMSprop hyperparams will be used for all parameters unless step() is modified to use ParamGroup's lr/weight_decay.");
    }

    fn load_state_dict(
        &mut self,
        state_dict: &OptimizerState,
    ) -> Result<(), NeuraRustError> {
        match state_dict {
            OptimizerState::RmsProp { param_states, lr, alpha, eps, weight_decay, momentum, centered, iterations } => {
                self.state = param_states.clone();
                self.lr = *lr;
                self.alpha = *alpha;
                self.eps = *eps;
                self.weight_decay = *weight_decay;
                self.momentum = *momentum;
                self.centered = *centered;
                self.iterations = *iterations;
                Ok(())
            }
            _ => Err(NeuraRustError::ConfigurationError(
                "Invalid state dictionary type for RmsPropOptimizer. Expected OptimizerState::RmsProp.".to_string(),
            )),
        }
    }

    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        Ok(OptimizerState::RmsProp {
            param_states: self.state.clone(),
            lr: self.lr,
            alpha: self.alpha,
            eps: self.eps,
            weight_decay: self.weight_decay,
            momentum: self.momentum,
            centered: self.centered,
            iterations: self.iterations,
        })
    }
} 