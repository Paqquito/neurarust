use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

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

#[derive(Debug)]
pub struct RmsPropParamState {
    pub square_avg: Tensor,
    pub grad_avg: Option<Tensor>,
    pub momentum_buffer: Option<Tensor>,
}

impl Clone for RmsPropParamState {
    fn clone(&self) -> Self {
        let new_square_avg = match self.square_avg.dtype() {
            DType::F32 => Tensor::new(
                self.square_avg.get_f32_data().expect("Failed to get F32 data for square_avg clone"),
                self.square_avg.shape(),
            ).expect("Failed to create new F32 Tensor for square_avg clone"),
            DType::F64 => Tensor::new_f64(
                self.square_avg.get_f64_data().expect("Failed to get F64 data for square_avg clone"),
                self.square_avg.shape(),
            ).expect("Failed to create new F64 Tensor for square_avg clone"),
            DType::I32 | DType::I64 | DType::Bool => todo!("rmsprop: non supporté pour ce DType (square_avg)"),
        };

        let new_grad_avg = self.grad_avg.as_ref().map(|ga| {
            match ga.dtype() {
                DType::F32 => Tensor::new(
                    ga.get_f32_data().expect("Failed to get F32 data for grad_avg clone"),
                    ga.shape(),
                ).expect("Failed to create new F32 Tensor for grad_avg clone"),
                DType::F64 => Tensor::new_f64(
                    ga.get_f64_data().expect("Failed to get F64 data for grad_avg clone"),
                    ga.shape(),
                ).expect("Failed to create new F64 Tensor for grad_avg clone"),
                DType::I32 | DType::I64 | DType::Bool => todo!("rmsprop: non supporté pour ce DType (grad_avg)"),
            }
        });

        let new_momentum_buffer = self.momentum_buffer.as_ref().map(|mb| {
            match mb.dtype() {
                DType::F32 => Tensor::new(
                    mb.get_f32_data().expect("Failed to get F32 data for momentum_buffer clone"),
                    mb.shape(),
                ).expect("Failed to create new F32 Tensor for momentum_buffer clone"),
                DType::F64 => Tensor::new_f64(
                    mb.get_f64_data().expect("Failed to get F64 data for momentum_buffer clone"),
                    mb.shape(),
                ).expect("Failed to create new F64 Tensor for momentum_buffer clone"),
                DType::I32 | DType::I64 | DType::Bool => todo!("rmsprop: non supporté pour ce DType (momentum_buffer)"),
            }
        });

        RmsPropParamState {
            square_avg: new_square_avg,
            grad_avg: new_grad_avg,
            momentum_buffer: new_momentum_buffer,
        }
    }
}

#[derive(Debug)]
pub struct RmsPropOptimizer {
    param_groups: Vec<ParamGroup>,
    alpha: f32,
    eps: f32,
    momentum: f32,
    centered: bool,
    iterations: u64,
    state: HashMap<String, RmsPropParamState>,
}

impl RmsPropOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: impl IntoIterator<Item = Arc<RwLock<Parameter>>>,
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

        let params_vec: Vec<Arc<RwLock<Parameter>>> = params.into_iter().collect();
        let mut main_param_group = ParamGroup::new(params_vec);
        main_param_group.options.lr = Some(lr);
        main_param_group.options.weight_decay = Some(weight_decay);

        Ok(Self {
            param_groups: vec![main_param_group],
            alpha,
            eps,
            momentum,
            centered,
            iterations: 0,
            state: HashMap::new(),
        })
    }
}

// Fonction utilitaire pour créer un tenseur scalaire du même type que `other`
fn full_like_with_val(other: &Tensor, value: f32) -> Result<Tensor, NeuraRustError> {
    match other.dtype() {
        DType::F32 => full(&[], value),
        DType::F64 => full_f64(&[], value as f64),
        DType::I32 | DType::I64 | DType::Bool => todo!("rmsprop: non supporté pour ce DType (other)"),
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

        for group_idx in 0..self.param_groups.len() {
            let group = &self.param_groups[group_idx];
            let lr = group.options.lr.ok_or_else(|| NeuraRustError::ConfigurationError("Missing LR in RmsProp param group".to_string()))?;
            let weight_decay = group.options.weight_decay.unwrap_or(0.0);

            for param_arc in &group.params {
                let mut param_locked = match param_arc.write() {
                    Ok(p) => p,
                    Err(_) => {
                        let param_name_display = param_arc.read().ok().and_then(|p_read| p_read.name().map(|n| n.to_string())).unwrap_or_else(|| format!("Unnamed_param_at_{:?}", Arc::as_ptr(param_arc)));
                        eprintln!("Warning: RmsPropOptimizer::step could not acquire write lock for parameter {:?}. Skipping update.", param_name_display);
                        continue;
                    }
                };

                let param_name = match param_locked.name() {
                    Some(name) => name.to_string(),
                    None => format!("unnamed_param_at_{:?}", Arc::as_ptr(param_arc)),
                };

                if param_locked.grad().is_none() || !param_locked.requires_grad() {
                    continue;
                }

                let grad_tensor = param_locked.grad().as_ref().unwrap().clone();
                
                let param_data = &param_locked.tensor;
                let wd_val_tensor = full_like_with_val(param_data, weight_decay)?;
                let wd_term = mul_op(param_data, &wd_val_tensor)?;
                let effective_grad = add_op(&grad_tensor, &wd_term)?;

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
                        return Err(NeuraRustError::InternalError(format!("RMSprop (centered): grad_avg is None for param '{}', this should not happen.", param_name)));
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
                        return Err(NeuraRustError::InternalError(format!("RMSprop (momentum): momentum_buffer is None for param '{}'", param_name)));
                    }
                } else {
                    update_val_no_lr
                };

                let lr_tensor = full_like_with_val(&effective_grad, lr)?;
                let param_delta = mul_op(&final_update_val, &lr_tensor)?;

                param_locked.tensor.direct_sub_inplace(&param_delta)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param_arc in &group.params {
                if let Ok(mut param) = param_arc.write() {
                    if param.requires_grad() {
                        param.zero_grad();
                    }
                } else {
                    let param_name_display = param_arc.read().ok().and_then(|p_read| p_read.name().map(|n| n.to_string())).unwrap_or_else(|| format!("Unnamed_param_at_{:?}", Arc::as_ptr(param_arc)));
                    eprintln!("Warning: RmsPropOptimizer::zero_grad could not acquire write lock for parameter {:?}.", param_name_display);
                }
            }
        }
    }

    fn add_param_group(&mut self, param_group: ParamGroup) {
        for param_arc in &param_group.params {
            let param_locked = param_arc.read().expect("Failed to lock param for RmsProp state init in add_param_group");
            let param_name = param_locked.name().map(|n| n.to_string()).unwrap_or_else(|| format!("unnamed_param_at_{:?}", Arc::as_ptr(param_arc)));
            
            if !self.state.contains_key(&param_name) {
                let ref_tensor = &param_locked.tensor;
                self.state.entry(param_name.clone()).or_insert_with(|| {
                    RmsPropParamState {
                        square_avg: zeros_like(ref_tensor).expect("Failed to init square_avg in add_param_group"),
                        grad_avg: if self.centered { Some(zeros_like(ref_tensor).expect("Failed to init grad_avg in add_param_group")) } else { None },
                        momentum_buffer: if self.momentum > 0.0 { Some(zeros_like(ref_tensor).expect("Failed to init momentum_buffer in add_param_group")) } else { None },
                    }
                });
            }
        }
        self.param_groups.push(param_group);
    }

    fn param_groups(&self) -> &[ParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.param_groups
    }

    fn load_state_dict(
        &mut self,
        state_dict: &OptimizerState,
    ) -> Result<(), NeuraRustError> {
        match state_dict {
            OptimizerState::RmsProp { param_states, lr: loaded_lr, alpha, eps, weight_decay: loaded_weight_decay, momentum, centered, iterations } => {
                self.state = param_states.clone();
                self.alpha = *alpha;
                self.eps = *eps;
                self.momentum = *momentum;
                self.centered = *centered;
                self.iterations = *iterations;

                // Mettre à jour les options du premier groupe de paramètres
                if let Some(group) = self.param_groups.get_mut(0) {
                    group.options.lr = Some(*loaded_lr);
                    group.options.weight_decay = Some(*loaded_weight_decay);
                } else {
                    // Gérer le cas où il n'y a pas de groupe de paramètres (devrait être rare pour un optimiseur chargé)
                    return Err(NeuraRustError::ConfigurationError("Cannot load state_dict: RmsPropOptimizer has no parameter groups.".to_string()));
                }
                Ok(())
            }
            _ => Err(NeuraRustError::ConfigurationError("Invalid state_dict type for RmsPropOptimizer".to_string())),
        }
    }

    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        let lr = self.param_groups.first().and_then(|pg| pg.options.lr).unwrap_or(self.eps);
        let weight_decay = self.param_groups.first().and_then(|pg| pg.options.weight_decay).unwrap_or(0.0);

        Ok(OptimizerState::RmsProp {
            param_states: self.state.clone(),
            lr,
            alpha: self.alpha,
            eps: self.eps,
            weight_decay,
            momentum: self.momentum,
            centered: self.centered,
            iterations: self.iterations,
        })
    }
} 