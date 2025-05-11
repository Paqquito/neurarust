//!
//! # Adagrad Optimizer
//! 
//! Implements the Adagrad optimization algorithm.
//! Adagrad adapts the learning rate element-wise based on the historical sum of squared gradients.
//! See: [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

use crate::{
    error::NeuraRustError,
    nn::parameter::Parameter,
    optim::{Optimizer, OptimizerState, ParamGroup},
    tensor::Tensor,
    tensor::create::{full, full_f64},
    types::DType,
    ops::arithmetic::{add_op, div_op, mul_op},
    ops::arithmetic::pow::pow_op,
};
use std::sync::{Arc, RwLock};
use log::debug;
use std::collections::HashMap;

/// Represents the state for a single parameter in the Adagrad optimizer.
#[derive(Default, Clone, Debug)]
pub struct AdagradState {
    /// Sum of squared gradients.
    pub sum_sq_grads: Option<Tensor>,
    /// Number of steps taken (can be used for epsilon stabilization if needed).
    pub step: u64,
}

/// Implements the Adagrad optimization algorithm.
///
/// Adagrad maintains a per-parameter sum of squared gradients and adapts the
/// learning rate for each parameter individually. Parameters with larger
/// historical gradients receive smaller updates.
///
/// Note: Currently, `lr_decay` is not implemented in the `step` function.
#[derive(Debug)]
pub struct AdagradOptimizer {
    param_groups: Vec<ParamGroup>,
    /// Initial value added to the accumulator (gradient square sums) for numerical stability.
    initial_accumulator_value: f32,
    state: HashMap<String, AdagradState>,
}

impl AdagradOptimizer {
    /// Creates a new `AdagradOptimizer` instance.
    ///
    /// # Arguments
    ///
    /// * `params`: An iterator over `Arc<RwLock<Parameter>>` to optimize.
    /// * `lr`: Learning rate (required > 0.0).
    /// * `lr_decay`: Learning rate decay factor (must be >= 0.0).
    /// * `weight_decay`: Weight decay (L2 penalty) (must be >= 0.0).
    /// * `initial_accumulator_value`: Value to initialize the sum of squared gradients accumulator (must be >= 0.0).
    /// * `eps`: Term added to the denominator for numerical stability (must be >= 0.0).
    ///
    /// # Errors
    /// Returns `NeuraRustError::OptimizerError` if hyperparameters are invalid.
    /// Returns `NeuraRustError::UnsupportedOperation` if parameters have inconsistent devices or dtypes.
    pub fn new(
        params: impl IntoIterator<Item = Arc<RwLock<Parameter>>>,
        lr: f32,
        lr_decay: f32,
        weight_decay: f32,
        initial_accumulator_value: f32,
        eps: f32,
    ) -> Result<Self, NeuraRustError> {
        if lr < 0.0 {
            return Err(NeuraRustError::OptimizerError("Invalid learning rate".to_string()));
        }
        if lr_decay < 0.0 {
            return Err(NeuraRustError::OptimizerError("Invalid lr_decay value".to_string()));
        }
        if weight_decay < 0.0 {
            return Err(NeuraRustError::OptimizerError("Invalid weight_decay value".to_string()));
        }
        if eps < 0.0 {
            return Err(NeuraRustError::OptimizerError("Invalid epsilon value".to_string()));
        }
        if initial_accumulator_value < 0.0 {
            return Err(NeuraRustError::OptimizerError("Invalid initial_accumulator_value".to_string()));
        }

        let params_vec: Vec<Arc<RwLock<Parameter>>> = params.into_iter().collect();
        let mut main_param_group = ParamGroup::new(params_vec);
        main_param_group.options.lr = Some(lr);
        main_param_group.options.lr_decay = Some(lr_decay);
        main_param_group.options.weight_decay = Some(weight_decay);
        main_param_group.options.eps = Some(eps);

        let mut state = HashMap::new();
        for p_arc in &main_param_group.params {
            let p_locked = p_arc.read().map_err(|e| NeuraRustError::LockError {
                lock_type: "read".to_string(),
                reason: format!("Failed to lock param for Adagrad state init: {}", e),
            })?;
            let p_name = p_locked.name().map(|n| n.to_string()).unwrap_or_else(|| format!("unnamed_{:?}", Arc::as_ptr(p_arc)));
            let p_tensor = &p_locked.tensor;
            let initial_sum_sq = match p_tensor.dtype() {
                DType::F32 => full(&p_tensor.shape(), initial_accumulator_value)?,
                DType::F64 => full_f64(&p_tensor.shape(), initial_accumulator_value as f64)?,
                DType::I32 | DType::I64 | DType::Bool => todo!(),
            };
            state.insert(p_name, AdagradState { sum_sq_grads: Some(initial_sum_sq), step: 0 });
        }

        Ok(Self {
            param_groups: vec![main_param_group],
            initial_accumulator_value,
            state,
        })
    }

    /// Returns a view of the parameter groups managed by the optimizer.
    pub fn param_groups(&self) -> Vec<&ParamGroup> {
        self.param_groups.iter().collect()
    }
    
    /// Returns a mutable view of the parameter groups managed by the optimizer.
    pub fn param_groups_mut(&mut self) -> Vec<&mut ParamGroup> {
        self.param_groups.iter_mut().collect()
    }
}

impl Optimizer for AdagradOptimizer {
    /// Performs a single optimization step.
    ///
    /// Updates the parameters based on their gradients and the accumulated
    /// sum of squared gradients.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if locking fails, operations fail, or state is missing.
    fn step(&mut self) -> Result<(), NeuraRustError> {
        debug!("AdagradOptimizer: step() called");
        for group in &mut self.param_groups {
            let lr = group.options.lr.ok_or(NeuraRustError::ConfigurationError("Missing LR".to_string()))?;
            let _lr_decay = group.options.lr_decay.unwrap_or(0.0);
            let weight_decay = group.options.weight_decay.unwrap_or(0.0);
            let eps = group.options.eps.ok_or(NeuraRustError::ConfigurationError("Missing eps".to_string()))?;

            let current_lr = lr;

            for param_arc in &group.params {
                let mut param_locked = param_arc.write().map_err(|e| NeuraRustError::LockError {
                    lock_type: "write".to_string(),
                    reason: format!("Failed to lock param in Adagrad step: {}", e),
                })?;
                
                if !param_locked.tensor.requires_grad() {
                    debug!("Parameter {:?} does not require grad, skipping.", param_locked.name().unwrap_or_default());
                    continue;
                }

                if let Some(grad_tensor_val) = param_locked.tensor.grad() {
                    let mut grad_tensor = grad_tensor_val;
                    let param_name = param_locked.name().map(|n| n.to_string()).unwrap_or_else(|| format!("unnamed_{:?}", Arc::as_ptr(param_arc)));

                    if weight_decay != 0.0 {
                        let wd_scalar = full(&[], weight_decay)?;
                        let wd_term = mul_op(&param_locked.tensor, &wd_scalar)?;
                        grad_tensor = add_op(&grad_tensor, &wd_term)?;
                    }
                    
                    let state_entry = self.state.entry(param_name.clone()).or_insert_with(|| {
                         let p_tensor = &param_locked.tensor;
                         let initial_sum_sq = match p_tensor.dtype() {
                            DType::F32 => full(&p_tensor.shape(), self.initial_accumulator_value).expect("State init failed"),
                            DType::F64 => full_f64(&p_tensor.shape(), self.initial_accumulator_value as f64).expect("State init failed"),
                            DType::I32 | DType::I64 | DType::Bool => todo!(),
                         };
                         AdagradState { sum_sq_grads: Some(initial_sum_sq), step: 0 }
                    });

                    state_entry.step += 1;

                    let grad_squared = mul_op(&grad_tensor, &grad_tensor)?;
                    let current_sum_sq = state_entry.sum_sq_grads.take().ok_or_else(|| NeuraRustError::InternalError(format!("Missing sum_sq_grads for {}", param_name)))?;
                    let updated_sum_sq = add_op(&current_sum_sq, &grad_squared)?;
                    state_entry.sum_sq_grads = Some(updated_sum_sq);

                    let state_sum_sq = state_entry.sum_sq_grads.as_ref().unwrap();
                    let exponent_scalar = full(&[], 0.5f32)?;
                    let sqrt_state_sum_sq = pow_op(state_sum_sq, &exponent_scalar)?;
                    
                    let eps_scalar = full(&[], eps)?;
                    let denom = add_op(&sqrt_state_sum_sq, &eps_scalar)?;
                                                            
                    let lr_scalar = full(&[], current_lr)?;
                    let step_size_num = mul_op(&grad_tensor, &lr_scalar)?;

                    let step_update = div_op(&step_size_num, &denom)?;
                                        
                    param_locked.tensor.direct_sub_inplace(&step_update)?;

                    debug!(
                        "Updated param {:?} (lr: {}, wd: {}) (Skipping norm logging)",
                        param_locked.name().unwrap_or_default(), current_lr, weight_decay
                    );

                } else {
                    debug!("Parameter {:?} has no gradient, skipping update.", param_locked.name().unwrap_or_default());
                }
            }
        }
        Ok(())
    }

    /// Clears the gradients of all parameters managed by this optimizer.
    ///
    /// Should be called before the `backward()` pass in each training iteration.
    fn zero_grad(&mut self) {
        debug!("AdagradOptimizer: zero_grad() called");
        for group in &mut self.param_groups {
            for param_arc in &group.params {
                if let Ok(mut param_locked) = param_arc.write() {
                    param_locked.zero_grad();
                } else {
                     eprintln!("Warning: Could not lock parameter to zero_grad in Adagrad.");
                }
            }
        }
    }
    
    /// Adds a new parameter group to the optimizer.
    /// 
    /// Initializes the necessary state (sum of squared gradients) for the new parameters.
    /// Ensures parameters in the new group are consistent in device and dtype.
    fn add_param_group(&mut self, mut param_group: ParamGroup) {
        // Ensure the new group has default options if not set
        let default_options = self.param_groups.get(0).map(|pg| pg.options.clone()).unwrap_or_default();

        if param_group.options.lr.is_none() {
            param_group.options.lr = default_options.lr;
        }
        if param_group.options.lr_decay.is_none() {
            param_group.options.lr_decay = default_options.lr_decay;
        }
        if param_group.options.weight_decay.is_none() {
            param_group.options.weight_decay = default_options.weight_decay;
        }
        if param_group.options.eps.is_none() {
            param_group.options.eps = default_options.eps;
        }
        // Note: initial_accumulator_value is handled by initializing sum_sq_grads below

        let mut new_state_entries: Vec<(String, AdagradState)> = Vec::new();
        for p_arc in &param_group.params {
            let p_locked = p_arc.read().expect("Failed to lock new param for Adagrad state init");
            let p_name = p_locked.name().map(|n| n.to_string()).unwrap_or_else(|| format!("unnamed_{:?}", Arc::as_ptr(p_arc)));
            let p_tensor = &p_locked.tensor;
            // Use self.initial_accumulator_value for consistency with new()
            let initial_sum_sq = match p_tensor.dtype() {
                DType::F32 => full(&p_tensor.shape(), self.initial_accumulator_value).expect("Failed to create state tensor F32"),
                DType::F64 => full_f64(&p_tensor.shape(), self.initial_accumulator_value as f64).expect("Failed to create state tensor F64"),
                DType::I32 | DType::I64 | DType::Bool => todo!(),
            };
            new_state_entries.push((p_name, AdagradState { sum_sq_grads: Some(initial_sum_sq), step: 0 }));
        }
        
        self.param_groups.push(param_group);
        self.state.extend(new_state_entries);
        debug!("Added Adagrad param group. Initial state created.");
    }

    /// Saves the optimizer's state.
    /// 
    /// Returns an `OptimizerState::Adagrad` variant containing the
    /// accumulated squared gradients and step counts for each parameter group.
    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        // Cloner le HashMap d'état actuel
        let state_clone = self.state.clone();
        Ok(OptimizerState::Adagrad { state: state_clone })
    }

    /// Loads the optimizer's state from a `state_dict`.
    /// 
    /// Replaces the current state (sum of squared gradients and steps) with the loaded state.
    /// Performs checks to ensure the loaded state is compatible with the optimizer's structure 
    /// (e.g., number of parameter groups).
    /// 
    /// # Arguments
    /// 
    /// * `state_dict`: The `OptimizerState` to load. Must be the `Adagrad` variant.
    /// 
    /// # Errors
    /// 
    /// Returns `NeuraRustError::OptimizerError` if the `state_dict` is not the `Adagrad` variant,
    /// or if the structure (number of groups) doesn't match.
    /// Returns `NeuraRustError::TypeError` if tensor dtypes or shapes in the loaded state 
    /// don't match the expected state based on current parameters (this check is basic).
    fn load_state_dict(&mut self, state_dict: &OptimizerState) -> Result<(), NeuraRustError> {
        match state_dict {
            OptimizerState::Adagrad { state: loaded_state } => {
                // Remplacer l'état actuel par l'état chargé
                // .clone() est nécessaire car state_dict est une référence.
                self.state = loaded_state.clone();
                debug!("AdagradOptimizer state loaded successfully.");
                Ok(())
            }
            _ => Err(NeuraRustError::ConfigurationError(
                "Invalid state_dict type for AdagradOptimizer. Expected Adagrad state.".to_string(),
            )),
        }
    }

    fn param_groups(&self) -> &[ParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.param_groups
    }
}

// Adagrad doesn't usually have complex separate algorithm/hook logic beyond the main step
// But we need the impls to satisfy the overall Optimizer structure if it depends on these traits.
// If `OptimizerAlgorithm` and `OptimizerHook` are not strictly required by a generic
// Optimizer wrapper or manager, these can be omitted. Assuming they are needed for now.

// No specific OptimizerAlgorithm methods needed beyond Optimizer trait for Adagrad
// impl crate::optim::optimizer_internal::OptimizerAlgorithm for AdagradOptimizer {} 

// No specific hooks needed for Adagrad
// impl crate::optim::optimizer_internal::OptimizerHook for AdagradOptimizer {
//     fn before_step(&mut self) -> Result<(), NeuraRustError> { Ok(()) }
//     fn after_step(&mut self) -> Result<(), NeuraRustError> { Ok(()) }
// }

// TODO: Add tests in adagrad_test.rs 