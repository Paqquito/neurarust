//!
//! # Adagrad Optimizer
//! 
//! Implements the Adagrad optimization algorithm.
//! Adagrad adapts the learning rate element-wise based on the historical sum of squared gradients.
//! See: [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

use crate::{
    device::StorageDevice,
    error::NeuraRustError,
    nn::parameter::Parameter,
    optim::{Optimizer, OptimizerState, ParamGroup},
    tensor::Tensor,
    tensor::create::{full as tensor_create_full, full_f64 as tensor_create_full_f64},
    types::DType,
    ops::arithmetic::{add_op, div_op, mul_op},
    ops::arithmetic::pow::pow_op,
};
use std::sync::{Arc, Mutex};
use log::debug;
use std::ops::DerefMut;

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
    /// Term added to the denominator for numerical stability.
    eps: f32,
    /// Stores the sum of squared gradients for each parameter in each group.
    pub(crate) state_sum_gradient_squares: Vec<Vec<Tensor>>,
    /// Stores the number of steps taken for each parameter group.
    pub(crate) steps: Vec<usize>,
}

/// Creates a scalar tensor on CPU for internal Adagrad use.
/// Note: Ignores the device argument as Tensor::to is not yet implemented.
fn create_scalar_tensor_adagrad_internal(value: f32, dtype: DType, _device: &StorageDevice) -> Result<Tensor, NeuraRustError> {
    match dtype {
        DType::F32 => tensor_create_full(&[], value),
        DType::F64 => tensor_create_full_f64(&[], value as f64),
        // _ => Err(NeuraRustError::UnsupportedOperation(format!(
        //     "Adagrad internal scalar tensor creation: Unsupported DType {:?}. Only F32 and F64 are supported.",
        //     dtype
        // ))),
    }
}

/// Creates a tensor with a given shape filled with the initial accumulator value on CPU.
/// Note: Ignores the device argument as Tensor::to is not yet implemented.
fn tensor_initial_state_adagrad(shape: &[usize], value: f32, dtype: DType, _device: &StorageDevice) -> Result<Tensor, NeuraRustError> {
    match dtype {
        DType::F32 => tensor_create_full(shape, value),
        DType::F64 => tensor_create_full_f64(shape, value as f64),
        // _ => Err(NeuraRustError::UnsupportedOperation(format!(
        //     "Adagrad initial state tensor creation: Unsupported DType {:?}. Only F32 and F64 are supported.",
        //     dtype
        // ))),
    }
}

impl AdagradOptimizer {
    /// Creates a new `AdagradOptimizer` instance.
    ///
    /// # Arguments
    ///
    /// * `params`: An iterator over `Arc<Mutex<Parameter>>` to optimize.
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
        params: impl Iterator<Item = Arc<Mutex<Parameter>>>,
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

        let params_vec: Vec<Arc<Mutex<Parameter>>> = params.collect();
        let main_param_group = ParamGroup::new(params_vec, lr, weight_decay, lr_decay);

        if main_param_group.params.is_empty() {
            debug!("AdagradOptimizer: creating optimizer with no parameters.");
            return Ok(Self {
                param_groups: vec![main_param_group],
                initial_accumulator_value,
                eps,
                state_sum_gradient_squares: Vec::new(),
                steps: vec![0], 
            });
        }
        
        let mut state_sum_gradient_squares_group = Vec::with_capacity(main_param_group.params.len());
        let first_param_guard = main_param_group.params[0].lock().map_err(|_| NeuraRustError::LockError{lock_type: "Mutex".to_string(), reason: "Failed to lock first param for device/dtype".to_string()})?;
        let device = first_param_guard.tensor.device();
        let dtype = first_param_guard.tensor.dtype();
        drop(first_param_guard);

        for p_arc in main_param_group.params.iter() {
            let p_guard = p_arc.lock().map_err(|_| NeuraRustError::LockError{lock_type: "Mutex".to_string(), reason: "Failed to lock param for state init".to_string()})?;
            let p_tensor = &p_guard.tensor;
            if p_tensor.device() != device || p_tensor.dtype() != dtype {
                return Err(NeuraRustError::UnsupportedOperation(
                    "All parameters in the initial group must have the same device and dtype.".to_string()
                ));
            }
            let grad_sq_sum = tensor_initial_state_adagrad(&p_tensor.shape(), initial_accumulator_value, p_tensor.dtype(), &p_tensor.device())?;
            state_sum_gradient_squares_group.push(grad_sq_sum);
        }

        Ok(Self {
            param_groups: vec![main_param_group],
            initial_accumulator_value,
            eps,
            state_sum_gradient_squares: vec![state_sum_gradient_squares_group],
            steps: vec![0],
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
        for (group_idx, group) in self.param_groups.iter_mut().enumerate() {
            let group_lr = group.lr;
            let group_weight_decay = group.weight_decay;
            let group_lr_decay = group.lr_decay;
            let eps_val = self.eps;

            self.steps[group_idx] += 1;
            let step_count = self.steps[group_idx] as f32; 
            
            let decayed_lr = if group_lr_decay > 0.0 {
                group_lr / (1.0 + (step_count - 1.0) * group_lr_decay)
            } else {
                group_lr
            };

            for (param_idx, param_arc) in group.params.iter().enumerate() {
                let mut param_guard = param_arc.lock().map_err(|_| NeuraRustError::LockError{lock_type: "Mutex".to_string(), reason: "Failed to lock param in step".to_string()})?;
                
                if !param_guard.tensor.requires_grad() {
                    debug!("Parameter {:?} does not require grad, skipping.", param_guard.name().unwrap_or_default());
                    continue;
                }

                if let Some(grad_tensor_val) = param_guard.tensor.grad() {
                    let mut grad_tensor = grad_tensor_val;

                    if group_weight_decay != 0.0 {
                        let wd_scalar = create_scalar_tensor_adagrad_internal(group_weight_decay, grad_tensor.dtype(), &grad_tensor.device())?;
                        let wd_term = mul_op(&param_guard.tensor, &wd_scalar)?;
                        grad_tensor = add_op(&grad_tensor, &wd_term)?;
                    }
                    
                    if self.state_sum_gradient_squares.get(group_idx).and_then(|g| g.get(param_idx)).is_none() {
                        debug!("Optimizer state for group {}, param {} is missing, attempting to create.", group_idx, param_idx);
                        let p_tensor = &param_guard.tensor;
                        match tensor_initial_state_adagrad(&p_tensor.shape(), self.initial_accumulator_value, p_tensor.dtype(), &p_tensor.device()) {
                            Ok(state_tensor) => {
                                while self.state_sum_gradient_squares.len() <= group_idx {
                                    self.state_sum_gradient_squares.push(Vec::new());
                                }
                                self.state_sum_gradient_squares[group_idx].push(state_tensor);
                                debug!("AdagradOptimizer: Created missing state tensor for group {}, param {}.", group_idx, param_idx);
                            },
                            Err(e) => {
                                log::error!("Failed to create state tensor for new Adagrad group: {:?}. Skipping group.", e);
                                return Err(e);
                            }
                        }
                    }
                    let state_sum_sq = &mut self.state_sum_gradient_squares[group_idx][param_idx];
                    
                    let grad_squared = mul_op(&grad_tensor, &grad_tensor)?;
                    *state_sum_sq = add_op(state_sum_sq, &grad_squared)?;
                    
                    let exponent_scalar = create_scalar_tensor_adagrad_internal(0.5f32, state_sum_sq.dtype(), &state_sum_sq.device())?;
                    let sqrt_state_sum_sq = pow_op(state_sum_sq, &exponent_scalar)?;
                    
                    let eps_scalar = create_scalar_tensor_adagrad_internal(eps_val, sqrt_state_sum_sq.dtype(), &sqrt_state_sum_sq.device())?;
                    let denom = add_op(&sqrt_state_sum_sq, &eps_scalar)?;
                                                            
                    let lr_scalar = create_scalar_tensor_adagrad_internal(decayed_lr, grad_tensor.dtype(), &grad_tensor.device())?;
                    let step_size_num = mul_op(&grad_tensor, &lr_scalar)?;

                    let step_update = div_op(&step_size_num, &denom)?;
                                        
                    let parameter_mut_ref: &mut Parameter = param_guard.deref_mut();
                    parameter_mut_ref.tensor.direct_sub_inplace(&step_update)?;

                    debug!(
                        "Updated param {:?} (lr: {}, wd: {}) (Skipping norm logging)",
                        param_guard.name().unwrap_or_default(), group_lr, group_weight_decay
                    );

                } else {
                    debug!("Parameter {:?} has no gradient, skipping update.", param_guard.name().unwrap_or_default());
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
        for group in self.param_groups.iter_mut() {
            for param_arc in group.params.iter() {
                if let Ok(param_guard) = param_arc.lock() {
                    param_guard.tensor.clear_grad();
                } else {
                     log::error!("Failed to acquire lock for param during zero_grad");
                }
            }
        }
    }
    
    /// Adds a new parameter group to the optimizer.
    /// 
    /// Initializes the necessary state (sum of squared gradients) for the new parameters.
    /// Ensures parameters in the new group are consistent in device and dtype.
    fn add_param_group(&mut self, param_group: ParamGroup) {
        let num_params_in_group = param_group.params.len();
        let mut new_state_group = Vec::with_capacity(num_params_in_group);

        if num_params_in_group > 0 {
            let first_param_guard = param_group.params[0].lock().unwrap();
            let device = first_param_guard.tensor.device();
            let dtype = first_param_guard.tensor.dtype();
            drop(first_param_guard);

            for p_arc in param_group.params.iter() {
                let p_guard = p_arc.lock().unwrap();
                let p_tensor = &p_guard.tensor;
                if p_tensor.device() != device || p_tensor.dtype() != dtype {
                    log::error!("Parameter in new group has mismatched device/dtype. Skipping state init for this param group.");
                    new_state_group.clear(); 
                    break; 
                }
                match tensor_initial_state_adagrad(&p_tensor.shape(), self.initial_accumulator_value, p_tensor.dtype(), &p_tensor.device()) {
                    Ok(state_tensor) => {
                        new_state_group.push(state_tensor)
                    },
                    Err(e) => {
                        log::error!("Failed to create state tensor for new Adagrad group: {:?}. Skipping group.", e);
                        new_state_group.clear();
                        break; 
                    }
                }
            }
        }
        
        if new_state_group.len() == num_params_in_group || num_params_in_group == 0 {
            self.param_groups.push(param_group);
            self.state_sum_gradient_squares.push(new_state_group);
            self.steps.push(0);
            debug!("AdagradOptimizer: Added new param group. Total groups: {}", self.param_groups.len());
        } else {
             debug!("AdagradOptimizer: Did not add new param group due to errors in state initialization for its parameters.");
        }
    }

    /// Saves the optimizer's state.
    /// 
    /// Returns an `OptimizerState::Adagrad` variant containing the
    /// accumulated squared gradients and step counts for each parameter group.
    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        // Cloner l'état interne. C'est important car les Tensors dans state_sum_gradient_squares
        // sont mutables en interne (via CoW ou accès direct). Un clonage profond est nécessaire
        // si les Tensors eux-mêmes doivent être clonés (ce qui est le cas par défaut avec Tensor::clone).
        let state_clone = self.state_sum_gradient_squares.clone();
        let steps_clone = self.steps.clone();

        Ok(OptimizerState::Adagrad {
            state_sum_gradient_squares: state_clone,
            steps: steps_clone,
        })
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
            OptimizerState::Adagrad { state_sum_gradient_squares: loaded_state_sum_sq, steps: loaded_steps } => {
                if loaded_state_sum_sq.len() != self.param_groups.len() || loaded_steps.len() != self.param_groups.len() {
                    return Err(NeuraRustError::OptimizerError(format!(
                        "Loaded state_dict has {} groups, but optimizer has {} groups.",
                        loaded_state_sum_sq.len(), self.param_groups.len()
                    )));
                }

                // Vérification de base de la compatibilité (nombre de tenseurs par groupe)
                for group_idx in 0..self.param_groups.len() {
                    if loaded_state_sum_sq[group_idx].len() != self.param_groups[group_idx].params.len() {
                         return Err(NeuraRustError::OptimizerError(format!(
                            "Loaded state_dict group {} has {} states, but optimizer group {} has {} parameters.",
                            group_idx, loaded_state_sum_sq[group_idx].len(), group_idx, self.param_groups[group_idx].params.len()
                        )));
                    }
                    // On pourrait ajouter des vérifications de shape/dtype ici, mais restons simple pour l'instant.
                }

                // Remplacer l'état actuel
                // .clone() est nécessaire car state_dict est une référence.
                self.state_sum_gradient_squares = loaded_state_sum_sq.clone();
                self.steps = loaded_steps.clone();
                
                debug!("AdagradOptimizer state loaded successfully. Steps: {:?}", self.steps);
                Ok(())
            }
            _ => Err(NeuraRustError::OptimizerError(
                "Invalid state_dict type for AdagradOptimizer".to_string(),
            )),
        }
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