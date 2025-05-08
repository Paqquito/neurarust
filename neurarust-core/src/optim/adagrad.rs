use std::sync::{Arc, Mutex};
use log::debug;

use crate::{
    error::NeuraRustError,
    optim::{optimizer_trait::Optimizer, ParamGroup},
    tensor::Tensor,
    nn::parameter::Parameter,
    types::DType,
    tensor::create::{full as tensor_full_fn, full_f64 as tensor_full_f64_fn},
    ops::arithmetic::{add_op, mul_op, div_op, sub_op},
    ops::arithmetic::pow::pow_op,
};

#[derive(Debug)]
pub struct AdagradOptimizer {
    param_groups: Vec<ParamGroup>,
    state_sum_gradient_squares: Vec<Vec<Tensor>>,
    steps: Vec<usize>,
    default_lr_decay: f32,
    default_eps: f32,
    default_initial_accumulator_value: f32,
}

// Helper function to create a scalar tensor with a specific dtype and device
fn create_scalar_tensor(value: f32, dtype: DType, device: &crate::device::StorageDevice) -> Result<Tensor, NeuraRustError> {
    let tensor = match dtype {
        DType::F32 => tensor_full_fn(&[], value)?,
        DType::F64 => tensor_full_f64_fn(&[], value as f64)?,
        // _ => return Err(NeuraRustError::UnsupportedOperation(format!("Scalar tensor creation not supported for dtype {:?}", dtype)))
        // Pour l'instant, les autres types ne sont pas supportés par les ops Adagrad de toute façon
        /* _ => Err(NeuraRustError::DataTypeMismatch{ // Commented out as it's currently unreachable
            expected: DType::F32, // Ou F64
            actual: dtype,
            operation: "create_scalar_tensor for Adagrad".to_string(),
        })?,*/
    };
    // Actuellement, tensor_full_fn crée sur CPU. Si le device cible est différent, il faudrait un .to(device)
    if &tensor.device() != device {
        // return tensor.to(device); // Tensor::to n'est pas encore implémenté
        return Err(NeuraRustError::DeviceMismatch {
            expected: tensor.device(), // ou *device
            actual: *device, // ou tensor.device()
            operation: "create_scalar_tensor - device transfer needed but not implemented".to_string(),
        });
    }
    Ok(tensor)
}

impl AdagradOptimizer {
    pub fn new(
        params: impl IntoIterator<Item = Arc<Mutex<Parameter>>>,
        lr: f32,
        lr_decay: f32,
        weight_decay: f32,
        initial_accumulator_value: f32,
        eps: f32,
    ) -> Result<Self, NeuraRustError> {
        let params_vec: Vec<Arc<Mutex<Parameter>>> = params.into_iter().collect();
        if params_vec.is_empty() && lr > 0.0 {
            return Err(NeuraRustError::OptimizerError("Cannot create optimizer with no parameters for a non-zero learning rate".to_string()));
        }
        let param_group = ParamGroup::new(params_vec, lr, weight_decay);
        
        let num_params_in_group = param_group.params.len();
        if num_params_in_group == 0 && lr > 0.0 {
            return Err(NeuraRustError::OptimizerError("Cannot create optimizer with no parameters for a non-zero learning rate (group check)".to_string()));
        }

        let mut state_sum_gradient_squares_group = Vec::with_capacity(num_params_in_group);
        for p_arc in param_group.params.iter() {
            let p_guard = p_arc.lock().map_err(|_| NeuraRustError::LockError{ lock_type: "mutex".to_string(), reason: "Failed to lock param for Adagrad state init".to_string()})?;
            let p_tensor = &p_guard.tensor;
            let grad_sq_sum = match p_tensor.dtype() {
                DType::F32 => tensor_full_fn(&p_tensor.shape(), initial_accumulator_value)?,
                DType::F64 => tensor_full_f64_fn(&p_tensor.shape(), initial_accumulator_value as f64)?,
                /* _ => return Err(NeuraRustError::DataTypeMismatch { // Commented out as it's currently unreachable
                    expected: DType::F32, 
                    actual: p_tensor.dtype(), 
                    operation: "Adagrad state initialization".to_string() 
                })*/
            };
            state_sum_gradient_squares_group.push(grad_sq_sum);
        }

        Ok(Self {
            param_groups: vec![param_group],
            state_sum_gradient_squares: vec![state_sum_gradient_squares_group],
            steps: vec![0],
            default_lr_decay: lr_decay,
            default_eps: eps,
            default_initial_accumulator_value: initial_accumulator_value,
        })
    }

    pub fn param_groups(&self) -> Vec<&ParamGroup> {
        self.param_groups.iter().collect()
    }
    
    pub fn param_groups_mut(&mut self) -> Vec<&mut ParamGroup> {
        self.param_groups.iter_mut().collect()
    }
}

impl Optimizer for AdagradOptimizer {
    fn step(&mut self) -> Result<(), NeuraRustError> {
        debug!("AdagradOptimizer: step() called");
        for (group_idx, group) in self.param_groups.iter_mut().enumerate() {
            let lr = group.lr;
            let lr_decay = self.default_lr_decay;
            let weight_decay = group.weight_decay;
            let eps = self.default_eps;

            self.steps[group_idx] += 1;
            let current_step = self.steps[group_idx];

            let effective_lr = if lr_decay > 0.0 {
                lr / (1.0 + (current_step - 1) as f32 * lr_decay)
            } else {
                lr
            };
            debug!("AdagradOptimizer: group {}, effective_lr = {}", group_idx, effective_lr);

            for (param_idx, param_arc) in group.params.iter().enumerate() {
                let mut param_guard = param_arc.lock().map_err(|_| NeuraRustError::LockError{ lock_type: "mutex".to_string(), reason: "Failed to lock param in Adagrad step".to_string()})?;
                if let Some(grad_tensor) = param_guard.grad() {
                    let detached_grad = Tensor::detach(&grad_tensor);
                    let mut grad_processed = detached_grad.clone();

                    if weight_decay > 0.0 {
                        let p_tensor = &param_guard.tensor;
                        let wd_scalar_tensor = create_scalar_tensor(weight_decay, p_tensor.dtype(), &p_tensor.device())?;
                        let wd_term = mul_op(p_tensor, &wd_scalar_tensor).map_err(|e| NeuraRustError::InternalError(format!("Weight decay mul_op failed: {}", e)))?;
                        grad_processed = add_op(&grad_processed, &wd_term).map_err(|e| NeuraRustError::InternalError(format!("Weight decay add failed: {}", e)))?;
                        debug!("AdagradOptimizer: Applied weight_decay {} to grad for param {}", weight_decay, param_idx);
                    }

                    let grad_sq = mul_op(&grad_processed, &grad_processed).map_err(|e| NeuraRustError::InternalError(format!("grad_sq mul failed: {}",e)))?;
                    
                    let new_sum_sq = add_op(&self.state_sum_gradient_squares[group_idx][param_idx], &grad_sq)
                        .map_err(|e| NeuraRustError::InternalError(format!("State update add failed: {}", e)))?;
                    self.state_sum_gradient_squares[group_idx][param_idx] = new_sum_sq;
                    debug!("AdagradOptimizer: Updated sum_gradient_squares for param {} in group {}", param_idx, group_idx);

                    let current_sum_sq = &self.state_sum_gradient_squares[group_idx][param_idx];
                    
                    let exponent_val = 0.5f32;
                    let exponent_tensor = create_scalar_tensor(exponent_val, current_sum_sq.dtype(), &current_sum_sq.device())?;
                    let sqrt_sum_sq = pow_op(current_sum_sq, &exponent_tensor).map_err(|e| NeuraRustError::InternalError(format!("denom sqrt (pow_op) failed: {}", e)))?;

                    let eps_tensor = create_scalar_tensor(eps, sqrt_sum_sq.dtype(), &sqrt_sum_sq.device())?;
                    let denom = add_op(&sqrt_sum_sq, &eps_tensor).map_err(|e| NeuraRustError::InternalError(format!("denom add_op failed: {}", e)))?;
                    
                    let lr_scalar_tensor = create_scalar_tensor(effective_lr, grad_processed.dtype(), &grad_processed.device())?;
                    let update_val_num = mul_op(&grad_processed, &lr_scalar_tensor).map_err(|e| NeuraRustError::InternalError(format!("Update_val mul_op failed: {}", e)))?;
                    let update_val = div_op(&update_val_num, &denom).map_err(|e| NeuraRustError::InternalError(format!("Update_val div failed: {}", e)))?;

                    let p_tensor = &param_guard.tensor;
                    let updated_p_data = sub_op(p_tensor, &update_val).map_err(|e| NeuraRustError::InternalError(format!("p_data sub failed: {}",e)))?;
                    
                    let detached_updated_p_data = Tensor::detach(&updated_p_data);
                    param_guard.tensor = detached_updated_p_data.clone();
                    debug!("AdagradOptimizer: Updated param {} in group {}", param_idx, group_idx);

                } else {
                    debug!("AdagradOptimizer: No gradient for param {} in group {}, skipping update.", param_idx, group_idx);
                }
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        debug!("AdagradOptimizer: zero_grad() called");
        for group in self.param_groups.iter_mut() {
            for param_arc in group.params.iter_mut() {
                match param_arc.lock() {
                    Ok(param_guard) => param_guard.clear_grad(),
                    Err(_) => {
                        eprintln!("AdagradOptimizer: Failed to lock parameter (poisoned) for zero_grad on param_arc: {:?}. This is a critical error.", param_arc);
                        // Consider panicking or returning an error if zero_grad could fail.
                        // For now, just printing error, as Optimizer trait's zero_grad returns ().
                    }
                }
            }
        }
    }

    fn add_param_group(&mut self, param_group: ParamGroup) {
        let num_params_in_group = param_group.params.len();
        if num_params_in_group == 0 {
            debug!("AdagradOptimizer: Skipping adding an empty param group.");
            return;
        }
        let mut new_state_group = Vec::with_capacity(num_params_in_group);
        let initial_acc_val = self.default_initial_accumulator_value;

        for p_arc in param_group.params.iter() {
            match p_arc.lock() {
                Ok(p_guard) => {
                    let p_tensor = &p_guard.tensor;
                    match match p_tensor.dtype() {
                        DType::F32 => tensor_full_fn(&p_tensor.shape(), initial_acc_val),
                        DType::F64 => tensor_full_f64_fn(&p_tensor.shape(), initial_acc_val as f64),
                        /* _ => Err(NeuraRustError::DataTypeMismatch { // Commented out as it's currently unreachable
                                expected: DType::F32, 
                                actual: p_tensor.dtype(), 
                                operation: "Adagrad state init for new group".to_string() 
                            })*/
                    } {
                        Ok(grad_sq_sum) => {
                            new_state_group.push(grad_sq_sum);
                        },
                        Err(e) => {
                            eprintln!("AdagradOptimizer: Failed to create state for new param group: {:?}. Skipping param.", e);
                        }
                    }
                },
                Err(_) => {
                    eprintln!("AdagradOptimizer: Failed to lock param for new group state init. Skipping param.");
                }
            }
        }
        if new_state_group.len() == num_params_in_group {
            self.param_groups.push(param_group);
            self.state_sum_gradient_squares.push(new_state_group);
            self.steps.push(0);
            debug!("AdagradOptimizer: Added new param group.");
        } else {
            debug!("AdagradOptimizer: Did not add new param group due to errors in state initialization.");
        }
    }

    fn state_dict(&self) -> Result<crate::optim::optimizer_state::OptimizerState, NeuraRustError> {
        debug!("AdagradOptimizer: state_dict() called (placeholder)");
        Err(NeuraRustError::OptimizerError("state_dict for Adagrad not implemented".to_string()))
    }

    fn load_state_dict(&mut self, _state_dict: &crate::optim::optimizer_state::OptimizerState) -> Result<(), NeuraRustError> {
        debug!("AdagradOptimizer: load_state_dict() called (placeholder)");
        Err(NeuraRustError::OptimizerError("load_state_dict for Adagrad not implemented".to_string()))
    }
}

// TODO: Add tests in adagrad_test.rs 