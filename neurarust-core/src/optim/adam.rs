use crate::error::NeuraRustError;
use crate::nn::parameter::Parameter;
use crate::optim::Optimizer;
use crate::optim::optimizer_state::OptimizerState; // Pour state_dict/load_state_dict
use crate::tensor::create::{zeros_like, full, full_f64}; // Ensure full/full_f64 are available
use crate::tensor::Tensor;
use crate::types::DType;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

// Import specific op functions from their confirmed/likely locations
use crate::ops::arithmetic::add::add_op;
use crate::ops::arithmetic::div::div_op;
use crate::ops::arithmetic::mul::{mul_op, mul_op_scalar};
use crate::ops::arithmetic::pow::pow_op;
use crate::ops::arithmetic::max_elemwise::max_elemwise_op; // Import pour AMSGrad

/// Represents the state for a single parameter in the Adam optimizer.
#[derive(Default, Clone, Debug)]
struct AdamState {
    /// First moment vector (exponential moving average of gradients).
    m: Option<Tensor>,
    /// Second moment vector (exponential moving average of squared gradients).
    v: Option<Tensor>,
    /// Maximum value of v_hat seen so far (for AMSGrad).
    v_max: Option<Tensor>, 
}

/// Adam and AdamW Optimizer.
#[derive(Debug)]
pub struct AdamOptimizer {
    param_refs: Vec<Weak<RwLock<Parameter>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    #[allow(dead_code)] // AMSGrad n'est pas implémenté pour le moment
    amsgrad: bool,
    iterations: u64,
    state: HashMap<String, AdamState>,
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

        let param_refs = params.iter().map(Arc::downgrade).collect();

        Ok(AdamOptimizer {
            param_refs,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            iterations: 0,
            state: HashMap::new(),
        })
    }

    /// Returns strong references to the parameters managed by the optimizer.
    pub fn get_params(&self) -> Vec<Arc<RwLock<Parameter>>> {
        self.param_refs.iter().filter_map(Weak::upgrade).collect()
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self) -> Result<(), NeuraRustError> {
        if self.param_refs.is_empty() {
            return Ok(());
        }
        self.iterations += 1;

        for param_weak_ref in &self.param_refs {
            let param_arc = match param_weak_ref.upgrade() {
                Some(arc) => arc,
                None => {
                    eprintln!("Warning: A parameter was dropped and cannot be updated by Adam.");
                    continue;
                }
            };

            let mut param_locked = param_arc
                .write()
                .expect("Failed to lock parameter for writing in Adam step");

            let param_name_opt = param_locked.name();
            let param_name = match param_name_opt {
                Some(name) => name.to_string(),
                None => {
                    let ptr_address = Arc::as_ptr(&param_arc);
                    // Using pointer address as a fallback ID if name is not set.
                    // This is not ideal as pointer addresses can be reused, but better than erroring or skipping.
                    // A more robust solution would be to assign unique IDs to parameters if names are not mandatory.
                    let temp_id = format!("unnamed_param_at_{:?}", ptr_address);
                    eprintln!(
                        "Warning: Parameter at {:?} has no name. Using temporary ID '{}' for Adam optimizer state. Consider naming all parameters.",
                        ptr_address, temp_id
                    );
                    temp_id
                }
            };

            if param_locked.grad().is_none() {
                if self.iterations == 1 { // Log only on first relevant iteration to avoid spam
                    eprintln!(
                        "Warning: Parameter '{}' (Name: {:?}) has no gradient during Adam step. Skipping update.",
                        param_name, param_name_opt
                    );
                }
                continue;
            }

            let grad = param_locked.grad().as_ref().unwrap().clone();
            let param_dtype = grad.dtype(); // Assuming grad has same dtype as param for state init
            
            let state_entry = self
                .state
                .entry(param_name.clone())
                .or_insert_with(AdamState::default);

            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            let m_prev = state_entry
                .m
                .clone()
                .unwrap_or_else(|| zeros_like(&grad).expect("Failed to create zeros_like for m_prev"));
            
            let term1_m = mul_op_scalar(&m_prev, self.beta1)?; 
            let term2_m = mul_op_scalar(&grad, 1.0 - self.beta1)?; 
            let m_t = add_op(&term1_m, &term2_m)?; 
            state_entry.m = Some(m_t.clone());

            // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            let v_prev = state_entry
                .v
                .clone()
                .unwrap_or_else(|| zeros_like(&grad).expect("Failed to create zeros_like for v_prev"));
            
            let grad_sq = mul_op(&grad, &grad)?; 
            let term1_v = mul_op_scalar(&v_prev, self.beta2)?; 
            let term2_v = mul_op_scalar(&grad_sq, 1.0 - self.beta2)?; 
            let v_t = add_op(&term1_v, &term2_v)?; 
            state_entry.v = Some(v_t.clone());

            let bias_correction1 = 1.0 - self.beta1.powi(self.iterations as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.iterations as i32);

            if bias_correction1.abs() < f32::EPSILON { // Check for effective zero
                return Err(NeuraRustError::ArithmeticError(format!(
                    "bias_correction1 is near zero for param '{}' at iteration {}. beta1={}, iterations={}. This can cause division by zero.",
                    param_name, self.iterations, self.beta1, self.iterations
                )));
            }
            if bias_correction2.abs() < f32::EPSILON { // Check for effective zero
                return Err(NeuraRustError::ArithmeticError(format!(
                    "bias_correction2 is near zero for param '{}' at iteration {}. beta2={}, iterations={}. This can cause division by zero.",
                    param_name, self.iterations, self.beta2, self.iterations
                )));
            }

            // m_hat = m_t / bias_correction1
            let m_hat = match param_dtype {
                 DType::F32 => {
                     let bias_tensor = full(&[], bias_correction1)?;
                     div_op(&m_t, &bias_tensor)?
                 },
                 DType::F64 => {
                     let bias_tensor = full_f64(&[], bias_correction1 as f64)?;
                     div_op(&m_t, &bias_tensor)?
                 }
            };
            // v_hat = v_t / bias_correction2
            let v_hat = match param_dtype {
                 DType::F32 => {
                     let bias_tensor = full(&[], bias_correction2)?;
                     div_op(&v_t, &bias_tensor)?
                 },
                 DType::F64 => {
                     let bias_tensor = full_f64(&[], bias_correction2 as f64)?;
                     div_op(&v_t, &bias_tensor)?
                 }
            };

            // --- AMSGrad Logic --- 
            let v_hat_for_update = if self.amsgrad {
                let v_max_prev = state_entry.v_max.clone().unwrap_or_else(|| v_hat.clone()); // Use v_hat if v_max is None
                
                let v_max_t = max_elemwise_op(&v_max_prev, &v_hat)?; 
                
                state_entry.v_max = Some(v_max_t.clone()); // Store the new max
                v_max_t // Use this for the denominator calculation
            } else {
                v_hat // Standard Adam: use v_hat directly
            };

            // sqrt_v_hat = v_hat ^ 0.5
            let sqrt_v_hat = match param_dtype {
                 DType::F32 => {
                    let exponent_tensor = full(&[], 0.5f32)?;
                    pow_op(&v_hat_for_update, &exponent_tensor)?
                 },
                 DType::F64 => {
                    let exponent_tensor = full_f64(&[], 0.5f64)?;
                    pow_op(&v_hat_for_update, &exponent_tensor)?
                 }
            };
            
            // denom = sqrt_v_hat + eps
            let denom = match param_dtype {
                 DType::F32 => {
                     let eps_tensor = full(&[], self.eps)?;
                     add_op(&sqrt_v_hat, &eps_tensor)?
                 },
                 DType::F64 => {
                     let eps_tensor = full_f64(&[], self.eps as f64)?;
                     add_op(&sqrt_v_hat, &eps_tensor)?
                 }
            };

            // adam_component = m_hat / denom
            let adam_component = div_op(&m_hat, &denom)?; 
            
            let final_lr = self.lr; // Effective learning rate (can be modified by schedulers later)

            let param_tensor_mut = param_locked.tensor_mut();
            let mut tensor_data_guard = param_tensor_mut.write_data();
            
            // Read immutable fields BEFORE mutable borrow of buffer
            let is_contiguous = tensor_data_guard.is_contiguous();
            let offset = tensor_data_guard.offset;
            let shape_clone = tensor_data_guard.shape.clone(); // Clone if needed later for error messages
            let strides_clone = tensor_data_guard.strides.clone(); // Clone if needed later
            let dtype_clone = tensor_data_guard.dtype; // DType is Copy

            if !is_contiguous || offset != 0 {
                return Err(NeuraRustError::OptimizerError(format!(
                    "Adam optimizer cannot update param '{}': non-contiguous or offset. Shape: {:?}, Strides: {:?}, Offset: {}. Not supported.",
                    param_name, shape_clone, strides_clone, offset
                )));
            }
            
            let buffer_mut = Arc::make_mut(&mut tensor_data_guard.buffer);

            match dtype_clone { // Use the cloned dtype
                DType::F32 => {
                    let data_slice_mut = buffer_mut.try_get_cpu_f32_mut().map_err(|e| {
                        NeuraRustError::OptimizerError(format!(
                            "Failed to get mutable f32 buffer for param '{}': {}. This might happen if the underlying Vec<f32> within the buffer is shared (e.g., by another Tensor explicitly constructed to share it, which is unusual for parameters).",
                            param_name, e
                        ))
                    })?;

                    // 1. Apply AdamW style weight decay (if enabled)
                    // param_new = param_current * (1 - lr * weight_decay)
                    if self.weight_decay > 0.0 {
                        let decay_factor = 1.0 - final_lr * self.weight_decay;
                        // It's possible for decay_factor to be negative if lr * weight_decay > 1.
                        // Standard libraries usually don't clamp this, but good to be aware.
                        for p_val in data_slice_mut.iter_mut() {
                            *p_val *= decay_factor;
                        }
                    }

                    // 2. Apply the Adam update: param_new = param_current - lr * adam_component
                    let adam_update_values_vec = adam_component.get_f32_data()?;
                    if data_slice_mut.len() != adam_update_values_vec.len() {
                        return Err(NeuraRustError::ShapeMismatch {
                            operation: format!("Adam F32 step update for param '{}'", param_name),
                            expected: format!( "{} elements in parameter buffer (physical length)", data_slice_mut.len()),
                            actual: format!("{} elements in calculated Adam update values (logical length)", adam_update_values_vec.len()),
                        });
                    }
                    for (p_val, u_val) in data_slice_mut.iter_mut().zip(adam_update_values_vec.iter()) {
                        *p_val -= final_lr * u_val;
                    }
                }
                DType::F64 => {
                    let data_slice_mut = buffer_mut.try_get_cpu_f64_mut().map_err(|e| {
                        NeuraRustError::OptimizerError(format!(
                            "Failed to get mutable f64 buffer for param '{}': {}", param_name, e
                        ))
                    })?;

                    if self.weight_decay > 0.0 {
                        let decay_factor_f32 = 1.0 - final_lr * self.weight_decay;
                        let decay_factor_f64 = decay_factor_f32 as f64;
                        for p_val in data_slice_mut.iter_mut() {
                            *p_val *= decay_factor_f64;
                        }
                    }
                    
                    // adam_component should be F64 if param is F64 due to op typing.
                    let adam_update_values_vec = adam_component.get_f64_data()?;
                     if data_slice_mut.len() != adam_update_values_vec.len() {
                        return Err(NeuraRustError::ShapeMismatch {
                            operation: format!("Adam F64 step update for param '{}'", param_name),
                             expected: format!( "{} elements in parameter buffer (physical length)", data_slice_mut.len()),
                            actual: format!("{} elements in calculated Adam update values (logical length)", adam_update_values_vec.len()),
                        });
                    }
                    for (p_val, u_val) in data_slice_mut.iter_mut().zip(adam_update_values_vec.iter()) {
                        *p_val -= final_lr as f64 * u_val; // Ensure lr is also f64 for the final multiplication
                    }
                }
            }
            // param_locked (RwLockWriteGuard for ParameterData) and tensor_data_guard (RwLockWriteGuard for TensorData)
            // are dropped here, releasing the locks.
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param_weak_ref in &self.param_refs {
            if let Some(param_arc) = param_weak_ref.upgrade() {
                let mut param_locked = param_arc
                    .write()
                    .expect("Failed to lock parameter for zero_grad");
                param_locked.zero_grad(); // Calls clear_grad on the internal Tensor
            } else {
                // Optional: Log if a parameter was dropped, though this is less critical for zero_grad.
                // eprintln!("Warning: A parameter was dropped before zero_grad could be called by Adam.");
            }
        }
    }

    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> {
        // TODO: Implement state_dict serialization
        // Should serialize self.iterations and self.state (m and v for each parameter)
        Err(NeuraRustError::UnsupportedOperation(
            "state_dict for AdamOptimizer is not yet implemented".to_string(),
        ))
    }

    fn load_state_dict(&mut self, _state_dict: &OptimizerState) -> Result<(), NeuraRustError> {
        // TODO: Implement load_state_dict deserialization
        // Should deserialize and set self.iterations and self.state
        Err(NeuraRustError::UnsupportedOperation(
            "load_state_dict for AdamOptimizer is not yet implemented".to_string(),
        ))
    }

    fn add_param_group(&mut self, _param_group: crate::optim::param_group::ParamGroup) {
        // This method needs to be present to satisfy the Optimizer trait.
        // For now, AdamOptimizer is initialized with all its parameters at once
        // and does not support adding new parameter groups with different hyperparameters later.
        // This could be a future enhancement.
        unimplemented!("AdamOptimizer does not yet support adding parameter groups after initialization. All parameters must be provided at construction time.");
    }
}

#[cfg(test)]
#[path = "adam_test.rs"]
mod tests; 