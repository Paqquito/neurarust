use crate::error::NeuraRustError;
use crate::nn::parameter::Parameter;
use crate::types::DType;
// use crate::ops::traits::numeric::NeuraNumeric; // Supprimé car plus utilisé
use std::sync::{Arc, RwLock};
use crate::buffer::{Buffer, CpuBuffer};
use crate::tensor::Tensor; // Ajout car on va créer de nouveaux tenseurs
use crate::ops::arithmetic::mul::mul_op_scalar; // Pour clip_grad_norm_
use crate::tensor::create::{from_vec_f32, from_vec_f64}; // Pour reconstruire les tenseurs clampés

/// Clips gradient of an iterable of parameters inplace.
///
/// The gradients are clipped element-wise in the range `[-clip_value, clip_value]`.
///
/// # Arguments
///
/// * `parameters`: An iterator over `Arc<RwLock<Parameter>>`s.
/// * `clip_value`: The maximum absolute value for each element in the gradients.
///
/// # Errors
///
/// Returns `NeuraRustError` if `clip_value` is negative, a lock cannot be acquired,
/// a gradient tensor is not of a floating-point type (F32 or F64),
/// or an underlying tensor operation fails.
pub fn clip_grad_value_<P>(parameters: P, clip_value: f64) -> Result<(), NeuraRustError>
where
    P: Iterator<Item = Arc<RwLock<Parameter>>>,
{
    if clip_value < 0.0 {
        return Err(NeuraRustError::ConfigurationError(
            "clip_value must be non-negative".to_string(),
        ));
    }

    for param_arc in parameters {
        let param_guard = param_arc.write().map_err(|_e| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: "Failed to acquire write lock on Parameter in clip_grad_value_".to_string(),
        })?;

        let tensor_data_arc = param_guard.tensor.data.clone(); 
        let mut tensor_data_guard = tensor_data_arc.write().map_err(|_e| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: format!("Failed to acquire write lock on TensorData for param {:?} in clip_grad_value_", param_guard.name().unwrap_or_default()),
        })?;

        if let Some(grad_tensor) = tensor_data_guard.grad.take() { // take() pour pouvoir le remplacer
            let dtype = grad_tensor.dtype();
            let shape = grad_tensor.shape().to_vec(); // Cloné pour le nouveau tenseur
            // device est CPU car from_vec_f32/f64 créent sur CPU. Si le grad original était sur GPU, il faudrait adapter.
            // Pour l'instant, on assume CPU pour les gradients ou que le device original sera respecté par une future API plus riche.

            let new_grad_tensor = match dtype {
                DType::F32 => {
                    let min_f32 = -clip_value as f32;
                    let max_f32 = clip_value as f32;
                    let current_data = grad_tensor.get_f32_data()?;
                    let clamped_data: Vec<f32> = current_data.into_iter().map(|x| x.clamp(min_f32, max_f32)).collect();
                    from_vec_f32(clamped_data, shape)?
                }
                DType::F64 => {
                    let min_f64 = -clip_value;
                    let max_f64 = clip_value;
                    let current_data = grad_tensor.get_f64_data()?;
                    let clamped_data: Vec<f64> = current_data.into_iter().map(|x| x.clamp(min_f64, max_f64)).collect();
                    from_vec_f64(clamped_data, shape)?
                }
                DType::I32 | DType::I64 | DType::Bool => {
                    return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32,
                        actual: dtype,
                        operation: "clip_grad_value_".to_string(),
                    });
                }
            };
            tensor_data_guard.grad = Some(new_grad_tensor);
        } // grad_tensor est droppé ici s'il a été pris
    }
    Ok(())
}

/// Clips the overall norm of gradients of an iterable of parameters inplace.
///
/// The gradients are viewed as a single concatenated vector, and if its total norm
/// exceeds `max_norm`, all gradients are scaled down by a common factor.
///
/// # Arguments
///
/// * `parameters`: An iterator over mutable references to `Parameter`s.
/// * `max_norm`: The maximum allowed norm for the combined gradients.
/// * `norm_type`: The p-norm to use for calculating the total norm (e.g., 2.0 for L2 norm).
///                Defaults to 2.0 if not specified or if an invalid value (e.g., <= 0) is given,
///                though for simplicity, we might enforce positive norm_type.
///
/// # Errors
///
/// Returns `NeuraRustError` if `max_norm` is negative, `norm_type` is not positive,
/// or if an underlying tensor operation fails, or if gradients are not F32/F64.
pub fn clip_grad_norm_<P>(
    parameters: P,
    max_norm: f64,
    norm_type: f64,
) -> Result<f64, NeuraRustError>
where
    P: Iterator<Item = Arc<RwLock<Parameter>>> + Clone,
{
    if max_norm < 0.0 {
        return Err(NeuraRustError::ConfigurationError(
            "max_norm must be non-negative".to_string(),
        ));
    }
    if norm_type <= 0.0 {
        return Err(NeuraRustError::ConfigurationError(
            "norm_type must be positive".to_string(),
        ));
    }

    let mut total_norm_pow_p: f64 = 0.0;
    let mut params_with_grads_info: Vec<(Arc<RwLock<Parameter>>, Tensor)> = Vec::new(); // Stocker Arc et clone du grad

    for param_arc in parameters.clone() { 
        let param_guard = param_arc.read().map_err(|_e| NeuraRustError::LockError {
            lock_type: "read".to_string(),
            reason: "Failed to acquire read lock on Parameter for norm calculation in clip_grad_norm_".to_string(),
        })?;
        let param_name_for_error_display = param_guard.name().unwrap_or_default();

        if let Some(grad_tensor) = param_guard.tensor.grad().as_ref() {
            // Vérifier le type de données immédiatement
            match grad_tensor.dtype() {
                DType::I32 | DType::I64 | DType::Bool => {
                    return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32,
                        actual: grad_tensor.dtype(),
                        operation: "clip_grad_norm_".to_string(),
                    });
                }
                _ => {}
            }

            // Stocker l'Arc et un clone du grad pour la deuxième passe
            // Cloner le tenseur grad ici pour éviter des problèmes de double emprunt mutable plus tard
            // si on essayait de lire le grad à nouveau dans la boucle de scaling.
            params_with_grads_info.push((param_arc.clone(), grad_tensor.clone())); 

            let data_guard = grad_tensor.data.read().map_err(|_e| NeuraRustError::LockError {
                lock_type: "read".to_string(),
                reason: format!(
                    "Failed to acquire read lock on grad TensorData for param {}", 
                    param_name_for_error_display
                ),
            })?;

            let buffer_data = match &*data_guard.buffer {
                Buffer::Cpu(CpuBuffer::F32(data_arc)) => data_arc.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
                Buffer::Cpu(CpuBuffer::F64(data_arc)) => data_arc.iter().copied().collect::<Vec<f64>>(),
                _ => return Err(NeuraRustError::UnsupportedDevice { 
                    device: data_guard.device, 
                    operation: format!("clip_grad_norm_ (norm calculation) on param {}", param_name_for_error_display)
                }),
            };

            if norm_type.is_infinite() && norm_type.is_sign_positive() { 
                for val in buffer_data {
                    if val.abs() > total_norm_pow_p {
                        total_norm_pow_p = val.abs();
                    }
                }
            } else { 
                for val in buffer_data {
                    total_norm_pow_p += val.abs().powf(norm_type);
                }
            }
        }
    }

    let total_norm = if norm_type.is_infinite() && norm_type.is_sign_positive() {
        total_norm_pow_p
    } else {
        total_norm_pow_p.powf(1.0 / norm_type)
    };

    if total_norm.is_nan() || total_norm == 0.0 { 
        return Ok(total_norm);
    }
    
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);

        if clip_coef < 1.0 { 
            for (param_arc, original_grad_clone) in params_with_grads_info { // Utilise les infos stockées
                // Pas besoin de relire param_guard ici pour obtenir le grad, on a original_grad_clone
                // Mais on a besoin de param_guard pour écrire le nouveau grad
                let param_guard = param_arc.write().map_err(|_e| NeuraRustError::LockError {
                    lock_type: "write".to_string(),
                    reason: "Failed to acquire write lock on Parameter for scaling in clip_grad_norm_".to_string(),
                })?;
                // let param_name_for_error_display_scaling = param_guard.name().unwrap_or_default();

                let scaled_grad = match original_grad_clone.dtype() {
                    DType::F32 => {
                        let clip_coef_f32 = clip_coef as f32;
                        mul_op_scalar(&original_grad_clone, clip_coef_f32)?
                    }
                    DType::F64 => {
                        mul_op_scalar(&original_grad_clone, clip_coef)?
                    }
                    DType::I32 | DType::I64 | DType::Bool => {
                        return Err(NeuraRustError::DataTypeMismatch {
                            expected: DType::F32,
                            actual: original_grad_clone.dtype(),
                            operation: "clip_grad_norm_".to_string(),
                        });
                    }
                };
                
                // Mettre à jour le gradient dans le paramètre
                let mut tensor_data_guard = param_guard.tensor.data.write().map_err(|_e| NeuraRustError::LockError{
                    lock_type: "write".to_string(),
                    reason: "Failed to write TensorData for scaled grad".to_string()
                })?;
                tensor_data_guard.grad = Some(scaled_grad);
            }
        }
    }
    Ok(total_norm)
} 