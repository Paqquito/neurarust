use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use crate::types::DType;
use crate::NeuraRustError;

/// Clips gradient of an iterable of parameters inplace.
///
/// The gradients are clipped element-wise in the range `[-clip_value, clip_value]`.
///
/// # Arguments
///
/// * `parameters`: An iterator over mutable references to `Parameter`s.
/// * `clip_value`: The maximum absolute value for each element in the gradients.
///
/// # Errors
///
/// Returns `NeuraRustError::InvalidInput` if a gradient tensor is not of a floating-point type (F32 or F64).
/// Propagates errors from underlying tensor operations (e.g., `clamp_`).
pub fn clip_grad_value_<'a>(
    parameters: impl Iterator<Item = &'a mut Parameter>,
    clip_value: f32,
) -> Result<(), NeuraRustError> {
    if clip_value < 0.0 {
        return Err(NeuraRustError::ConfigurationError(
            "clip_value must be non-negative".to_string(),
        ));
    }

    for param in parameters {
        // param est &mut Parameter. Grâce à DerefMut<Target=Tensor>, 
        // on peut le traiter comme &mut Tensor pour accéder à ses champs internes comme `data`.
        // Le champ `data` dans Tensor est `pub(crate) data: Arc<RwLock<TensorData>>`
        let tensor_data_arc = param.data.clone(); // Cloner l'Arc pour obtenir la possession temporaire pour le write lock

        let mut tensor_data_guard = tensor_data_arc.write().map_err(|_e| {
            NeuraRustError::LockError {
                lock_type: "write".to_string(),
                reason: format!("Failed to acquire write lock on TensorData for parameter {:?}", param.name.as_deref().unwrap_or("<unnamed>")),
            }
        })?;

        // tensor_data_guard est maintenant un RwLockWriteGuard<TensorData>
        // Accéder au champ grad de TensorData
        if let Some(grad_tensor) = tensor_data_guard.grad.as_mut() {
            let dtype = grad_tensor.dtype();
            if dtype == DType::F32 {
                grad_tensor.clamp_(Some(-clip_value), Some(clip_value))?;
            } else if dtype == DType::F64 {
                grad_tensor.clamp_(Some(-clip_value as f64), Some(clip_value as f64))?;
            } else {
                // Ce cas sera atteint lorsque DType sera étendu (I32, Bool, etc.)
                return Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32, // Ou une description plus générique indiquant un type flottant
                    actual: dtype,
                    operation: "clip_grad_value_".to_string(),
                });
            }
        }
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
pub fn clip_grad_norm_<'a>(
    parameters: impl Iterator<Item = &'a mut Parameter> + Clone, // Clone needed for second pass
    max_norm: f32,
    norm_type: f32,
) -> Result<f32, NeuraRustError> { // Return the total norm
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

    // First pass: calculate the total norm
    let mut total_norm_pow_p: f32 = 0.0;

    let mut grads_to_process: Vec<Tensor> = Vec::new();

    for param in parameters.clone() { 
        let tensor_data_arc = param.data.clone();
        let tensor_data_guard = tensor_data_arc.read().map_err(|_e| {
            NeuraRustError::LockError {
                lock_type: "read".to_string(),
                reason: format!("Failed to acquire read lock on TensorData for norm calculation on param {:?}", param.name.as_deref().unwrap_or("<unnamed>")),
            }
        })?;

        if let Some(grad_tensor) = tensor_data_guard.grad.as_ref() {
            grads_to_process.push(grad_tensor.clone());
        }
    }

    for grad_tensor in grads_to_process.iter() {
        let dtype = grad_tensor.dtype();
        if dtype == DType::F32 {
            let data = grad_tensor.get_f32_data()?;
            for val in data {
                total_norm_pow_p += val.abs().powf(norm_type);
            }
        } else if dtype == DType::F64 {
            let data = grad_tensor.get_f64_data()?;
            for val in data {
                total_norm_pow_p += (val.abs() as f32).powf(norm_type); // Cast to f32 for consistent sum
            }
        } else {
            return Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32, 
                actual: dtype,
                operation: "clip_grad_norm_ (norm calculation)".to_string(),
            });
        }
    }

    let total_norm = total_norm_pow_p.powf(1.0 / norm_type);

    if total_norm.is_nan() || total_norm.is_infinite() {
        // Si total_norm est NaN ou Inf (par exemple, si tous les gradients sont nuls et norm_type > 1, 0^inf -> 0. Si norm_type=0 -> ?)
        // Si total_norm_pow_p est 0.0, et 1.0/norm_type est aussi 0 (norm_type -> infini), alors 0.0^0.0 -> NaN
        // Si tous les grads sont 0, total_norm_pow_p = 0. total_norm = 0.powf(1/p). Si 1/p > 0, total_norm = 0. Sinon NaN/Inf.
        // Si total_norm_pow_p = 0, total_norm sera 0.0, sauf si 1.0/norm_type est <= 0 ou NaN.
        // norm_type est positif, donc 1.0/norm_type est positif.
        // 0.0.powf(positive) est 0.0. Un NaN ici signifierait que total_norm_pow_p était NaN.
        // Si total_norm_pow_p devient NaN (ex: val.abs().powf(norm_type) est NaN), alors total_norm sera NaN.
        // Si total_norm est 0.0 et max_norm est aussi 0.0, clip_coef sera NaN (0.0 / 1e-6). Doit être géré.
        // Si total_norm est très petit mais > 0, et max_norm = 0, clip_coef = 0.
    }

    if total_norm > max_norm {
        // Gérer le cas où total_norm est zéro pour éviter la division par zéro si max_norm > 0.
        // Si total_norm est très proche de zéro, total_norm + 1e-6 évite la division par zéro stricte.
        // Si total_norm est NaN ou Inf, clip_coef peut aussi devenir NaN/Inf ou 0.
        // PyTorch: clip_coef = max_norm / (total_norm + 1e-6). total_norm est calculé comme (sum(p.grad.data.pow(norm_type) for p in parameters)) .pow(1./norm_type)
        // Si la norme est Inf, le coefficient devient 0, ce qui est correct (annule les gradients Inf).
        let clip_coef = if total_norm.is_finite() && total_norm > 0.0 { max_norm / (total_norm + 1e-6) } else { 1.0 }; // Ne pas clipper si total_norm est 0 ou non-fini (NaN/Inf), sauf si max_norm est 0.
        // Si max_norm est 0, clip_coef devrait être 0 pour annuler tous les gradients.
        let final_clip_coef = if max_norm == 0.0 { 0.0 } else { clip_coef };
        
        // Si total_norm est Inf et max_norm est fini, PyTorch met à l'échelle par 0. 
        // Si total_norm est NaN, PyTorch ne fait rien.
        // Notre logique actuelle avec `total_norm + 1e-6` : max_norm / Inf -> 0. max_norm / NaN -> NaN.
        // Si total_norm est 0, clip_coef = max_norm / 1e-6. Si max_norm est aussi 0, alors 0.
        // La condition `total_norm > max_norm` gère le cas où total_norm est 0 (0 > max_norm est faux sauf si max_norm < 0, ce qui est vérifié).
        
        if final_clip_coef < 1.0 { // Ne scale que si le coefficient réduit la norme
            for param in parameters { 
                let tensor_data_arc = param.data.clone();
                let mut tensor_data_guard = tensor_data_arc.write().map_err(|_e| {
                    NeuraRustError::LockError {
                        lock_type: "write".to_string(),
                        reason: format!("Failed to acquire write lock on TensorData for scaling on param {:?}", param.name.as_deref().unwrap_or("<unnamed>")),
                    }
                })?;

                if let Some(grad_tensor) = tensor_data_guard.grad.as_mut() {
                    let dtype = grad_tensor.dtype();
                    if dtype == DType::F32 {
                        grad_tensor.mul_scalar_f32(final_clip_coef)?;
                    } else if dtype == DType::F64 {
                        grad_tensor.mul_scalar_f64(final_clip_coef as f64)?;
                    } else { 
                        // Ce cas ne devrait pas être atteint si la première boucle de calcul de la norme a réussi
                        // et a vérifié les types de données. Par sécurité :
                        return Err(NeuraRustError::DataTypeMismatch {
                            expected: DType::F32, 
                            actual: dtype,
                            operation: "clip_grad_norm_ (scaling)".to_string(),
                        });
                    }
                }
            }
        }
    }
    Ok(total_norm)
} 