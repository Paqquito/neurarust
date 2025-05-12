use crate::tensor::Tensor;
// Les ops sont utilisées avec leur chemin complet, donc pas d'imports directs ici.
use crate::nn::parameter::Parameter; // Rétablir l'import de Parameter
use crate::error::NeuraRustError;
use crate::nn::module::Module;
// kaiming_uniform_ et zeros_ sont bien supprimés car non utilisés.
use crate::types::DType;
use crate::tensor::create::{randn, full, full_f64};
use std::sync::{Arc, RwLock};

use std::fmt::Debug;

/// A fully connected linear layer: y = xA^T + b
#[derive(Debug)]
pub struct Linear {
    // Utiliser Parameter importé, pas le chemin complet ici si importé.
    weight: Arc<RwLock<Parameter>>,
    bias: Option<Arc<RwLock<Parameter>>>,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        dtype: DType,
    ) -> Result<Self, NeuraRustError> {
        let k_val = (1.0 / in_features as f64).sqrt();
        
        let mut weight_data = randn(vec![out_features, in_features])?;
        weight_data = crate::ops::arithmetic::mul::mul_op_scalar(&weight_data, k_val * 2.0)?; 
        
        let neg_k_tensor = match weight_data.dtype() {
            DType::F32 => full(&[], -k_val as f32)?,
            DType::F64 => full_f64(&[], -k_val)?,
            DType::I32 | DType::I64 | DType::Bool => {
                return Err(NeuraRustError::UnsupportedOperation(
                    "Linear::new n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                );
            }
        };
        weight_data = crate::ops::arithmetic::add::add_op(&weight_data, &neg_k_tensor)?;

        weight_data = if weight_data.dtype() != dtype {
            crate::ops::dtype::cast_op(&weight_data, dtype)?
        } else {
            weight_data
        };
        let weight_param = Parameter::new(weight_data, Some("weight".to_string()));
        let weight = Arc::new(RwLock::new(weight_param));

        let bias = if use_bias {
            let mut bias_data = randn(vec![1, out_features])?;
            bias_data = crate::ops::arithmetic::mul::mul_op_scalar(&bias_data, k_val * 2.0)?;
            
            let neg_k_tensor_bias = match bias_data.dtype() {
                DType::F32 => full(&[], -k_val as f32)?,
                DType::F64 => full_f64(&[], -k_val)?,
                DType::I32 | DType::I64 | DType::Bool => {
                    return Err(NeuraRustError::UnsupportedOperation(
                        "Linear::new n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                    );
                }
            };
            bias_data = crate::ops::arithmetic::add::add_op(&bias_data, &neg_k_tensor_bias)?;

            bias_data = if bias_data.dtype() != dtype {
                crate::ops::dtype::cast_op(&bias_data, dtype)?
            } else {
                bias_data
            };
            let bias_param = Parameter::new(bias_data, Some("bias".to_string()));
            Some(Arc::new(RwLock::new(bias_param)))
        } else {
            None
        };

        Ok(Linear { weight, bias })
    }

    /// Retourne une référence au paramètre des poids.
    pub fn weight(&self) -> &Arc<RwLock<Parameter>> {
        &self.weight
    }

    /// Retourne une référence mutable au paramètre des poids.
    pub fn weight_mut(&mut self) -> &mut Arc<RwLock<Parameter>> {
        &mut self.weight
    }

    /// Retourne une référence optionnelle au paramètre du biais.
    pub fn bias(&self) -> Option<&Arc<RwLock<Parameter>>> {
        self.bias.as_ref()
    }

    /// Retourne une référence mutable optionnelle au paramètre du biais.
    pub fn bias_mut(&mut self) -> Option<&mut Arc<RwLock<Parameter>>> {
        self.bias.as_mut()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let weight_guard = self.weight.read()
            .map_err(|e| NeuraRustError::LockError { 
                lock_type: "read".to_string(), 
                reason: format!("Failed to lock weight for reading: {}", e.to_string())
            })?;
        let weight_tensor = &weight_guard.tensor;
        
        let transposed_weight = weight_tensor.transpose(0, 1)?;
        let mut output = crate::ops::linalg::matmul::matmul_op(input, &transposed_weight)?;

        if let Some(bias_arc) = &self.bias {
            let bias_guard = bias_arc.read()
                .map_err(|e| NeuraRustError::LockError { 
                    lock_type: "read".to_string(), 
                    reason: format!("Failed to lock bias for reading: {}", e.to_string())
                })?;
            let bias_tensor = &bias_guard.tensor;
            output = crate::ops::arithmetic::add::add_op(&output, bias_tensor)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        let mut params = Vec::new();
        params.push(Arc::clone(&self.weight));
        if let Some(b) = &self.bias {
            params.push(Arc::clone(b));
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
        let mut params = Vec::new();
        let weight_name = self.weight.read().unwrap().name().unwrap_or("weight").to_string();
        params.push((weight_name, Arc::clone(&self.weight)));
        
        if let Some(b) = &self.bias {
            let bias_name = b.read().unwrap().name().unwrap_or("bias").to_string();
            params.push((bias_name, Arc::clone(b)));
        }
        params
    }

    fn modules(&self) -> Vec<&dyn Module> {
        vec![self] 
    }
    
    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
        f(self);
    }
}

#[cfg(test)]
#[path = "linear_test.rs"]
mod tests;
 