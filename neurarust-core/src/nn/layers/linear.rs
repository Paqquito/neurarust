use crate::tensor::Tensor;
use crate::ops::linalg::matmul::matmul_op;
use crate::ops::arithmetic::add_op;
use crate::nn::Parameter;
use crate::error::NeuraRustError;
use crate::nn::module::Module;
use crate::nn::init::{kaiming_uniform_, zeros_};
use crate::types::DType;

use std::fmt::Debug;
// Traits std::ops et autres probablement plus nécessaires directement

/// A fully connected linear layer: y = xA^T + b
#[derive(Debug)]
pub struct Linear {
    in_features: usize,
    _out_features: usize,
    pub weights: Parameter,
    pub bias: Option<Parameter>,
    // On pourrait stocker le DType si nécessaire pour des opérations internes,
    // mais les Tensors eux-mêmes connaissent leur DType.
    // dtype: DType,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias_flag: bool, dtype: DType) -> Result<Self, NeuraRustError> {
        let mut weights = {
            let shape_weights_vec = vec![out_features, in_features];
            match dtype {
                DType::F32 => crate::tensor::zeros(&shape_weights_vec)?,
                DType::F64 => crate::tensor::zeros_f64(&shape_weights_vec)?,
            }
        };
        kaiming_uniform_(&mut weights)?;
        let weights = Parameter::new(weights);

        let bias = if bias_flag {
            let shape_bias_vec = vec![1, out_features];
            let mut bias_tensor = match dtype {
                DType::F32 => crate::tensor::zeros(&shape_bias_vec)?,
                DType::F64 => crate::tensor::zeros_f64(&shape_bias_vec)?,
            };
            zeros_(&mut bias_tensor)?;
            Some(Parameter::new(bias_tensor))
        } else {
            None
        };

        Ok(Linear {
            weights,
            bias,
            in_features,
            _out_features: out_features,
            // dtype,
        })
    }

    /// Retourne une référence au paramètre des poids.
    pub fn weight(&self) -> &Parameter {
        &self.weights
    }

    /// Retourne une référence mutable au paramètre des poids.
    pub fn weight_mut(&mut self) -> &mut Parameter {
        &mut self.weights
    }

    /// Retourne une référence optionnelle au paramètre du biais.
    pub fn bias(&self) -> Option<&Parameter> {
        self.bias.as_ref()
    }

    /// Retourne une référence mutable optionnelle au paramètre du biais.
    pub fn bias_mut(&mut self) -> Option<&mut Parameter> {
        self.bias.as_mut()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        // Vérification de la dimension d'entrée
        if input.shape().last() != Some(&self.in_features) {
            return Err(NeuraRustError::ShapeMismatch {
                expected: format!("... x {}", self.in_features),
                actual: format!("{:?}", input.shape()),
                operation: "Linear forward (input feature dimension)".to_string(),
            });
        }

        let weight_tensor = &self.weights;
        let transposed_weight = weight_tensor.transpose(1, 0)?;
        
        // Utiliser matmul_op importé
        let mut output = matmul_op(input, &transposed_weight)?;

        if let Some(bias_param) = &self.bias {
            let bias_tensor = &bias_param;
            // Garder add_op comme fonction si output.add() n'est pas la bonne approche
            output = add_op(&output, bias_tensor)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        params.push(&self.weights);
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[cfg(test)]
#[path = "linear_test.rs"]
mod tests;
 