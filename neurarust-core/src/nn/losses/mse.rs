// neurarust-core/src/nn/losses/mse.rs

use crate::tensor::Tensor;
// crate::ops is likely not needed if operations are tensor methods
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::error::NeuraRustError;
use crate::types::DType; // Renommé depuis crate::DType pour la consistance, ou utiliser crate::DType directement
// use crate::ops; // Supprimé
// use std::str::FromStr; // Supprimé
// use crate::autograd::{BackwardOp, NodeId, OpOutputs}; // Ligne supprimée

use std::fmt::Debug;
use std::sync::{Arc, RwLock, Weak};
// Only Neg might be needed if .neg() is a trait method not inherent to Tensor
// use std::ops::Neg; // Import std::ops::Neg supprimé/commenté

// Importer les opérations nécessaires
use crate::ops::arithmetic::{sub_op, mul_op, div_op}; // neg_op retiré
// Ne pas importer mean_all/sum_all, utiliser les méthodes de Tensor

/// Specifies the reduction to apply to the output:
/// 'none' | 'mean' | 'sum'
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reduction {
    Mean,
    Sum,
    // None, // TODO: Consider if 'None' (no reduction) is needed
}

impl Reduction {
    pub fn from_str(s: &str) -> Result<Self, NeuraRustError> {
        match s.to_lowercase().as_str() {
            "mean" => Ok(Reduction::Mean),
            "sum" => Ok(Reduction::Sum),
            // "none" => Ok(Reduction::None),
            _ => Err(NeuraRustError::UnsupportedOperation(format!("Unsupported reduction type: {}", s))),
        }
    }
}

/// Computes the Mean Squared Error (MSE) loss between input and target tensors.
///
/// The loss can be configured to compute the mean or sum of squared errors.
///
/// # Fields
/// * `reduction`: Specifies the type of reduction to apply to the output: `Mean` or `Sum`.
/// * `cached_input_shape`: Optionally stores the shape of the input tensor from the last forward pass.
///                         Used for validation in the backward pass.
/// * `cached_target_shape`: Optionally stores the shape of the target tensor from the last forward pass.
///                          Used for validation in the backward pass.
#[derive(Debug, Clone)]
pub struct MSELoss {
    reduction: Reduction,
    // We don't need to store shapes if TensorData holds them and we have pointers
    // cached_input_shape: Option<Vec<usize>>,
    // cached_target_shape: Option<Vec<usize>>,
}

impl MSELoss {
    /// Creates a new `MSELoss` module.
    ///
    /// # Arguments
    /// * `reduction`: The reduction method to apply (`Mean` or `Sum`).
    ///
    /// # Panics
    /// Panics if an unsupported reduction type string is provided (should be an error).
    //TODO: Change to Result later
    pub fn new(reduction_str: &str) -> Self {
        let reduction = Reduction::from_str(reduction_str)
            .unwrap_or_else(|e| panic!("Failed to create MSELoss: {}", e)); // Or handle error appropriately
        MSELoss {
            reduction,
            // cached_input_shape: None,
            // cached_target_shape: None,
        }
    }

    // Renommer en `calculate` ou garder `forward` comme méthode spécifique à la perte
    pub fn calculate(&self, input: &Tensor, target: &Tensor) -> Result<Tensor, NeuraRustError> {
        if input.shape() != target.shape() {
            return Err(NeuraRustError::ShapeMismatch {
                expected: format!("{:?}", target.shape()),
                actual: format!("{:?}", input.shape()),
                operation: "MSELoss calculate".to_string(),
            });
        }

        // input.set_requires_grad(true); // Should be set by the user if input is a parameter or needs grad

        let diff = sub_op(input, target)?;
        let squared_diff = mul_op(&diff, &diff)?;

        // Utiliser les méthodes mean/sum de Tensor pour la réduction globale
        // Supposant que None pour axes signifie réduction globale
        let loss_val_tensor = match self.reduction {
            Reduction::Mean => squared_diff.mean(None, false)?,
            Reduction::Sum => squared_diff.sum(None, false)?,
        };

        // If either input or target requires grad, set up for backward pass
        if input.requires_grad() || target.requires_grad() {
            let grad_fn = MSEBackward {
                input: Arc::downgrade(&input.data),
                target: Arc::downgrade(&target.data),
                reduction: self.reduction.clone(),
            };
            loss_val_tensor.set_grad_fn(Some(Arc::new(grad_fn) as Arc<dyn BackwardOp>))?;
        }
        
        Ok(loss_val_tensor)
    }
}

// Placeholder for MSEBackward. This will be heavily modified.
#[derive(Debug)]
struct MSEBackward {
    input: Weak<RwLock<TensorData>>,
    target: Weak<RwLock<TensorData>>,
    reduction: Reduction,
}

impl BackwardOp for MSEBackward {
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut parent_ptrs = Vec::with_capacity(2);
        // It's critical that arcs are available. If not, graph is broken.
        // Panicking here makes sense if the invariant (parents live longer than child's grad_fn) is broken.
        match self.input.upgrade() {
            Some(input_arc) => parent_ptrs.push(Arc::as_ptr(&input_arc) as *const RwLock<TensorData>),
            None => panic!("MSEBackward: Input tensor weak reference expired during inputs() call. Graph integrity likely compromised."),
        }
        match self.target.upgrade() {
            Some(target_arc) => parent_ptrs.push(Arc::as_ptr(&target_arc) as *const RwLock<TensorData>),
            None => panic!("MSEBackward: Target tensor weak reference expired during inputs() call. Graph integrity likely compromised."),
        }
        parent_ptrs
    }

    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let input_arc = self.input.upgrade().ok_or_else(|| 
            NeuraRustError::BackwardError("Input tensor weak reference expired for MSEBackward backward pass.".to_string())
        )?;
        let target_arc = self.target.upgrade().ok_or_else(|| 
            NeuraRustError::BackwardError("Target tensor weak reference expired for MSEBackward backward pass.".to_string())
        )?;

        let input_tensor = Tensor { data: input_arc };
        let target_tensor = Tensor { data: target_arc };

        if !(grad_output.shape().is_empty() || grad_output.shape() == [1] || grad_output.numel() == 1) {
            return Err(NeuraRustError::ShapeMismatch {
                expected: "scalar (shape [] or [1])".to_string(),
                actual: format!("{:?}", grad_output.shape()),
                operation: "MSEBackward: grad_output must be scalar".to_string(),
            });
        }

        let diff = sub_op(&input_tensor, &target_tensor)?;

        let dtype = diff.dtype();
        
        let two_scalar = match dtype {
            DType::F32 => crate::tensor::create::full(&[], 2.0f32)?,
            DType::F64 => crate::tensor::create::full_f64(&[], 2.0f64)?,
            DType::I32 | DType::I64 | DType::Bool => todo!("mse: non supporté pour ce DType (two_scalar)"),
        };
        
        let common_term = mul_op(&diff, &two_scalar)?;
        let grad_input_unscaled = mul_op(grad_output, &common_term)?;

        let final_grad_input = if self.reduction == Reduction::Mean {
            let num_elements = input_tensor.numel();
            if num_elements == 0 {
                return Err(NeuraRustError::InternalError("MSEBackward: Number of elements is zero, cannot take mean.".to_string()));
            }
            let num_elements_scalar = match dtype {
                DType::F32 => crate::tensor::create::full(&[], num_elements as f32)?,
                DType::F64 => crate::tensor::create::full_f64(&[], num_elements as f64)?,
                DType::I32 | DType::I64 | DType::Bool => todo!("mse: non supporté pour ce DType (num_elements_scalar)"),
            };
            div_op(&grad_input_unscaled, &num_elements_scalar)?
        } else { // Reduction::Sum
            grad_input_unscaled
        };

        let minus_one_scalar = match dtype {
            DType::F32 => crate::tensor::create::full(&[], -1.0f32)?,
            DType::F64 => crate::tensor::create::full_f64(&[], -1.0f64)?,
            DType::I32 | DType::I64 | DType::Bool => todo!("mse: non supporté pour ce DType (minus_one_scalar)"),
        };
        let final_grad_target = mul_op(&final_grad_input, &minus_one_scalar)?;
        // Ou si neg() existe : let final_grad_target = final_grad_input.neg()?;

        Ok(vec![final_grad_input, final_grad_target])
    }
}

// Note: The original test module is removed from this file.
// It is expected to be in mse_test.rs.
#[cfg(test)]
#[path = "mse_test.rs"]
mod tests; 