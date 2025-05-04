use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use std::fmt::Debug;
use thiserror::Error;
use std::sync::{Arc, RwLock};
// Remove unused approx traits for now
// use approx::{AbsDiffEq, RelativeEq, UlpsEq};
// Remove unused num_traits
// use num_traits::{Float, One, Signed, Zero};
// Remove unused Arc
// use std::sync::Arc;

// Import specific ops needed for calculate_loss
use crate::ops::reduction::sum::sum_op; // Use sum_op for loss calculation
use crate::ops::arithmetic::mul_op; // Import corrigé pour mul_op
use crate::types::DType;
use crate::device::StorageDevice;
use crate::tensor_data::TensorData;
use crate::buffer::{Buffer, CpuBuffer};

/// Error type specifically for gradient checking failures.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GradCheckError {
    #[error("Gradient check failed for input tensor at index {input_index}, element index {element_index}: Analytical grad {analytical_grad:?} != Numerical grad {numerical_grad:?}. Difference: {difference:?}")]
    GradientMismatch {
        input_index: usize,
        element_index: usize,
        analytical_grad: f64, // Use f64 for precision
        numerical_grad: f64,
        difference: f64,
    },

    // Keep other error variants as defined before...
    #[error("Failed to get mutable data for input tensor {input_index} at element {element_index}: {source}")]
    MutationError {
        input_index: usize,
        element_index: usize,
        source: NeuraRustError,
    },
    #[error("Forward function execution failed during gradient check: {0}")]
    ForwardPassError(NeuraRustError),
    #[error("Backward pass execution failed during gradient check: {0}")]
    BackwardPassError(NeuraRustError),
    #[error("Could not access analytical gradient for input {input_index}: {source}")]
    AnalyticalGradAccessError {
        input_index: usize,
        source: NeuraRustError,
    },
    #[error("Tensor error during intermediate calculation: {0}")]
    TensorError(NeuraRustError),
    #[error("Unsupported data type for gradient check: expected F32, got {0:?}")]
    UnsupportedDType(DType),
    #[error("Input tensor {input_index} requires grad but has no gradient after backward pass.")]
    MissingAnalyticalGrad { input_index: usize },
    #[error("Numerical gradient is NaN or infinite for input {input_index}, element {element_index}. Details: Loss+: {loss_plus:?}, Loss-: {loss_minus:?}")]
    NumericalGradNaNOrInfinite {
        input_index: usize,
        element_index: usize,
        loss_plus: f64,
        loss_minus: f64,
    },
    #[error("Analytical gradient is NaN or infinite for input {input_index}, element {element_index}. Value: {value:?}")]
    AnalyticalGradNaNOrInfinite {
        input_index: usize,
        element_index: usize,
        value: f64,
    },
    #[error("Gradient checking on non-contiguous tensors not yet supported (Input {input_index}).")]
    NonContiguousInput { input_index: usize },
    #[error("Gradient checking only supported on CPU tensors (Input {input_index}). Got: {device:?}")]
    NonCpuInput {
        input_index: usize,
        device: StorageDevice,
    },
    #[error("Gradient check input tensor must be a leaf node (no grad_fn). Input index: {input_index}")]
    InputNotLeaf { input_index: usize },
    #[error("Function did not propagate requires_grad correctly.")]
    RequiresGradPropagationError,

}

// Map NeuraRustError to GradCheckError::TensorError
impl From<NeuraRustError> for GradCheckError {
    fn from(err: NeuraRustError) -> Self {
        GradCheckError::TensorError(err)
    }
}

/// Checks analytical gradients against numerical gradients using finite differences.
///
/// Compares the analytical gradient computed by the backward pass against a numerical
/// approximation computed using finite differences: `num_grad ≈ [L(x+eps) - L(x-eps)] / (2*eps)`,
/// where `L = sum(output * output_grad)` is a scalar loss.
///
/// The comparison uses both absolute and relative tolerances:
/// `|analytical_grad - numerical_grad| <= abs_tol + rel_tol * |numerical_grad|`
///
/// # Arguments
///
/// * `func`: The function (often a closure) whose gradient is being checked. It takes a slice
///   of input `Tensor`s and returns a `Result<Tensor, NeuraRustError>`.
/// * `inputs`: A slice of input `Tensor`s provided to `func`. Tensors requiring grad must be leaf nodes.
/// * `output_grad`: The gradient flowing into the output of `func` (often a tensor of ones).
/// * `epsilon`: The small perturbation used for finite differences (e.g., `1e-5`). Should be `f64`.
/// * `abs_tol`: The absolute tolerance for the gradient comparison (e.g., `1e-7`). Should be `f64`.
/// * `rel_tol`: The relative tolerance for the gradient comparison (e.g., `1e-5`). Should be `f64`.
///
/// # Errors
///
/// Returns `GradCheckError` if the check fails or an error occurs during computation.
// Remove T generic
pub fn check_grad<F>(
    func: F,
    inputs: &[Tensor], // Use Tensor
    output_grad: &Tensor, // Use Tensor
    epsilon: f64,
    // tolerance: f64, // Remplacé par abs_tol et rel_tol
    abs_tol: f64,
    rel_tol: f64,
) -> Result<(), GradCheckError>
where
    // Update F signature
    F: Fn(&[Tensor]) -> Result<Tensor, NeuraRustError>,
{
    // --- Constants ---
    let two = 2.0f64;

    // --- Initial Checks ---
    for (i, input) in inputs.iter().enumerate() {
        let dtype = input.dtype();
        let device = input.device();
        if dtype != DType::F32 {
            return Err(GradCheckError::UnsupportedDType(dtype));
        }
        if device != StorageDevice::CPU {
            return Err(GradCheckError::NonCpuInput { input_index: i, device });
        }
        if !input.is_contiguous() {
            return Err(GradCheckError::NonContiguousInput { input_index: i });
        }
         // Ensure inputs requiring grad are leaf nodes
        if input.requires_grad() && input.read_data().grad_fn.is_some() {
            return Err(GradCheckError::InputNotLeaf { input_index: i });
        }
    }
    let output_grad_dtype = output_grad.dtype();
    if output_grad_dtype != DType::F32 {
         return Err(GradCheckError::UnsupportedDType(output_grad_dtype));
    }
    let output_grad_device = output_grad.device();
    if output_grad_device != StorageDevice::CPU {
        // Error if output_grad is not CPU
         return Err(GradCheckError::NonCpuInput { input_index: usize::MAX, device: output_grad_device }); // Use usize::MAX for output_grad index
    }

    // --- 1. Initial Forward and Backward Pass ---
    let initial_inputs: Vec<Tensor> = inputs.iter().map(|t| t.clone()).collect();

    // Clear grads before forward/backward
    for input in initial_inputs.iter() {
        if input.requires_grad() {
            input.clear_grad(); // Use helper method
        }
    }

    let output = func(&initial_inputs).map_err(GradCheckError::ForwardPassError)?;

    // Check requires_grad propagation
    let any_input_requires_grad = inputs.iter().any(|t| t.requires_grad());
    if any_input_requires_grad && !output.requires_grad() {
        return Err(GradCheckError::RequiresGradPropagationError);
    }

    // Perform backward pass to get analytical gradients
    if output.requires_grad() {
        output
            .backward(Some(output_grad.clone()))
            .map_err(GradCheckError::BackwardPassError)?;
    } // else: no backward pass needed/possible

    // Store analytical gradients (as Option<Tensor>)
    let analytical_grads_opt: Vec<Option<Tensor>> = initial_inputs.iter()
                                                               .map(|t| t.grad())
                                                               .collect();

    // --- 3. Iterate through Inputs ---
    for (i, original_input) in inputs.iter().enumerate() {
        if !original_input.requires_grad() {
            continue; // Skip inputs that don't require grad
        }

        // --- 4. Get Analytical Gradient Data (as f64) ---
        let analytical_grad_tensor = match analytical_grads_opt[i].as_ref() {
            Some(grad) => grad,
            None => {
                 return Err(GradCheckError::MissingAnalyticalGrad{ input_index: i });
            }
        };
        // Extraire les données du gradient analytique (DOIT être contigu pour get_f32_data)
        let analytical_grad_data: Vec<f64> = analytical_grad_tensor
            .contiguous()? // S'assurer que le GRADIENT est contigu pour l'extraction
            .get_f32_data()? 
            .iter()
            .map(|&x| x as f64)
            .collect();

        // --- 5. Iterate through Elements (Logical) and Calculate Numerical Gradient ---
        let numel = original_input.numel();
        let rank = original_input.shape().len();
        let original_shape = original_input.shape(); // Obtenir shape une fois
        let original_strides = original_input.strides(); // Obtenir strides une fois
        let original_offset = original_input.read_data().offset; // Obtenir offset une fois
        let original_device = original_input.device(); // Should be CPU based on initial checks
        
        // Obtenir le buffer original (Arc<Vec<f32>>) une fois
        let original_buffer_arc = original_input.read_data().buffer().try_get_cpu_f32()?.clone();

        for elem_idx in 0..numel { // elem_idx représente l'index logique linéaire
            
            // --- 5.1 Calculer l'offset physique --- 
            let mut current_logical_indices = vec![0; rank];
            let mut current_linear = elem_idx;
            for dim in (0..rank).rev() {
                 let shape_val = original_shape[dim];
                 if shape_val > 0 { current_logical_indices[dim] = current_linear % shape_val; current_linear /= shape_val; } else { current_logical_indices[dim] = 0; }
            }
            let physical_offset = original_offset + current_logical_indices.iter().zip(original_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();

            // --- 5.2 Calculate Loss for f(x + eps) --- 
            let loss_plus = {
                let mut inputs_plus = inputs.iter().map(|t| t.clone()).collect::<Vec<_>>();
                
                // --- 5.2.1 Créer le tenseur perturbé (+) vue --- 
                let buffer_plus_arc = {
                    let mut buffer_vec = original_buffer_arc.as_ref().clone(); 
                    if physical_offset >= buffer_vec.len() { return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Offset out of bounds (+) perturbation".to_string()))); }
                    
                    // --- Modification pour calcul en f64 --- 
                    let current_val_f32 = buffer_vec[physical_offset];
                    let current_val_f64 = current_val_f32 as f64;
                    let perturbed_val_f64 = current_val_f64 + epsilon; // Calcul en f64
                    let perturbed_val_f32 = perturbed_val_f64 as f32; // Reconversion en f32
                    // ---------------------------------------

                    buffer_vec[physical_offset] = perturbed_val_f32; // Ecriture f32
                    Arc::new(buffer_vec) 
                };
                let td_plus = TensorData::new_view(
                    Arc::new(Buffer::Cpu(CpuBuffer::F32(buffer_plus_arc))),
                    original_device, // Utiliser le device original
                    original_offset, 
                    original_shape.clone(), 
                    original_strides.clone(),
                );
                let perturbed_tensor_plus = Tensor { data: Arc::new(RwLock::new(td_plus)) };

                inputs_plus[i] = perturbed_tensor_plus;
                let output_plus = func(&inputs_plus).map_err(GradCheckError::ForwardPassError)?;
                calculate_loss(&output_plus, output_grad)? // Renvoie f64
            };

            // --- 5.3 Calculate Loss for f(x - eps) --- 
            let loss_minus = {
                let mut inputs_minus = inputs.iter().map(|t| t.clone()).collect::<Vec<_>>();
                
                 // --- 5.3.1 Créer le tenseur perturbé (-) vue --- 
                 let buffer_minus_arc = {
                    let mut buffer_vec = original_buffer_arc.as_ref().clone(); 
                    if physical_offset >= buffer_vec.len() { return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Offset out of bounds (-) perturbation".to_string()))); }

                    // --- Modification pour calcul en f64 --- 
                    let current_val_f32 = buffer_vec[physical_offset];
                    let current_val_f64 = current_val_f32 as f64;
                    let perturbed_val_f64 = current_val_f64 - epsilon; // Calcul en f64
                    let perturbed_val_f32 = perturbed_val_f64 as f32; // Reconversion en f32
                    // ---------------------------------------
                    
                    buffer_vec[physical_offset] = perturbed_val_f32; // Ecriture f32
                    Arc::new(buffer_vec) 
                };
                 let td_minus = TensorData::new_view(
                    Arc::new(Buffer::Cpu(CpuBuffer::F32(buffer_minus_arc))),
                    original_device,
                    original_offset, 
                    original_shape.clone(), 
                    original_strides.clone(),
                );
                 let perturbed_tensor_minus = Tensor { data: Arc::new(RwLock::new(td_minus)) };

                 inputs_minus[i] = perturbed_tensor_minus;
                 let output_minus = func(&inputs_minus).map_err(GradCheckError::ForwardPassError)?;
                 calculate_loss(&output_minus, output_grad)? // Renvoie f64
            };
            
            // --- 5.4 Calculate Numerical Gradient ---
            let numerical_grad = (loss_plus - loss_minus) / (two * epsilon);

            // --- 5.5 Get Analytical Gradient --- 
             let analytical_grad = analytical_grad_data[elem_idx]; // Utiliser l'index logique

            // --- 5.6 Check for NaN/Infinite Gradients ---
            if !numerical_grad.is_finite() {
                return Err(GradCheckError::NumericalGradNaNOrInfinite {
                    input_index: i,
                    element_index: elem_idx,
                    loss_plus,
                    loss_minus,
                });
            }
            if !analytical_grad.is_finite() {
                 return Err(GradCheckError::AnalyticalGradNaNOrInfinite {
                    input_index: i,
                    element_index: elem_idx,
                    value: analytical_grad,
                 });
            }

            // --- 5.7 Compare Gradients ---
            let difference = (analytical_grad - numerical_grad).abs();
            let allowed_tolerance = abs_tol + rel_tol * numerical_grad.abs();

            if difference > allowed_tolerance {
                return Err(GradCheckError::GradientMismatch {
                    input_index: i,
                    element_index: elem_idx,
                    analytical_grad,
                    numerical_grad,
                    difference,
                });
            }
        } // End element loop
    } // End input loop

    Ok(())
}

/// Helper function to calculate a scalar loss for gradient checking.
/// Usually, this is the sum of the output tensor weighted by the output gradient.
// Adapt signature to use Tensor
fn calculate_loss(tensor: &Tensor, output_grad: &Tensor) -> Result<f64, GradCheckError> {
    // Ensure shapes match or are broadcastable (though they should match here)
    if tensor.shape() != output_grad.shape() {
        // Basic check, could be enhanced with broadcasting logic if needed
         return Err(GradCheckError::TensorError(NeuraRustError::ShapeMismatch {
             operation: "calculate_loss (grad_check)".to_string(),
             expected: format!("{:?}", tensor.shape()), 
             actual: format!("{:?}", output_grad.shape()),
         }));
    }

    // Calculer la perte comme sum(tensor * output_grad) - UTILISER LES TENSEURS ORIGINAUX
    let weighted_output = mul_op(tensor, output_grad)?;
    
    let loss_tensor = sum_op(&weighted_output, None, false)?;

    // Extract the scalar value (expecting F32 CPU)
    let loss_data = loss_tensor.get_f32_data()?;
    if loss_data.len() != 1 {
         return Err(GradCheckError::TensorError(NeuraRustError::InternalError(
             "Loss calculation did not result in a scalar tensor".to_string(),
         )));
    }
    Ok(loss_data[0] as f64) // Convert F32 scalar to f64
}
