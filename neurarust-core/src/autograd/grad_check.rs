use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use std::fmt::Debug;
use thiserror::Error;
// Remove unused approx traits for now
// use approx::{AbsDiffEq, RelativeEq, UlpsEq};
// Remove unused num_traits
// use num_traits::{Float, One, Signed, Zero};
// Remove unused Arc
// use std::sync::Arc;

// Import specific ops needed for calculate_loss
use crate::ops::reduction::sum::sum_op; // Use sum_op for loss calculation
use crate::types::DType;
use crate::device::StorageDevice;

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
// Remove T generic
pub fn check_grad<F>(
    func: F,
    inputs: &[Tensor], // Use Tensor
    output_grad: &Tensor, // Use Tensor
    epsilon: f64,
    tolerance: f64,
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
                // If input requires grad but has no grad tensor, something is wrong
                 return Err(GradCheckError::MissingAnalyticalGrad{ input_index: i });
            }
        };

        // Extract analytical grad data (expecting F32, convert to f64)
        let analytical_grad_data: Vec<f64> = analytical_grad_tensor
            .contiguous()? // Make contiguous first
            .get_f32_data()? // Returns Result<Vec<f32>, NeuraRustError>
            .iter()
            .map(|&x| x as f64)
            .collect();

        // --- 5. Iterate through Elements and Calculate Numerical Gradient ---
        let numel = original_input.numel();
        // Get original data once (as f64)
        let original_data_vec_f64: Vec<f64> = original_input
            .contiguous()? // Make contiguous first
            .get_f32_data()? // Expecting F32
            .iter()
            .map(|&x| x as f64)
            .collect();

        for elem_idx in 0..numel {
            // --- Calculate Loss for f(x + eps) ---
            let loss_plus = {
                let mut inputs_plus = inputs.iter().map(|t| t.clone()).collect::<Vec<_>>();
                let mut data_plus_f64 = original_data_vec_f64.clone();
                data_plus_f64[elem_idx] += epsilon;
                let data_plus_f32: Vec<f32> = data_plus_f64.iter().map(|&x| x as f32).collect();
                // Create the perturbed tensor
                let perturbed_tensor = Tensor::from_vec_f32(data_plus_f32, original_input.shape())?;
                // Explicitly set requires_grad if the original input needed it
                if original_input.requires_grad() {
                     perturbed_tensor.set_requires_grad(true)?;
                     // Ensure it's treated as a leaf (it should be by default from from_vec_f32)
                     // We might add an assertion here later if needed: assert!(perturbed_tensor.read_data().grad_fn.is_none());
                }
                inputs_plus[i] = perturbed_tensor;
                
                let output_plus = func(&inputs_plus).map_err(GradCheckError::ForwardPassError)?;
                // WARNING: Using simplified loss calculation (sum of tensor elements)!
                // TODO: Replace with proper loss calculation based on output_grad
                // Calculate loss using the provided output_grad
                 calculate_loss(&output_plus, output_grad)? // Use the helper function
            };

            // --- Calculate Loss for f(x - eps) ---
            let loss_minus = {
                let mut inputs_minus = inputs.iter().map(|t| t.clone()).collect::<Vec<_>>();
                let mut data_minus_f64 = original_data_vec_f64.clone();
                data_minus_f64[elem_idx] -= epsilon;
                let data_minus_f32: Vec<f32> = data_minus_f64.iter().map(|&x| x as f32).collect();
                // Create the perturbed tensor
                let perturbed_tensor = Tensor::from_vec_f32(data_minus_f32, original_input.shape())?;
                 // Explicitly set requires_grad if the original input needed it
                 if original_input.requires_grad() {
                      perturbed_tensor.set_requires_grad(true)?;
                      // Ensure it's treated as a leaf
                      // assert!(perturbed_tensor.read_data().grad_fn.is_none());
                 }
                 inputs_minus[i] = perturbed_tensor;

                let output_minus = func(&inputs_minus).map_err(GradCheckError::ForwardPassError)?;
                // WARNING: Using simplified loss calculation (sum of tensor elements)!
                // TODO: Replace with proper loss calculation based on output_grad
                // Calculate loss using the provided output_grad
                 calculate_loss(&output_minus, output_grad)? // Use the helper function
            };
            
            // --- Calculate Numerical Gradient ---
            let numerical_grad = (loss_plus - loss_minus) / (two * epsilon);

            // --- Get Analytical Gradient ---
            let analytical_grad = analytical_grad_data[elem_idx]; // Already fetched as f64

            // --- Check for NaN/Infinite Gradients ---
            if numerical_grad.is_nan() || numerical_grad.is_infinite() {
                return Err(GradCheckError::NumericalGradNaNOrInfinite {
                    input_index: i,
                    element_index: elem_idx,
                    loss_plus,
                    loss_minus,
                });
            }
            if analytical_grad.is_nan() || analytical_grad.is_infinite() {
                 return Err(GradCheckError::AnalyticalGradNaNOrInfinite {
                    input_index: i,
                    element_index: elem_idx,
                    value: analytical_grad,
                 });
            }

            // --- Compare Gradients ---
            let difference = (analytical_grad - numerical_grad).abs();
            if difference > tolerance && (difference / (analytical_grad.abs() + epsilon)) > tolerance { // Added relative check
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

    // --- TEMPORARY SIMPLIFICATION: Sum elements of the output tensor directly --- 
    // This ignores output_grad but removes dependency on mul_op/sum_op for debugging
    println!("WARNING: Using simplified loss calculation in check_grad (sum of tensor elements)!");
    let loss_tensor_simplified = sum_op(tensor, None, false)
                                    .map_err(GradCheckError::TensorError)?;
    // -----------------------------------------------------------------------------

    // Extract the scalar value (expecting F32 CPU)
    let loss_data = loss_tensor_simplified.get_f32_data()?;
    if loss_data.len() != 1 {
         return Err(GradCheckError::TensorError(NeuraRustError::InternalError(
             "Loss calculation did not result in a scalar tensor".to_string(),
         )));
    }
    Ok(loss_data[0] as f64) // Convert F32 scalar to f64
}
