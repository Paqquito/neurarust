//!
//! # Gradient Checking Utilities
//!
//! This module provides tools for verifying the correctness of analytical gradients
//! computed by the `autograd` system. The primary function is [`check_grad`], which
//! compares the analytical gradient of a function with respect to its inputs against
//! a numerical approximation computed using the finite difference method.
//!
//! Gradient checking is a crucial debugging technique for ensuring that the backward
//! implementations (`BackwardOp`) for custom operations are correct.
//!

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
///
/// Encapsulates various issues that can occur during the gradient checking process,
/// including mismatches between analytical and numerical gradients, errors during
/// forward/backward passes, and unsupported configurations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GradCheckError {
    /// Indicates a mismatch between the analytical and numerical gradients for a specific element.
    #[error("Gradient check failed for input tensor at index {input_index}, element index {element_index}: Analytical grad {analytical_grad:?} != Numerical grad {numerical_grad:?}. Difference: {difference:?}")]
    GradientMismatch {
        /// Index of the input tensor in the `inputs` slice provided to `check_grad`.
        input_index: usize,
        /// Linear index of the element within the flattened input tensor where the mismatch occurred.
        element_index: usize,
        /// The analytical gradient value computed by the backward pass.
        analytical_grad: f64, // Use f64 for precision
        /// The numerical gradient value estimated using finite differences.
        numerical_grad: f64,
        /// The absolute difference between the analytical and numerical gradients.
        difference: f64,
    },

    /// Error occurring while trying to mutate an input tensor element for finite differences.
    #[error("Failed to get mutable data for input tensor {input_index} at element {element_index}: {source}")]
    MutationError {
        input_index: usize,
        element_index: usize,
        #[source] source: NeuraRustError,
    },
    /// Error occurred during the execution of the forward function `func`.
    #[error("Forward function execution failed during gradient check: {0}")]
    ForwardPassError(#[source] NeuraRustError),
    /// Error occurred during the execution of the backward pass (`.backward()`).
    #[error("Backward pass execution failed during gradient check: {0}")]
    BackwardPassError(#[source] NeuraRustError),
    /// Error accessing the analytical gradient (`.grad`) of an input tensor after backward pass.
    #[error("Could not access analytical gradient for input {input_index}: {source}")]
    AnalyticalGradAccessError {
        input_index: usize,
        #[source] source: NeuraRustError,
    },
    /// Generic tensor-related error during intermediate calculations.
    #[error("Tensor error during intermediate calculation: {0}")]
    TensorError(#[source] NeuraRustError),
    /// The data type of an input or output tensor is not supported (currently only F32/F64).
    #[error("Unsupported data type for gradient check: expected F32/F64, got {0:?}")]
    UnsupportedDType(DType),
    /// An input tensor required gradients, but its `.grad` field was `None` after the backward pass.
    #[error("Input tensor {input_index} requires grad but has no gradient after backward pass.")]
    MissingAnalyticalGrad { input_index: usize },
    /// The calculated numerical gradient was NaN or infinite.
    #[error("Numerical gradient is NaN or infinite for input {input_index}, element {element_index}. Details: Loss+: {loss_plus:?}, Loss-: {loss_minus:?}")]
    NumericalGradNaNOrInfinite {
        input_index: usize,
        element_index: usize,
        loss_plus: f64,
        loss_minus: f64,
    },
    /// The analytical gradient was NaN or infinite.
    #[error("Analytical gradient is NaN or infinite for input {input_index}, element {element_index}. Value: {value:?}")]
    AnalyticalGradNaNOrInfinite {
        input_index: usize,
        element_index: usize,
        value: f64,
    },
    /// Gradient checking on non-contiguous input tensors is not currently supported.
    #[error("Gradient checking on non-contiguous tensors not yet supported (Input {input_index}).")]
    NonContiguousInput { input_index: usize },
    /// Gradient checking is currently only supported for tensors residing on the CPU.
    #[error("Gradient checking only supported on CPU tensors (Input {input_index}). Got: {device:?}")]
    NonCpuInput {
        input_index: usize,
        device: StorageDevice,
    },
    /// An input tensor provided for checking requires gradients but is not a leaf node (it has a `grad_fn`).
    #[error("Gradient check input tensor must be a leaf node (no grad_fn). Input index: {input_index}")]
    InputNotLeaf { input_index: usize },
    /// The function being tested did not correctly propagate the `requires_grad` flag to its output.
    #[error("Function did not propagate requires_grad correctly.")]
    RequiresGradPropagationError,
}

// Map NeuraRustError to GradCheckError::TensorError
impl From<NeuraRustError> for GradCheckError {
    fn from(err: NeuraRustError) -> Self {
        GradCheckError::TensorError(err)
    }
}

/// Checks the analytical gradients computed by a function against numerical approximations.
///
/// This function is a cornerstone for verifying the correctness of `BackwardOp` implementations.
/// It operates by comparing the gradient computed via the `backward()` pass (analytical gradient)
/// with a gradient estimated using the finite difference formula:
///
/// \\[ \nabla_{\text{num}} L(x_i) \approx \frac{L(x_1, ..., x_i + \epsilon, ...) - L(x_1, ..., x_i - \epsilon, ...)}{2 \epsilon} \\]
///
/// Where:
/// - \( L \) is a scalar loss function, implicitly defined here as \( L = \sum (\text{func}(\text{inputs}) \cdot \text{output\_grad}) \).
///   This reduction to a scalar is necessary for the finite difference method.
/// - \( x_i \) represents a single element within one of the input tensors.
/// - \( \epsilon \) is a small perturbation (`epsilon` argument).
///
/// The comparison between the analytical gradient (\( \nabla_{\text{analytical}} L(x_i) \)) and the numerical one
/// (\( \nabla_{\text{num}} L(x_i) \)) uses both absolute and relative tolerances to account for floating-point inaccuracies:
///
/// \\[ | \nabla_{\text{analytical}} - \nabla_{\text{num}} | \le \text{abs\_tol} + \text{rel\_tol} \times | \nabla_{\text{num}} | \\]
///
/// # Type Constraints
/// - Currently, gradient checking is only supported for `DType::F32` or `DType::F64` tensors.
/// - All input tensors and the `output_grad` tensor must reside on the `CPU`.
/// - All input tensors must be contiguous.
/// - Input tensors for which gradients are checked (`requires_grad = true`) must be leaf nodes
///   in the computation graph (i.e., they must not have a `grad_fn`).
///
/// # Arguments
/// * `func`: The function \( f \) whose gradient implementation is being tested. It takes a slice
///   of input `Tensor`s (`&[Tensor]`) and returns a `Result<Tensor, NeuraRustError>` representing the output tensor.
///   This function should internally build the computation graph if its inputs require gradients.
/// * `inputs`: A slice containing the input `Tensor` instances to be passed to `func`. Tensors within this slice
///   that have `requires_grad = true` will have their gradients checked.
/// * `output_grad`: A `Tensor` representing the initial gradient \( \frac{dL}{d\text{Output}} \) to be backpropagated
///   from the output of `func`. It must have the same shape, `DType`, and `StorageDevice` (CPU) as the expected output of `func`.
///   Often, this is a tensor of ones for checking unweighted gradients.
/// * `epsilon`: A small `f64` value used as the perturbation \( \epsilon \) for the finite difference calculation (e.g., `1e-5`).
/// * `abs_tol`: The absolute tolerance (`f64`) allowed for the difference between analytical and numerical gradients (e.g., `1e-7`).
/// * `rel_tol`: The relative tolerance (`f64`) allowed for the difference, scaled by the magnitude of the numerical gradient (e.g., `1e-5`).
///
/// # Returns
/// * `Ok(())`: If the analytical and numerical gradients match within the specified tolerances for all checked elements.
/// * `Err(GradCheckError)`: If any gradient mismatch exceeds the tolerances, or if any other error occurs during the process
///   (e.g., errors in forward/backward passes, unsupported types/devices, non-leaf inputs, numerical instability).
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
    let mut first_input_dtype: Option<DType> = None;

    for (i, input) in inputs.iter().enumerate() {
        let dtype = input.dtype();
        let device = input.device();

        // Check device
        if device != StorageDevice::CPU {
            return Err(GradCheckError::NonCpuInput { input_index: i, device });
        }

        // Check dtype and consistency
        match dtype {
            DType::F32 | DType::F64 => {
                if let Some(first_dtype) = first_input_dtype {
                    if dtype != first_dtype {
                        return Err(GradCheckError::TensorError(NeuraRustError::DataTypeMismatch {
                            expected: first_dtype,
                            actual: dtype,
                            operation: format!("check_grad input consistency (input {})", i),
                        }));
                    }
                } else {
                    first_input_dtype = Some(dtype);
                }
            }
        }

        // Check contiguity
        if !input.is_contiguous() {
            return Err(GradCheckError::NonContiguousInput { input_index: i });
        }
         // Ensure inputs requiring grad are leaf nodes
        if input.requires_grad() && input.read_data().grad_fn.is_some() {
            return Err(GradCheckError::InputNotLeaf { input_index: i });
        }
    }

    // Check output_grad
    let output_grad_dtype = output_grad.dtype();
    let output_grad_device = output_grad.device();

    if output_grad_device != StorageDevice::CPU {
         return Err(GradCheckError::NonCpuInput { input_index: usize::MAX, device: output_grad_device }); // Use usize::MAX for output_grad index
    }

    // Ensure output_grad dtype matches input dtype (if inputs exist)
    if let Some(input_dtype) = first_input_dtype {
        if output_grad_dtype != input_dtype {
            return Err(GradCheckError::TensorError(NeuraRustError::DataTypeMismatch {
                expected: input_dtype,
                actual: output_grad_dtype,
                operation: "check_grad output_grad consistency".to_string(),
            }));
        }
    } else {
        // No inputs, check if output_grad is F32 or F64
        match output_grad_dtype {
            DType::F32 | DType::F64 => { /* Ok */ }
        }
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

        // Ensure analytical gradient tensor is contiguous before extracting data
        let contiguous_analytical_grad = analytical_grad_tensor.contiguous()?;

        // Extract analytical gradient data based on DType
        let analytical_grad_data: Vec<f64> = match contiguous_analytical_grad.dtype() {
            DType::F32 => {
                contiguous_analytical_grad.get_f32_data()? // Get Vec<f32>
                    .iter()
                    .map(|&x| x as f64) // Convert to Vec<f64>
                    .collect()
            }
            DType::F64 => {
                contiguous_analytical_grad.get_f64_data()? // Get Vec<f64>
            }
        };

        // Ensure analytical grad data is valid
        for (elem_idx, &_val) in analytical_grad_data.iter().enumerate() {
            // --- 5.1 Calculate Physical Offset ---
            let mut current_logical_indices = vec![0; original_input.shape().len()];
            let mut current_linear = elem_idx;
            for dim in (0..original_input.shape().len()).rev() {
                let shape_val = original_input.shape()[dim];
                if shape_val > 0 { current_logical_indices[dim] = current_linear % shape_val; current_linear /= shape_val; } else { current_logical_indices[dim] = 0; }
            }
            let physical_offset = original_input.read_data().offset + current_logical_indices.iter().zip(original_input.strides().iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();

            // --- 5.2 Calculate Loss for f(x + eps) ---
            let loss_plus = {
                let mut inputs_plus = inputs.iter().map(|t| t.clone()).collect::<Vec<_>>();

                // --- 5.2.1 Create perturbed buffer (+) ---
                // Corrected logic: Match on dereferenced Arc<Buffer>, clone inner Vec, perturb, create new Arc<Buffer>
                let perturbed_buffer_arc_plus = match &*original_input.read_data().buffer { // Dereference Arc<Buffer> to &Buffer
                    Buffer::Cpu(CpuBuffer::F32(arc_vec)) => {
                        let mut buffer_vec = arc_vec.as_ref().clone(); // Clone Vec<f32>
                        if physical_offset >= buffer_vec.len() { return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Offset out of bounds F32 (+)".to_string()))); }
                        let perturbed_val_f32 = (buffer_vec[physical_offset] as f64 + epsilon) as f32;
                        buffer_vec[physical_offset] = perturbed_val_f32;
                        Arc::new(Buffer::Cpu(CpuBuffer::F32(Arc::new(buffer_vec)))) // Create new Arc<Buffer>
                    }
                    Buffer::Cpu(CpuBuffer::F64(arc_vec)) => {
                        let mut buffer_vec = arc_vec.as_ref().clone(); // Clone Vec<f64>
                        if physical_offset >= buffer_vec.len() { return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Offset out of bounds F64 (+)".to_string()))); }
                        let perturbed_val_f64 = buffer_vec[physical_offset] + epsilon;
                        buffer_vec[physical_offset] = perturbed_val_f64;
                        Arc::new(Buffer::Cpu(CpuBuffer::F64(Arc::new(buffer_vec)))) // Create new Arc<Buffer>
                    }
                    // Non-CPU buffers are checked earlier
                    _ => return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Unexpected buffer type (+)".to_string())))
                };

                // --- 5.2.2 Create perturbed tensor view (+) ---
                let td_plus = TensorData::new_view(
                    perturbed_buffer_arc_plus, // Use the newly created Arc<Buffer>
                    original_input.device(),
                    original_input.read_data().offset,
                    original_input.shape().clone(),
                    original_input.strides().clone(),
                )?;
                inputs_plus[i] = Tensor { data: Arc::new(RwLock::new(td_plus)) };

                // --- 5.2.3 Run forward pass (+) ---
                let output_plus = func(&inputs_plus).map_err(GradCheckError::ForwardPassError)?;
                calculate_loss(&output_plus, output_grad)?
            };

            // --- 5.3 Calculate Loss for f(x - eps) ---
            let loss_minus = {
                let mut inputs_minus = inputs.iter().map(|t| t.clone()).collect::<Vec<_>>();

                // --- 5.3.1 Create perturbed buffer (-) ---
                // Corrected logic: Match on dereferenced Arc<Buffer>, clone inner Vec, perturb, create new Arc<Buffer>
                let perturbed_buffer_arc_minus = match &*original_input.read_data().buffer { // Dereference Arc<Buffer> to &Buffer
                     Buffer::Cpu(CpuBuffer::F32(arc_vec)) => {
                        let mut buffer_vec = arc_vec.as_ref().clone(); // Clone Vec<f32>
                        if physical_offset >= buffer_vec.len() { return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Offset out of bounds F32 (-)".to_string()))); }
                        let perturbed_val_f32 = (buffer_vec[physical_offset] as f64 - epsilon) as f32;
                        buffer_vec[physical_offset] = perturbed_val_f32;
                        Arc::new(Buffer::Cpu(CpuBuffer::F32(Arc::new(buffer_vec)))) // Create new Arc<Buffer>
                    }
                    Buffer::Cpu(CpuBuffer::F64(arc_vec)) => {
                        let mut buffer_vec = arc_vec.as_ref().clone(); // Clone Vec<f64>
                        if physical_offset >= buffer_vec.len() { return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Offset out of bounds F64 (-)".to_string()))); }
                        let perturbed_val_f64 = buffer_vec[physical_offset] - epsilon;
                        buffer_vec[physical_offset] = perturbed_val_f64;
                        Arc::new(Buffer::Cpu(CpuBuffer::F64(Arc::new(buffer_vec)))) // Create new Arc<Buffer>
                    }
                    // Non-CPU buffers are checked earlier
                    _ => return Err(GradCheckError::TensorError(NeuraRustError::InternalError("Unexpected buffer type (-)".to_string())))
                };

                // --- 5.3.2 Create perturbed tensor view (-) ---
                let td_minus = TensorData::new_view(
                    perturbed_buffer_arc_minus, // Use the newly created Arc<Buffer>
                    original_input.device(),
                    original_input.read_data().offset,
                    original_input.shape().clone(),
                    original_input.strides().clone(),
                )?;
                inputs_minus[i] = Tensor { data: Arc::new(RwLock::new(td_minus)) };

                // --- 5.3.3 Run forward pass (-) ---
                let output_minus = func(&inputs_minus).map_err(GradCheckError::ForwardPassError)?;
                calculate_loss(&output_minus, output_grad)?
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

/// Calculates a scalar loss suitable for finite difference gradient checking.
///
/// The finite difference method requires evaluating a scalar loss function \( L \).
/// This helper function computes such a scalar loss, typically defined as the sum
/// of the element-wise product between the function's output tensor and the provided
/// `output_grad` (which represents \( \frac{dL_{final}}{dOutput} \) where \( L_{final} \) is the final scalar loss
/// of the overall computation, often just 1.0 for unweighted gradients).
///
/// \\[ L = \sum (\text{tensor} \odot \text{output\_grad}) \\]
///
/// This effectively weights the contribution of each output element to the final loss.
/// The result is returned as an `f64` for numerical stability in the finite difference calculation.
/// Handles potential `F32` to `F64` conversion internally.
///
/// # Arguments
/// * `tensor`: The output tensor produced by the function being checked during one of the
///             finite difference evaluations (e.g., \( f(x+\epsilon) \)).
/// * `output_grad`: The gradient tensor provided to `check_grad`, representing \( \frac{dL_{final}}{dOutput} \).
///
/// # Returns
/// * `Ok(f64)`: The computed scalar loss value.
/// * `Err(GradCheckError)`: If an error occurs during the element-wise multiplication or sum operations,
///                        or during data conversion/access.
fn calculate_loss(tensor: &Tensor, output_grad: &Tensor) -> Result<f64, GradCheckError> {
    // Ensure shapes match or are broadcastable (though they should match here)
    if tensor.shape() != output_grad.shape() {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "calculate_loss (grad_check)".to_string(),
            expected: format!("{:?}", output_grad.shape()),
            actual: format!("{:?}", tensor.shape()),
        }.into()); // Convert NeuraRustError to GradCheckError
    }

    // Perform element-wise multiplication
    let product = mul_op(tensor, output_grad)?;

    // Sum all elements to get a scalar
    let sum_tensor = sum_op(&product, None, false)?; // Sum over all axes

    // Extract the scalar value as f64
    sum_tensor.item_f64().map_err(|e| e.into()) // Convert NeuraRustError to GradCheckError
}
