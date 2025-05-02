use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use thiserror::Error;
use std::fmt::Debug;
// Import necessary traits for floating point numbers
use num_traits::{Float, Zero, One, Signed};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::sync::Arc;

/// Error type specifically for gradient checking failures.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GradCheckError<T: Float + Debug> { // T must be Float for epsilon checks
    #[error("Gradient check failed for input tensor at index {input_index}, element index {element_index}: Analytical grad {analytical_grad:?} != Numerical grad {numerical_grad:?}. Difference: {difference:?}")]
    GradientMismatch {
        input_index: usize,
        element_index: usize,
        analytical_grad: T,
        numerical_grad: T,
        difference: T,
    },

    #[error("Failed to get mutable data for input tensor {input_index} at element {element_index}: {source}")]
    MutationError {
        input_index: usize,
        element_index: usize,
        source: NeuraRustError, // Assuming getting mutable data might return NeuraRustError
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
}

/// Checks the analytical gradients computed by the autograd system against numerical gradients.
///
/// This function perturbs each element of the input tensors (that require grad) slightly,
/// recomputes the output using the provided forward function, and approximates the gradient
/// using the finite difference method: `(f(x + eps) - f(x - eps)) / (2 * eps)`.
/// It then compares this numerical gradient to the analytical gradient obtained by calling
/// `.backward()` on the original output.
///
/// # Type Parameters
/// * `T`: The floating-point data type of the tensors (e.g., `f32`, `f64`). Must implement `Float`,
///        `Debug`, `Copy`, `Send`, `Sync`, `Default`, `PartialOrd`, `Signed`, `Zero`, `One`,
///        `AddAssign`, `Sum` and comparison traits from `approx`.
/// * `F`: The type of the forward function being tested. It must take a slice of `&Tensor<T>`
///        as input and return a `Result<Tensor<T>, NeuraRustError>`.
///
/// # Arguments
/// * `func`: The forward function to test (e.g., `|inputs| add_op(inputs[0], inputs[1])`).
/// * `inputs`: A slice of input tensors provided to the forward function.
/// * `output_grad`: The gradient to propagate backward from the output tensor (usually ones).
/// * `epsilon`: A small floating-point value used for the finite difference perturbation.
/// * `tolerance`: The absolute difference tolerance allowed between numerical and analytical gradients.
///
/// # Returns
/// * `Ok(())` if all gradient checks pass within the specified tolerance.
/// * `Err(GradCheckError<T>)` if any check fails or an error occurs during computation.
pub fn check_grad<T, F>(
    func: F,
    inputs: &[Tensor<T>],
    output_grad: &Tensor<T>,
    epsilon: T,
    tolerance: T,
) -> Result<(), GradCheckError<T>>
where
    T: Float // Basic requirement for epsilon
       + Debug + Copy + Send + Sync + Default + PartialOrd + Signed + Zero + One
       + std::ops::AddAssign + std::iter::Sum // For internal ops like sum_axes
       + AbsDiffEq<Epsilon = T> + RelativeEq<Epsilon = T> + UlpsEq<Epsilon = T> // For approx comparison
       + 'static,
    F: Fn(&[Tensor<T>]) -> Result<Tensor<T>, NeuraRustError>,
{
    // --- Constants ---
    let two = T::one() + T::one();

    // --- 1. Initial Forward and Backward Pass ---
    // Clone inputs for the initial pass to ensure original inputs remain unchanged
    let initial_inputs: Vec<Tensor<T>> = inputs.iter().map(|t| t.clone()).collect();
    // Set requires_grad=true on inputs that need it for the initial pass
    for input in initial_inputs.iter() {
        if input.requires_grad() {
            // Need to ensure requires_grad can be set even if it was already true
            // or handle potential errors/warnings from set_requires_grad if called on non-leaf?
            // For grad check, assume inputs ARE leaf nodes or we detach them first.
            // Let's assume for now the user provides leaf tensors requiring grad.
            // If not, maybe clone and detach -> set_requires_grad?
            // Safest: Clone, set_requires_grad(true) on the clone for the grad check process.
        }
    }

    let output = func(&initial_inputs).map_err(GradCheckError::ForwardPassError)?;

    // Perform backward pass to get analytical gradients
    // Ensure output requires grad if any input did (func should handle this)
    if !output.requires_grad() && inputs.iter().any(|t| t.requires_grad()) {
        // This might happen if `func` doesn't correctly propagate requires_grad.
        // For now, proceed, but backward will likely do nothing or error.
        // Consider adding a warning or specific error later.
    }

    // We need mutable access to initial_inputs later to retrieve grads.
    // Let's clear grads first to ensure clean state.
    for input in initial_inputs.iter() {
        if input.requires_grad() {
            let mut input_guard = input.write_data(); // Use write_data defined in tensor/mod.rs?
            input_guard.grad = None;
        }
    }

    if output.requires_grad() {
        output.backward(Some(output_grad.clone())).map_err(GradCheckError::BackwardPassError)?;
    } // else: no backward pass possible, analytical grads will be None

    // Store analytical gradients
    let mut analytical_grads: Vec<Option<Tensor<T>>> = Vec::with_capacity(inputs.len());
    for input in initial_inputs.iter() {
        analytical_grads.push(input.grad()); // grad() clones the Option<Tensor>
    }

    // --- 3. Iterate through Inputs --- 
    for (i, original_input) in inputs.iter().enumerate() {
        if !original_input.requires_grad() {
            continue; // Skip inputs that don't require grad
        }

        let analytical_grad_tensor = match analytical_grads[i].as_ref() {
            Some(grad) => grad,
            None => {
                // If analytical grad is None but input requires_grad, check numerical is also zero
                // TODO: Add check later, for now, continue if no analytical grad was computed.
                // This could happen if the input wasn't part of the graph leading to the output.
                continue;
            }
        };

        // --- 4. Iterate through Elements --- 
        let numel = original_input.numel();

        // We need mutable access to the *data* to perturb it.
        // Approach 1: Clone the tensor, then get mutable access to the *cloned* data.

        for elem_idx in 0..numel {
            // --- 5.a & 5.b: Create Perturbed Inputs (+/- eps) ---

            // Helper closure to create a perturbed tensor
            let create_perturbed_input = |perturbation: T| -> Result<Tensor<T>, GradCheckError<T>> {
                // Make immutable, assigned once
                let perturbed_input = original_input.clone();
                // Get mutable access to the *cloned* tensor's data buffer (CPU only for now)
                let mut data_guard = perturbed_input.write_data(); // Get write lock on TensorData
                if data_guard.device != crate::device::StorageDevice::CPU {
                    return Err(GradCheckError::TensorError(NeuraRustError::UnsupportedOperation(
                        "Gradient checking perturbation only supported on CPU.".to_string()
                    )));
                }
                let buffer_arc = Arc::clone(&data_guard.data);
                // Need mutable access to the Vec<T> inside the Buffer::Cpu
                // This requires Arc::get_mut which only works if ref count is 1.
                // Since we just cloned, the TensorData Arc has count 1, but the Buffer Arc might not.
                // Solution: If Buffer Arc is shared, we MUST clone the Buffer itself.
                let mut cpu_buffer = match Arc::try_unwrap(buffer_arc) {
                    Ok(buffer) => {
                        // We have unique ownership of the Buffer
                        match buffer {
                             crate::buffer::Buffer::Cpu(vec_arc) => {
                                 // Try to get mutable access to the Vec
                                 match Arc::try_unwrap(vec_arc) {
                                     Ok(vec_data) => vec_data, // We got the Vec<T>
                                     Err(returned_arc) => {
                                         // Vec is still shared, clone it
                                         returned_arc.as_ref().clone()
                                     }
                                 }
                            }
                             // Correct match for Gpu variant
                             crate::buffer::Buffer::Gpu { .. } => {
                                return Err(GradCheckError::TensorError(NeuraRustError::UnsupportedOperation(
                                    "Gradient checking perturbation only supported on CPU.".to_string()
                                )));
                             }
                         }
                    },
                    Err(returned_arc) => {
                        // Buffer Arc is shared, clone the underlying data
                        match returned_arc.as_ref() {
                             crate::buffer::Buffer::Cpu(vec_arc) => {
                                 vec_arc.as_ref().clone() // Clone the Vec<T>
                             }
                             // Correct match for Gpu variant
                             crate::buffer::Buffer::Gpu { .. }=> {
                                 return Err(GradCheckError::TensorError(NeuraRustError::UnsupportedOperation(
                                    "Gradient checking perturbation only supported on CPU.".to_string()
                                )));
                             }
                        }
                    }
                };

                // Now we have a mutable `cpu_buffer: Vec<T>`
                // Calculate the linear index corresponding to elem_idx (assuming contiguous for simplicity now)
                // TODO: Handle strides correctly if inputs can be non-contiguous!
                if !data_guard.is_contiguous() {
                     return Err(GradCheckError::TensorError(NeuraRustError::UnsupportedOperation(
                        "Gradient checking on non-contiguous tensors not yet supported.".to_string()
                    )));
                }
                // For contiguous, linear index is elem_idx + offset (offset should be 0 if we clone data)
                let linear_idx = data_guard.offset + elem_idx; // Check if offset matters after clone
                if linear_idx >= cpu_buffer.len() {
                    return Err(GradCheckError::TensorError(NeuraRustError::InternalError(
                        format!("Calculated linear index {} out of bounds for buffer len {}.", linear_idx, cpu_buffer.len())
                    )));
                }

                // Apply perturbation
                cpu_buffer[linear_idx] = cpu_buffer[linear_idx] + perturbation;

                // Put the potentially cloned buffer back into TensorData
                data_guard.data = Arc::new(crate::buffer::Buffer::Cpu(Arc::new(cpu_buffer)));
                // Ensure requires_grad is true for the perturbed inputs when calling func
                data_guard.requires_grad = true;
                data_guard.grad_fn = None; // Perturbed inputs are treated as leaves
                data_guard.grad = None;

                drop(data_guard); // Release write lock
                Ok(perturbed_input)
            };

            let input_plus_eps = create_perturbed_input(epsilon)?;
            let input_minus_eps = create_perturbed_input(-epsilon)?;

            // --- 5.c & 5.d: Call func with perturbed inputs ---
            // Need to substitute the perturbed tensor into the input slice
            let mut inputs_for_plus: Vec<Tensor<T>> = inputs.iter().cloned().collect();
            inputs_for_plus[i] = input_plus_eps;
            let output_plus_eps = func(&inputs_for_plus).map_err(GradCheckError::ForwardPassError)?;

            let mut inputs_for_minus: Vec<Tensor<T>> = inputs.iter().cloned().collect();
            inputs_for_minus[i] = input_minus_eps;
            let output_minus_eps = func(&inputs_for_minus).map_err(GradCheckError::ForwardPassError)?;

            // --- 5.e & 5.f: Calculate scalar losses ---
            // loss = sum(output * output_grad)
            // Need mul_op and sum_all_op (sum_axes with no axes)
            // Map NeuraRustError to GradCheckError::TensorError
            let loss_plus_eps_tensor = crate::ops::arithmetic::mul::mul_op(&output_plus_eps, output_grad)
                .map_err(GradCheckError::TensorError)?;
            let loss_plus_eps_scalar = crate::ops::reduction::sum_axes(&loss_plus_eps_tensor, &[], false)
                .map_err(GradCheckError::TensorError)?;

            let loss_minus_eps_tensor = crate::ops::arithmetic::mul::mul_op(&output_minus_eps, output_grad)
                .map_err(GradCheckError::TensorError)?;
            let loss_minus_eps_scalar = crate::ops::reduction::sum_axes(&loss_minus_eps_tensor, &[], false)
                .map_err(GradCheckError::TensorError)?;

            // Extract scalar values (assuming CPU)
            let loss_plus = loss_plus_eps_scalar.read_data().data.cpu_data()
                .map_err(GradCheckError::TensorError)?.clone()[0]; // Map error
            let loss_minus = loss_minus_eps_scalar.read_data().data.cpu_data()
                .map_err(GradCheckError::TensorError)?.clone()[0]; // Map error

            // --- 5.g: Calculate numerical gradient ---
            let numerical_grad = (loss_plus - loss_minus) / (two * epsilon);

            // --- 5.h: Get analytical gradient ---
            // Assuming analytical_grad_tensor is contiguous for now
            if !analytical_grad_tensor.is_contiguous() {
                 return Err(GradCheckError::TensorError(NeuraRustError::UnsupportedOperation(
                    "Gradient checking on non-contiguous analytical gradients not yet supported.".to_string()
                )));
            }
            let analytical_data_arc = analytical_grad_tensor.read_data().data.cpu_data()
                .map_err(|e| GradCheckError::AnalyticalGradAccessError { input_index: i, source: e })?.clone(); // Map error specifically
            // TODO: Handle strides for analytical grad access if non-contiguous
            let analytical_linear_idx = analytical_grad_tensor.read_data().offset + elem_idx;
             if analytical_linear_idx >= analytical_data_arc.len() {
                 return Err(GradCheckError::TensorError(NeuraRustError::InternalError(
                    format!("Calculated analytical linear index {} out of bounds for buffer len {}.", analytical_linear_idx, analytical_data_arc.len())
                 )));
             }
            let analytical_grad = analytical_data_arc[analytical_linear_idx];

            // --- 5.i: Compare gradients ---
            if !numerical_grad.abs_diff_eq(&analytical_grad, tolerance) {
                 return Err(GradCheckError::GradientMismatch {
                     input_index: i,
                     element_index: elem_idx,
                     analytical_grad,
                     numerical_grad,
                     difference: (analytical_grad - numerical_grad).abs(),
                 });
            }
        }
    }

    Ok(())
} 