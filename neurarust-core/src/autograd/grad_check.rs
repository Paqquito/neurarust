use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use std::fmt::Debug;
use thiserror::Error;
// Import necessary traits for floating point numbers
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::{Float, One, Signed, Zero};
use std::sync::Arc;

// Import traits for tensor operations used in calculate_loss
// use crate::ops::binary_ops::Broadcast; // Incorrect trait
// use crate::ops::arithmetic::ArithmeticOps; // Incorrect trait
// use crate::ops::reduction::ReductionOps; // Incorrect trait
use crate::ops::arithmetic::mul::mul_op; // Use the mul_op function
use crate::ops::reduction::sum::sum_axes; // Use the sum_axes function

/// Error type specifically for gradient checking failures.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GradCheckError<T: Float + Debug> {
    // T must be Float for epsilon checks
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
        + Debug
        + Copy
        + Send
        + Sync
        + Default
        + PartialOrd
        + Signed
        + Zero
        + One
        + std::ops::AddAssign
        + std::iter::Sum // For internal ops like sum_axes
        + AbsDiffEq<Epsilon = T>
        + RelativeEq<Epsilon = T>
        + UlpsEq<Epsilon = T> // For approx comparison
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
        output
            .backward(Some(output_grad.clone()))
            .map_err(GradCheckError::BackwardPassError)?;
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

        // --- Get a truly independent copy of the original input data for reference ---
        let original_input_data_copy = original_input
            .read_data()
            .data
            .cpu_data()
            .map_err(|e| GradCheckError::TensorError(e))?
            .clone(); // Clone the Arc<Vec<T>>
                      // We might need to clone the Vec itself if the Arc is shared, but let's try this first.
                      // Let's assume cpu_data() gives us Arc<Vec<T>>.
                      // For safety, clone the Vec data itself to ensure independence
        let original_data_vec = original_input_data_copy.as_ref().clone();

        for elem_idx in 0..numel {
            // --- 5.a & 5.b: Create Perturbed Inputs (+/- eps) ---

            // Helper closure to create a perturbed tensor
            let create_perturbed_input = |perturbation: T| -> Result<Tensor<T>, GradCheckError<T>> {
                // --- 1. Get original properties (before locking original_input) ---
                let original_shape = original_input.shape(); // Clone shape
                let original_device = original_input.device();
                // Assuming contiguous for now, recalculate strides for the new TensorData
                let new_strides = crate::tensor_data::TensorData::<T>::calculate_contiguous_strides(
                    &original_shape,
                );
                if !original_input.read_data().is_contiguous() {
                    return Err(GradCheckError::TensorError(
                        NeuraRustError::UnsupportedOperation(
                            "Gradient checking on non-contiguous tensors not yet supported."
                                .to_string(),
                        ),
                    ));
                }
                if original_device != crate::device::StorageDevice::CPU {
                    return Err(GradCheckError::TensorError(
                        NeuraRustError::UnsupportedOperation(
                            "Gradient checking perturbation only supported on CPU.".to_string(),
                        ),
                    ));
                }

                // --- 2. Create the new perturbed data buffer ---
                // Clone the original vector data for modification
                let mut new_data_vec = original_data_vec.clone();

                // Calculate linear index (assuming contiguous)
                let linear_idx = elem_idx; // Offset is 0 for the vec copy
                if linear_idx >= new_data_vec.len() {
                    return Err(GradCheckError::TensorError(NeuraRustError::InternalError(
                        format!(
                            "Calculated linear index {} out of bounds for buffer len {}.",
                            linear_idx,
                            new_data_vec.len()
                        ),
                    )));
                }

                // Apply perturbation based on the ORIGINAL data
                let original_value_from_copy = original_data_vec[linear_idx]; // Reference value
                new_data_vec[linear_idx] = original_value_from_copy + perturbation;
                // DEBUG: Print perturbed value (Remove)
                // println!("Perturbing input {} elem {} from {:?} to {:?}", i, elem_idx, original_value_from_copy, new_data_vec[linear_idx]);

                // --- 3. Create NEW TensorData and Tensor ---
                // Create a new CPU buffer with the modified data
                let new_buffer = crate::buffer::Buffer::new_cpu(new_data_vec);

                // Create completely new TensorData
                let new_tensor_data = crate::tensor_data::TensorData {
                    data: Arc::new(new_buffer),
                    device: original_device, // Should be CPU
                    offset: 0,
                    shape: original_shape, // Use cloned original shape
                    strides: new_strides,  // Use recalculated strides
                    // Initialize autograd fields - perturbed inputs are leaves
                    requires_grad: true, // Need this true for func call, even if not leaf conceptually?
                    // Let's set it true, func might expect it.
                    grad: None,
                    grad_fn: None,
                };

                // Create the final Tensor pointing to the new TensorData
                let perturbed_tensor = Tensor {
                    data: Arc::new(std::sync::RwLock::new(new_tensor_data)),
                };

                Ok(perturbed_tensor)
            };

            let input_plus_eps = create_perturbed_input(epsilon)?;
            // println!("Input +eps ptr: {:p}", input_plus_eps.id_ptr()); // <-- DEBUG PTR (Remove)
            let input_minus_eps = create_perturbed_input(-epsilon)?;
            // println!("Input -eps ptr: {:p}", input_minus_eps.id_ptr()); // <-- DEBUG PTR (Remove)

            // --- Define Loss Calculation Closure First ---
            let calculate_loss = |output_tensor: Tensor<T>| -> Result<T, GradCheckError<T>> {
                // Note: mul_op handles internal broadcasting.
                let product = mul_op(&output_tensor, output_grad) // Use the mul_op function
                    .map_err(|e| GradCheckError::TensorError(e))?;

                // Sum all elements of the product tensor using sum_axes
                let total_loss_tensor =
                    sum_axes(&product, &[], false) // Use the sum_axes function
                        .map_err(|e| GradCheckError::TensorError(e))?;

                // The result of sum_axes is a scalar tensor, get its value using get(&[])
                let loss_value = total_loss_tensor
                    .get(&[])
                    .map_err(|e| GradCheckError::TensorError(e))?;
                // DEBUG: Print calculated loss (inside the closure) (Remove)
                // println!("--> Loss calculated inside closure: {:?}", loss_value);
                Ok(loss_value)
            };

            // --- 5.c & 5.d: Call func with perturbed inputs ---
            let mut inputs_for_plus: Vec<Tensor<T>> = inputs.iter().cloned().collect();
            inputs_for_plus[i] = input_plus_eps;
            // println!("Calling func for +eps for input {}, elem {}", i, elem_idx); // DEBUG (Remove)
            let output_plus_eps =
                func(&inputs_for_plus).map_err(GradCheckError::ForwardPassError)?;
            // --- Removed Println for Data ---
            // match output_plus_eps.read_data().data.cpu_data() {
            //     Ok(data_arc) => println!("Output +eps data: {:?}", data_arc.as_slice()),
            //     Err(e) => println!("Output +eps data: Error getting CPU data - {:?}", e),
            // }
            // ---------------------------
            // println!("Called func for +eps. Calling calculate_loss for +eps."); // DEBUG (Remove)
            let loss_plus_eps = calculate_loss(output_plus_eps)?;
            // println!("Calculated loss_plus_eps: {:?}", loss_plus_eps); // DEBUG (Remove)

            let mut inputs_for_minus: Vec<Tensor<T>> = inputs.iter().cloned().collect();
            inputs_for_minus[i] = input_minus_eps;
            // println!("Calling func for -eps for input {}, elem {}", i, elem_idx); // DEBUG (Remove)
            let output_minus_eps =
                func(&inputs_for_minus).map_err(GradCheckError::ForwardPassError)?;
            // --- Removed Println for Data ---
            // match output_minus_eps.read_data().data.cpu_data() {
            //     Ok(data_arc) => println!("Output -eps data: {:?}", data_arc.as_slice()),
            //     Err(e) => println!("Output -eps data: Error getting CPU data - {:?}", e),
            // }
            // ---------------------------
            // println!("Called func for -eps. Calling calculate_loss for -eps."); // DEBUG (Remove)
            let loss_minus_eps = calculate_loss(output_minus_eps)?;
            // println!("Calculated loss_minus_eps: {:?}", loss_minus_eps); // DEBUG (Remove)

            // --- 5.g: Calculate Numerical Gradient ---
            // Note: Sections 5.e & 5.f (manual diff_sum calc) are removed as loss is calculated via closure.
            let numerical_grad = (loss_plus_eps - loss_minus_eps) / (two * epsilon);
            // DEBUG: Print numerical gradient (Remove)
            // Use loss_plus_eps and loss_minus_eps in the format string
            // println!("Input {}, Elem {}: Numerical Grad = ({:?} - {:?}) / (2 * {:?}) = {:?}", i, elem_idx, loss_plus_eps, loss_minus_eps, epsilon, numerical_grad);

            // --- 5.h: Get analytical gradient ---
            // Assuming analytical_grad_tensor is contiguous for now
            if !analytical_grad_tensor.is_contiguous() {
                return Err(GradCheckError::TensorError(NeuraRustError::UnsupportedOperation(
                    "Gradient checking on non-contiguous analytical gradients not yet supported.".to_string()
                )));
            }
            let analytical_data_arc = analytical_grad_tensor
                .read_data()
                .data
                .cpu_data()
                .map_err(|e| GradCheckError::AnalyticalGradAccessError {
                    input_index: i,
                    source: e,
                })?
                .clone(); // Map error specifically
                          // TODO: Handle strides for analytical grad access if non-contiguous
            let analytical_linear_idx = analytical_grad_tensor.read_data().offset + elem_idx;
            if analytical_linear_idx >= analytical_data_arc.len() {
                return Err(GradCheckError::TensorError(NeuraRustError::InternalError(
                    format!(
                        "Calculated analytical linear index {} out of bounds for buffer len {}.",
                        analytical_linear_idx,
                        analytical_data_arc.len()
                    ),
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
