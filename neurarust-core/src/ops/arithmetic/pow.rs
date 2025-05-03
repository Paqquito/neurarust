use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, index_to_coord};

use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg};
use std::sync::{Arc, RwLock};

// --- PowBackward Definition ---

/// Backward operation context for `pow_op`.
#[derive(Debug)]
struct PowBackward<T: Float + Debug + Copy + Send + Sync + 'static> {
    base_node: Arc<RwLock<TensorData<T>>>, // Need base for grad_exponent (ln(base))
    exponent_node: Arc<RwLock<TensorData<T>>>, // Need exponent for grad_base
    output_node: Arc<RwLock<TensorData<T>>>, // Need output for grad_exponent
    base_requires_grad: bool,
    exponent_requires_grad: bool,
}

// --- BackwardOp Implementation for PowBackward (Manual Element-wise) ---

impl<T> BackwardOp<T> for PowBackward<T>
where
    T: Float // Requires Float for powf, ln
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Zero
        + One
        + Add<Output = T>
        + AddAssign
        + Mul<Output = T>
        + Div<Output = T> // For exponent - 1 (if T=Float)
        + Neg<Output = T> // Might be needed by mul/div backward
        + Default
        + PartialEq
        + PartialOrd
        + std::iter::Sum
        + std::iter::Product, // Added Product
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let mut input_grads: Vec<Tensor<T>> = Vec::with_capacity(2);

        let base_tensor = Tensor { data: self.base_node.clone() };
        let exponent_tensor = Tensor { data: self.exponent_node.clone() };
        let output_tensor = Tensor { data: self.output_node.clone() };

        let base_guard = base_tensor.read_data();
        let exponent_guard = exponent_tensor.read_data();
        let output_guard = output_tensor.read_data(); // Needed for grad_exponent
        let grad_output_guard = grad_output.read_data();

        // --- Device Checks (Simplified: Assume CPU) ---
        if base_guard.device != StorageDevice::CPU
            || exponent_guard.device != StorageDevice::CPU
            || output_guard.device != StorageDevice::CPU
            || grad_output_guard.device != StorageDevice::CPU
        {
            return Err(NeuraRustError::UnsupportedOperation(
                "Pow backward currently only supports CPU".to_string(),
            ));
        }

        // Get buffers
        let base_buffer = base_guard.data.cpu_data()?.clone();
        let exponent_buffer = exponent_guard.data.cpu_data()?.clone();
        let output_buffer = output_guard.data.cpu_data()?.clone(); // Needed for grad_exponent
        let grad_output_buffer = grad_output_guard.data.cpu_data()?.clone();

        // Determine broadcast shape for the *operation output*
        let output_broadcast_shape = broadcast_shapes(&base_guard.shape, &exponent_guard.shape)?;
        let output_broadcast_numel = output_broadcast_shape.iter().product::<usize>();
        // Assume grad_output has this broadcast shape
        if grad_output_guard.shape != output_broadcast_shape {
            return Err(NeuraRustError::ShapeMismatch {
                expected: output_broadcast_shape.clone(), // Clone here
                actual: grad_output_guard.shape.clone(),
                operation: "pow_backward (grad_output shape)".to_string(),
            });
        }
        // Strides for iterating through the broadcasted shape contiguously
        let output_broadcast_strides = TensorData::<T>::calculate_contiguous_strides(&output_broadcast_shape);

        // --- Calculate Gradient for Base --- dL/dbase = dL/doutput * exponent * base.powf(exponent - 1)
        if self.base_requires_grad {
            let mut grad_base_data = vec![T::zero(); output_broadcast_numel];
            let one = T::one();
            let mut current_coords = vec![0; output_broadcast_shape.len()];

            for i in 0..output_broadcast_numel {
                // Calculate multi-dimensional coords from linear index `i`
                // Note: index_to_coord requires strides of the shape we are iterating
                let _coords_ignored = index_to_coord(i, &output_broadcast_shape, &output_broadcast_strides); // Use current_coords instead

                // Calculate physical index into base buffer, handling broadcast
                let mut base_physical_idx = base_guard.offset;
                for dim in 0..base_guard.shape.len() {
                    let broadcast_dim_offset = output_broadcast_shape.len() - base_guard.shape.len();
                    let coord_idx = broadcast_dim_offset + dim;
                    let index = if base_guard.shape[dim] == 1 && output_broadcast_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
                    base_physical_idx += index * base_guard.strides[dim];
                }

                // Calculate physical index into exponent buffer, handling broadcast
                let mut exp_physical_idx = exponent_guard.offset;
                for dim in 0..exponent_guard.shape.len() {
                    let broadcast_dim_offset = output_broadcast_shape.len() - exponent_guard.shape.len();
                    let coord_idx = broadcast_dim_offset + dim;
                    let index = if exponent_guard.shape[dim] == 1 && output_broadcast_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
                    exp_physical_idx += index * exponent_guard.strides[dim];
                }

                // Calculate physical index into grad_output buffer (use its strides)
                let mut grad_out_physical_idx = grad_output_guard.offset;
                for dim in 0..output_broadcast_shape.len() {
                     grad_out_physical_idx += current_coords[dim] * grad_output_guard.strides[dim];
                }

                let base_val = base_buffer[base_physical_idx];
                let exp_val = exponent_buffer[exp_physical_idx];
                let grad_out_val = grad_output_buffer[grad_out_physical_idx];

                // grad_base = grad_output * exponent * base.powf(exponent - 1)
                let grad_val = grad_out_val * exp_val * base_val.powf(exp_val - one);
                grad_base_data[i] = grad_val; // Store in linear index i

                // Increment multi-dimensional coords for the next iteration
                if i < output_broadcast_numel - 1 {
                    let mut dim_to_inc = output_broadcast_shape.len();
                    while dim_to_inc > 0 {
                        dim_to_inc -= 1;
                        current_coords[dim_to_inc] += 1;
                        if current_coords[dim_to_inc] < output_broadcast_shape[dim_to_inc] {
                            break;
                        }
                        current_coords[dim_to_inc] = 0;
                    }
                }
            }
            let grad_base_unreduced = Tensor::new(grad_base_data, output_broadcast_shape.clone())?;
            // Reduce gradient if base was broadcasted
            let grad_base = grad_base_unreduced.reduce_to_shape(&base_guard.shape)?;
            input_grads.push(grad_base);
        }

        // --- Calculate Gradient for Exponent --- dL/dexponent = dL/doutput * output * ln(base)
        if self.exponent_requires_grad {
            let mut grad_exp_data = vec![T::zero(); output_broadcast_numel];
            let mut current_coords = vec![0; output_broadcast_shape.len()]; // Reset coords for this calculation

            for i in 0..output_broadcast_numel {
                // Calculate multi-dimensional coords from linear index `i`
                let _coords_ignored = index_to_coord(i, &output_broadcast_shape, &output_broadcast_strides);

                // Calculate index into base buffer (same as above)
                let mut base_physical_idx = base_guard.offset;
                 for dim in 0..base_guard.shape.len() {
                    let broadcast_dim_offset = output_broadcast_shape.len() - base_guard.shape.len();
                    let coord_idx = broadcast_dim_offset + dim;
                    let index = if base_guard.shape[dim] == 1 && output_broadcast_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
                    base_physical_idx += index * base_guard.strides[dim];
                }

                // Calculate index into output buffer (use its strides)
                let mut output_physical_idx = output_guard.offset;
                 for dim in 0..output_guard.shape.len() { // Use output_guard.shape.len() - assumes matches broadcast
                    let index = current_coords[dim];
                     if dim < output_guard.strides.len() { // Safety check
                        output_physical_idx += index * output_guard.strides[dim];
                     } else {
                         return Err(NeuraRustError::InternalError("Output shape/stride mismatch in PowBackward".to_string()));
                     }
                 }

                // Calculate index into grad_output buffer (same as above)
                let mut grad_out_physical_idx = grad_output_guard.offset;
                 for dim in 0..output_broadcast_shape.len() {
                     grad_out_physical_idx += current_coords[dim] * grad_output_guard.strides[dim];
                 }

                let base_val = base_buffer[base_physical_idx];
                let output_val = output_buffer[output_physical_idx];
                let grad_out_val = grad_output_buffer[grad_out_physical_idx];

                if base_val <= T::zero() {
                    return Err(NeuraRustError::UnsupportedOperation(
                        "Gradient calculation for pow exponent requires base > 0 for ln(base)"
                            .to_string(),
                    ));
                }
                let ln_base_val = base_val.ln();

                // grad_exponent = grad_output * output * ln(base)
                let grad_val = grad_out_val * output_val * ln_base_val;
                grad_exp_data[i] = grad_val;

                // Increment multi-dimensional coords (same as above)
                if i < output_broadcast_numel - 1 {
                    let mut dim_to_inc = output_broadcast_shape.len();
                    while dim_to_inc > 0 {
                        dim_to_inc -= 1;
                        current_coords[dim_to_inc] += 1;
                        if current_coords[dim_to_inc] < output_broadcast_shape[dim_to_inc] {
                            break;
                        }
                        current_coords[dim_to_inc] = 0;
                    }
                }
            }
            let grad_exponent_unreduced = Tensor::new(grad_exp_data, output_broadcast_shape.clone())?;
            // Reduce gradient if exponent was broadcasted
            let grad_exponent = grad_exponent_unreduced.reduce_to_shape(&exponent_guard.shape)?;
            input_grads.push(grad_exponent);
        }

        Ok(input_grads)
    }

    fn inputs(&self) -> Vec<NodeId<T>> {
        let mut ids = Vec::new();
        if self.base_requires_grad { ids.push(Arc::as_ptr(&self.base_node)); }
        if self.exponent_requires_grad { ids.push(Arc::as_ptr(&self.exponent_node)); }
        ids
    }
}

// --- pow_op Implementation (Manual Element-wise Forward) ---

/// Computes element-wise power of `base` raised to `exponent`.
/// Supports broadcasting.
pub fn pow_op<T>(base: &Tensor<T>, exponent: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Float // Requires Float for powf
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Zero
        + One
        + Add<Output = T>
        + AddAssign
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Default
        + PartialEq
        + PartialOrd
        + std::iter::Sum
        + std::iter::Product,
{
    let base_requires_grad = base.requires_grad();
    let exponent_requires_grad = exponent.requires_grad();
    let requires_grad = base_requires_grad || exponent_requires_grad;

    let base_node_arc = if requires_grad { Some(base.data.clone()) } else { None };
    let exponent_node_arc = if requires_grad { Some(exponent.data.clone()) } else { None };

    let base_guard = base.read_data();
    let exponent_guard = exponent.read_data();

    // --- Device Checks (Simplified: Assume CPU) ---
     if base_guard.device != StorageDevice::CPU || exponent_guard.device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
             "Pow forward currently only supports CPU".to_string()
         ));
     }

    // --- Broadcasting and Forward Calculation ---
    let output_shape = broadcast_shapes(&base_guard.shape, &exponent_guard.shape)?;
    let output_numel = output_shape.iter().product::<usize>();
    let mut output_data = vec![T::zero(); output_numel];

    let base_buffer = base_guard.data.cpu_data()?.clone();
    let exponent_buffer = exponent_guard.data.cpu_data()?.clone();

    // Strides for iterating through the broadcasted output shape contiguously
    let output_strides = TensorData::<T>::calculate_contiguous_strides(&output_shape);
    let mut current_coords = vec![0; output_shape.len()];

    for i in 0..output_numel {
        // Calculate multi-dimensional coords from linear index `i`
        let _coords_ignored = index_to_coord(i, &output_shape, &output_strides); // Use current_coords instead

        // Calculate physical index into base buffer, handling broadcast
        let mut base_physical_idx = base_guard.offset;
         for dim in 0..base_guard.shape.len() {
             let broadcast_dim_offset = output_shape.len() - base_guard.shape.len();
             let coord_idx = broadcast_dim_offset + dim;
             let index = if base_guard.shape[dim] == 1 && output_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
             base_physical_idx += index * base_guard.strides[dim];
         }

        // Calculate physical index into exponent buffer, handling broadcast
        let mut exp_physical_idx = exponent_guard.offset;
         for dim in 0..exponent_guard.shape.len() {
             let broadcast_dim_offset = output_shape.len() - exponent_guard.shape.len();
             let coord_idx = broadcast_dim_offset + dim;
             let index = if exponent_guard.shape[dim] == 1 && output_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
             exp_physical_idx += index * exponent_guard.strides[dim];
         }

        let base_val = base_buffer[base_physical_idx];
        let exp_val = exponent_buffer[exp_physical_idx];

        output_data[i] = base_val.powf(exp_val);

        // Increment multi-dimensional coords for the next iteration
         if i < output_numel - 1 {
             let mut dim_to_inc = output_shape.len();
             while dim_to_inc > 0 {
                 dim_to_inc -= 1;
                 current_coords[dim_to_inc] += 1;
                 if current_coords[dim_to_inc] < output_shape[dim_to_inc] { break; }
                 current_coords[dim_to_inc] = 0;
             }
         }
    }

    let output_tensor = Tensor::new(output_data, output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let (Some(base_arc), Some(exp_arc)) = (base_node_arc, exponent_node_arc) {
            let grad_fn = PowBackward {
                base_node: base_arc,
                exponent_node: exp_arc,
                output_node: output_tensor.data.clone(), // Pass clone of output tensor data
                base_requires_grad,
                exponent_requires_grad,
            };
            output_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
            output_tensor.set_requires_grad(true)?;
        } else {
             return Err(NeuraRustError::InternalError(
                 "Input requires_grad but Arc could not be cloned for Pow".to_string(),
             ));
        }
    }

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::Tensor;
    use approx::assert_relative_eq;

    // Helper for f64 tests
    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_pow_forward() {
        let base = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let exp = create_tensor_f64_with_grad(vec![2.0, 3.0, 0.5], vec![3]);
        let output = pow_op(&base, &exp).unwrap();
        let expected_data = vec![1.0f64.powf(2.0), 2.0f64.powf(3.0), 3.0f64.powf(0.5)];
        let output_data = output.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(output.shape(), vec![3]);
         output_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e, epsilon = 1e-7));
    }

    #[test]
    fn test_pow_forward_broadcast() {
        let base = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let exp = create_tensor_f64_with_grad(vec![2.0], vec![1]); // Scalar exponent effectively
        let output = pow_op(&base, &exp).unwrap();
        let expected_data = vec![1.0, 4.0, 9.0, 16.0];
        let output_data = output.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(output.shape(), vec![2, 2]);
         output_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e, epsilon = 1e-7));
    }

    #[test]
    fn test_pow_backward_simple() {
        let base = create_tensor_f64_with_grad(vec![2.0, 3.0], vec![2]);
        let exp = create_tensor_f64_with_grad(vec![3.0, 2.0], vec![2]);
        let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[base, exp], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Pow simple backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_pow_backward_broadcast_exponent() {
         let base = create_tensor_f64_with_grad(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
         let exp = create_tensor_f64_with_grad(vec![2.0], vec![1]); // Broadcast exponent
         let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &inputs[1]);

         let output_shape = vec![2, 2];
         let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
         let epsilon = 1e-5;
         let tolerance = 1e-7;

         let grad_check_result = check_grad(func, &[base, exp], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Pow broadcast exponent backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_pow_backward_broadcast_base() {
         let base = create_tensor_f64_with_grad(vec![2.0], vec![1]); // Broadcast base
         let exp = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
         let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &inputs[1]);

         let output_shape = vec![2, 2];
         let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
         let epsilon = 1e-5;
         let tolerance = 1e-6;

         let grad_check_result = check_grad(func, &[base, exp], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Pow broadcast base backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
     fn test_pow_backward_only_base_grad() {
         let base = create_tensor_f64_with_grad(vec![2.0, 3.0], vec![2]);
         let exp = Tensor::new(vec![3.0, 2.0], vec![2]).unwrap(); // Exponent does not require grad
         let func = |inputs: &[Tensor<f64>]| pow_op(&inputs[0], &exp); // Pass exp by reference

         let output_shape = vec![2];
         let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
         let epsilon = 1e-5;
         let tolerance = 1e-7;

         // Check grad only for base (index 0)
         let grad_check_result = check_grad(func, &[base.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Pow only base grad check failed: {:?}", grad_check_result.err());

         // Manual check: ensure exponent grad is None
         let output = pow_op(&base, &exp).unwrap();
         output.backward(Some(output_grad)).unwrap();
         assert!(exp.grad().is_none());
         assert!(base.grad().is_some());
     }

      #[test]
     fn test_pow_backward_only_exponent_grad() {
         let base = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap(); // Base does not require grad
         let exp = create_tensor_f64_with_grad(vec![3.0, 2.0], vec![2]);
         let func = |inputs: &[Tensor<f64>]| pow_op(&base, &inputs[0]); // Pass base by reference

         let output_shape = vec![2];
         let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
         let epsilon = 1e-5;
         let tolerance = 1e-6; // ln might need slightly higher tolerance

         // Check grad only for exponent (index 0 in the slice passed to check_grad)
         let grad_check_result = check_grad(func, &[exp.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Pow only exponent grad check failed: {:?}", grad_check_result.err());

          // Manual check: ensure base grad is None
         let output = pow_op(&base, &exp).unwrap();
         output.backward(Some(output_grad)).unwrap();
         assert!(base.grad().is_none());
         assert!(exp.grad().is_some());
     }
} 