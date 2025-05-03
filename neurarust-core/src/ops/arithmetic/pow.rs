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

// --- pow_kernel (Private Calculation Core) ---

/// Private kernel for element-wise power calculation with broadcasting.
fn pow_kernel<T>(
    output_shape: &[usize],
    base_data: &[T],
    base_shape: &[usize],
    base_strides: &[usize],
    base_offset: usize,
    exponent_data: &[T],
    exponent_shape: &[usize],
    exponent_strides: &[usize],
    exponent_offset: usize,
) -> Result<Vec<T>, NeuraRustError>
where
    T: Float + Copy + Zero,
{
    let output_numel = output_shape.iter().product::<usize>();
    let mut output_data = vec![T::zero(); output_numel];
    let mut current_coords = vec![0; output_shape.len()];
    // No need for output_broadcast_strides if we index output_data linearly

    for i in 0..output_numel {
        // Calculate physical index into base buffer, handling broadcast
        let mut base_physical_idx = base_offset;
        for dim in 0..base_shape.len() {
            let broadcast_dim_offset = output_shape.len() - base_shape.len();
            let coord_idx = broadcast_dim_offset + dim;
            // Use usize for indexing and comparison
            let index = if base_shape[dim] == 1 && output_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
            base_physical_idx += index * base_strides[dim];
        }

        // Calculate physical index into exponent buffer, handling broadcast
        let mut exp_physical_idx = exponent_offset;
        for dim in 0..exponent_shape.len() {
            let broadcast_dim_offset = output_shape.len() - exponent_shape.len();
            let coord_idx = broadcast_dim_offset + dim;
            // Use usize for indexing and comparison
            let index = if exponent_shape[dim] == 1 && output_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
            exp_physical_idx += index * exponent_strides[dim];
        }

        // Bounds check (optional but safer)
        if base_physical_idx >= base_data.len() || exp_physical_idx >= exponent_data.len() {
            return Err(NeuraRustError::InternalError(
                "Pow kernel index out of bounds".to_string()
            ));
        }

        let base_val = base_data[base_physical_idx];
        let exp_val = exponent_data[exp_physical_idx];

        // Core calculation
        output_data[i] = base_val.powf(exp_val);

        // Increment multi-dimensional coords for the next iteration
        if i < output_numel - 1 {
            let mut dim_to_inc = output_shape.len();
            while dim_to_inc > 0 {
                dim_to_inc -= 1;
                current_coords[dim_to_inc] += 1;
                if current_coords[dim_to_inc] < output_shape[dim_to_inc] {
                    break;
                }
                current_coords[dim_to_inc] = 0;
            }
        }
    }
    Ok(output_data)
}

// --- pow_op Implementation (Public API + Autograd Setup) ---

/// Computes element-wise power of `base` raised to `exponent`.
/// Supports broadcasting.
pub fn pow_op<T>(base: &Tensor<T>, exponent: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Float // Requires Float for powf and backward ln
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
        + Div<Output = T> // For backward
        + Neg<Output = T> // For backward
        + Default
        + PartialEq
        + PartialOrd
        + std::iter::Sum
        + std::iter::Product,
{
    let base_requires_grad = base.requires_grad();
    let exponent_requires_grad = exponent.requires_grad();
    let requires_grad = base_requires_grad || exponent_requires_grad;

    let base_guard = base.read_data();
    let exponent_guard = exponent.read_data();

    // --- Device Checks --- (Simplified: Assume CPU)
    if base_guard.device != StorageDevice::CPU || exponent_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "Pow currently only supports CPU".to_string(),
        ));
    }

    // --- Shape Broadcasting ---
    let output_shape = broadcast_shapes(&base_guard.shape, &exponent_guard.shape)?;

    // --- Extract data for kernel ---
    let base_data_arc = base_guard.data.cpu_data()?.clone();
    let exponent_data_arc = exponent_guard.data.cpu_data()?.clone();
    let base_data_slice = base_data_arc.as_slice();
    let exponent_data_slice = exponent_data_arc.as_slice();

    // Clone shapes, strides, offsets BEFORE dropping guards
    let base_shape = base_guard.shape.clone();
    let base_strides = base_guard.strides.clone();
    let base_offset = base_guard.offset;
    let exponent_shape = exponent_guard.shape.clone();
    let exponent_strides = exponent_guard.strides.clone();
    let exponent_offset = exponent_guard.offset;

    // Clone Arcs needed for backward pass *before* dropping guards
    let base_node_arc = if requires_grad { Some(base.data.clone()) } else { None };
    let exponent_node_arc = if requires_grad { Some(exponent.data.clone()) } else { None };

    // Drop guards after getting all necessary data/arcs
    drop(base_guard);
    drop(exponent_guard);

    // --- Calculation via Kernel ---
    let output_data = pow_kernel(
        &output_shape,
        base_data_slice,
        &base_shape,
        &base_strides,
        base_offset,
        exponent_data_slice,
        &exponent_shape,
        &exponent_strides,
        exponent_offset,
    )?;

    // --- Create Output Tensor ---
    let output_tensor = Tensor::new(output_data, output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let (Some(base_arc), Some(exp_arc)) = (base_node_arc, exponent_node_arc) {
            // Clone output Arc for backward context
            let output_node_arc = output_tensor.data.clone();
            let grad_fn = PowBackward {
                base_node: base_arc,
                exponent_node: exp_arc,
                output_node: output_node_arc, // Store output for grad_exponent
                base_requires_grad,
                exponent_requires_grad,
            };
            output_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
            output_tensor.set_requires_grad(true)?;
        } else {
            // This case should ideally not happen if requires_grad is true
            return Err(NeuraRustError::InternalError(
                "Input requires_grad but Arc could not be cloned for Pow".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
#[path = "pow_test.rs"]
mod tests; 