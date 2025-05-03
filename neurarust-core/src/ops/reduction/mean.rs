use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::device::StorageDevice;
use crate::ops::reduction::sum::sum_kernel;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLock, RwLockReadGuard};
use crate::types::DType;

// --- MeanBackward Definition ---

/// Backward operation context for `mean_axes`.
#[derive(Debug)]
struct MeanBackward {
    input_node: Arc<RwLock<TensorData>>,
    num_elements_reduced: usize,
}

// --- BackwardOp Implementation for MeanBackward ---

impl BackwardOp for MeanBackward {
    fn backward(&self, _grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // TODO: Needs expand_op (which requires its own backward) and mul_op
        // TODO: Also needs the scaling factor
        let _scaling_factor = 1.0f32 / (self.num_elements_reduced as f32);

        todo!("Implement Mean backward using expand_op and mul_op");
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)] // Return NodeId from Arc
    }
}

// --- Kernel de Calcul (F32 CPU) ---

/// Noyau de calcul privé pour la moyenne avec réduction d'axes. F32 CPU.
fn mean_kernel(
    input_guard: &RwLockReadGuard<'_, TensorData>,
    input_data_slice: &[f32],
    axes: &[usize],
    keep_dims: bool,
    output_shape: &[usize],
    n: f32,                   // Use f32 for divisor
) -> Result<Vec<f32>, NeuraRustError>
{
    // 1. Calculer la somme en utilisant le kernel de sum (expects f32)
    let sum_data = sum_kernel(input_guard, input_data_slice, axes, keep_dims, output_shape)?;

    // 2. Diviser chaque élément par N
    if n == 0.0f32 {
        return Err(NeuraRustError::DivisionByZero);
    }

    // Division f32 / f32
    let mean_data: Vec<f32> = sum_data.into_iter().map(|val| val / n).collect();

    Ok(mean_data)
}

// --- mean_axes Implementation (Public API - F32 CPU) ---

/// Calculates the mean of elements along specified axes. F32 CPU only.
pub fn mean_axes(
    input: &Tensor,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor, NeuraRustError>
{
    let requires_grad = input.requires_grad();
    let input_node_arc = if requires_grad { Some(input.data.clone()) } else { None };

    let input_guard = input.read_data();

    // --- Device and DType Check ---
    if input_guard.device != StorageDevice::CPU || input_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Mean operation is currently only supported on F32 CPU tensors, not {:?}/{:?}",
            input_guard.device, input_guard.dtype
        )));
    }

    // --- Get CPU Data Buffer ---
    // Correct access using buffer()
    let input_data_arc = input_guard.buffer().try_get_cpu_f32()?.clone();
    let input_data_slice = input_data_arc.as_slice();

    // --- Shape and Axis Validation / Calculation de N ---
    let input_shape = &input_guard.shape;
    let input_rank = input_shape.len();

    let processed_axes = {
        let mut pa = Vec::new();
        if !axes.is_empty() {
            for &axis in axes {
                if axis >= input_rank {
                     // Correct error type
                    return Err(NeuraRustError::DimensionMismatch {
                        expected: input_rank,
                        actual: axis,
                    });
                }
                if !pa.contains(&axis) { // Avoid duplicates
                    pa.push(axis);
                }
            }
            pa.sort_unstable();
        }
        pa
    };

    let n: usize = {
        if processed_axes.is_empty() {
            input_guard.numel()
        } else {
            processed_axes.iter().map(|&axis| input_shape[axis]).product()
        }
    };

    // Convert n to f32 for the kernel
    let n_f32 = n as f32;
    if n == 0 {
         // Handle case where reduction results in zero elements (e.g., reducing a dim of size 0)
         // Division by zero will be handled in mean_kernel, but we might return early?
         // For now, let mean_kernel handle it.
         // However, if n_f32 is 0.0 due to large usize, that's an issue.
         if n_f32 == 0.0 && n > 0 { // Check potential overflow/precision loss
             return Err(NeuraRustError::InternalError(
                "Element count N is too large to represent accurately as f32".to_string()
             ));
         }
    }

    // --- Calculate Output Shape ---
    let output_shape: Vec<usize> = {
        let mut shape = Vec::new();
        for (dim, &size) in input_shape.iter().enumerate() {
            if !processed_axes.contains(&dim) {
                shape.push(size);
            } else if keep_dims {
                shape.push(1);
            }
        }
         // Handle reduction to scalar or empty tensor cases
        if shape.is_empty() && input_rank > 0 && !keep_dims { // Reduced all dims, or input was vector reduced
            vec![]
        } else if shape.is_empty() && input_rank > 0 && keep_dims {
            vec![1; input_rank] // Keep original rank with 1s
        } else if input_rank == 0 {
            vec![] // Input was scalar
        } else {
            shape
        }
    };


    // --- Perform Mean Calculation (Appel au Kernel F32) ---
    // Clone necessary data before dropping guard
    let kernel_axes = processed_axes.clone();
    let kernel_output_shape = output_shape.clone();

    let result_data = mean_kernel(
        &input_guard,
        input_data_slice,
        &kernel_axes,
        keep_dims,
        &kernel_output_shape,
        n_f32,
    )?;

    // Drop lock
    drop(input_guard);

    // --- Create Result Tensor (expects Vec<f32>) ---
    let result_tensor = Tensor::new(result_data, kernel_output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let Some(node_arc) = input_node_arc {
            let backward_context = MeanBackward {
                input_node: node_arc,
                num_elements_reduced: n,
            };
            let backward_op_arc: Arc<dyn BackwardOp + Send + Sync> = Arc::new(backward_context);

            let mut result_tensor_guard = result_tensor.write_data();
            result_tensor_guard.requires_grad = true;
            result_tensor_guard.grad_fn = Some(backward_op_arc);
        } else {
             return Err(NeuraRustError::InternalError(
                "Mean requires grad but input Arc was not available".to_string(),
            ));
        }
    }

    Ok(result_tensor)
}

// Wrapper function (Kept for compatibility for now)
pub(crate) fn mean_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let all_axes: Vec<usize> = (0..tensor.shape().len()).collect();
    let axes_to_mean = axes.unwrap_or(&all_axes);
    mean_axes(tensor, axes_to_mean, keep_dims)
}

// --- Tests ---
#[cfg(test)]
#[path = "mean_test.rs"]
mod tests; 