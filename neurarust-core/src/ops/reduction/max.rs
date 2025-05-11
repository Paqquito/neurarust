use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;

// Add imports needed for backward
use crate::ops::comparison::equal_op; 
use crate::ops::arithmetic::mul_op; 
use crate::ops::view::{expand_op, reshape_op}; 

// Importer les utilitaires de réduction
use super::utils::{process_reduction_axes, calculate_reduction_output_shape, calculate_grad_broadcast_shape};

use std::sync::{Arc, RwLock};
use std::fmt::Debug;
use num_traits::Float; // Use Float for min_value()

/// Backward operation context for `max` reduction.
///
/// Stores information needed to compute the gradient of the max operation:
/// - A reference to the original input tensor's data (`input_node`).
/// - A reference to the output tensor's data (`output_node`) from the forward pass.
///   This is needed because the gradient only flows back to the input elements
///   that were equal to the maximum value in their respective reduction slice.
/// - The axes along which the reduction was performed (`axes`).
/// - Whether the reduced dimensions were kept in the output (`keep_dims`).
#[derive(Debug)]
struct MaxBackward {
    input_node: Arc<RwLock<TensorData>>,
    output_node: Arc<RwLock<TensorData>>, // Store output tensor to compare with input
    axes: Option<Vec<usize>>,
    keep_dims: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for MaxBackward {
    /// Computes the gradient for the max reduction operation.
    ///
    /// The gradient flows back only to the element(s) in the input tensor that contributed
    /// to the maximum value along the reduced axes. This is achieved by creating a mask
    /// where `input == output` (after appropriate expansion/reshaping) and multiplying
    /// the incoming gradient by this mask.
    ///
    /// Gradient = `expand(grad_output) * (input == expand(output))`
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original max operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with a single element: the gradient
    /// with respect to the original input tensor. Returns an error if reshaping, expansion,
    /// comparison, multiplication, or device operations fail.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // Backward of max: grad_input = expanded_grad_output * (input == expanded_output)
        
        // 1. Recreate tensors from stored Arcs & get shapes
        let input_tensor = Tensor { data: self.input_node.clone() };
        let output_tensor = Tensor { data: self.output_node.clone() };
        let input_shape = input_tensor.shape();
        let grad_output_shape = grad_output.shape(); // Shape du gradient entrant

        // 2. Determine target shape for grad_output using utility
        //    Handles None axes case implicitly via process_reduction_axes used during forward pass
        let axes_vec = self.axes.as_deref().unwrap_or(&[]); // Get axes slice, empty if None
        let target_grad_shape = calculate_grad_broadcast_shape(
            &input_shape,
            &grad_output_shape, 
            axes_vec,
            self.keep_dims,
        );

        // 3. Reshape grad_output if needed
        let grad_output_reshaped = if grad_output_shape != target_grad_shape {
            // TODO: Consider grad_output.contiguous() before reshape? Seems less critical here
            // as expand later will handle non-contiguous, but might be cleaner.
            reshape_op(grad_output, target_grad_shape)?
        } else {
            grad_output.clone()
        };

        // 4. Prepare output_tensor for expansion (Only needed if keep_dims=false originally)
        //    We still need to potentially reshape the *original* output if keep_dims was false,
        //    so it can be expanded for the mask comparison.
        let output_to_expand = if !self.keep_dims && self.axes.is_some() {
            // Calculate the shape with ones inserted (same logic as before, but only for output)
            let axes_vec = self.axes.as_ref().unwrap(); // Already checked is_some
            let mut target_shape_with_ones = input_shape.clone();
            for &axis in axes_vec {
                if axis < target_shape_with_ones.len() {
                    target_shape_with_ones[axis] = 1;
                } else {
                     return Err(NeuraRustError::InternalError(format!(
                        "Invalid axis {} in MaxBackward output reshape for input {:?}", 
                        axis, input_shape
                    )));
                }
            }
            reshape_op(&output_tensor, target_shape_with_ones)?
        } else {
            output_tensor.clone()
        };
        
        // 5. Expand output and (reshaped) grad_output to input shape
        let target_shape_isize: Vec<isize> = input_shape.iter().map(|&d| d as isize).collect();
        let expanded_output = expand_op(&output_to_expand, &target_shape_isize)?;
        let expanded_grad_output = expand_op(&grad_output_reshaped, &target_shape_isize)?;

        // 6. Create the mask: (input == expanded_output)
        let mask = equal_op(&input_tensor, &expanded_output)?;
        
        // 7. Calculate grad_input = expanded_grad_output * mask
        let grad_input = mul_op(&expanded_grad_output, &mask)?;

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Kernel for max reduction on CPU for f32.
fn max_cpu_f32_kernel(
    input_data: &[f32],
    input_shape: &[usize],
    input_strides: &[usize],
    input_offset: usize,
    output_shape: &[usize],
    axes: &[usize], // Utiliser axes directement
    _keep_dims: bool // Préfixer car non utilisé dans la logique interne du kernel max
) -> Result<Vec<f32>, NeuraRustError> {
    let output_numel = output_shape.iter().product();
    let mut output_data = vec![f32::min_value(); output_numel];
    let input_rank = input_shape.len();
    let input_numel = input_shape.iter().product();

    // Recalculer reduced_axes_mask basé sur axes et input_rank
    let mut reduced_axes_mask = vec![false; input_rank];
    if axes.is_empty() { // Si axes est vide (via process_reduction_axes), on réduit tout
        reduced_axes_mask.fill(true);
    } else {
        for &axis in axes {
            if axis < input_rank { // Vérification (devrait être déjà faite)
                reduced_axes_mask[axis] = true;
            }
        }
    }

    let output_rank = output_shape.len();
    let mut input_indices = vec![0; input_rank];

    for i in 0..input_numel {
        // Calculate current input coordinates
        let mut current_linear = i;
        for dim in (0..input_rank).rev() {
            let shape_val = input_shape[dim];
            if shape_val > 0 { input_indices[dim] = current_linear % shape_val; current_linear /= shape_val; } else { input_indices[dim] = 0; }
        }

        // Calculate the linear index in the output tensor
        let mut output_linear_index = 0;
        let mut current_output_stride = 1;
        let mut output_dim_idx = output_rank;
        for input_dim_idx in (0..input_rank).rev() {
            if !reduced_axes_mask[input_dim_idx] { // If this axis is NOT reduced
                if output_dim_idx == 0 {
                    return Err(NeuraRustError::InternalError("Output dimension index underflow in max_kernel".to_string()));
                }
                output_dim_idx -= 1;
                output_linear_index += input_indices[input_dim_idx] * current_output_stride;
                current_output_stride *= output_shape[output_dim_idx];
            }
        }
        
        if output_linear_index >= output_numel {
             return Err(NeuraRustError::InternalError(format!(
                 "Output index {} out of bounds ({}) in max_kernel", output_linear_index, output_numel
             )));
        }

        // Calculate physical offset in input buffer
        let mut physical_offset = input_offset;
        for dim in 0..input_rank {
            physical_offset += input_indices[dim] * input_strides[dim];
        }
        if physical_offset >= input_data.len() {
             return Err(NeuraRustError::InternalError(format!(
                 "Input physical offset {} out of bounds ({}) in max_kernel", physical_offset, input_data.len()
             )));
        }
        let value = input_data[physical_offset];

        // Update the max value in the output
        if value > output_data[output_linear_index] {
            output_data[output_linear_index] = value;
        }
    }
    Ok(output_data)
}

// Placeholder pour F64
fn max_cpu_f64_kernel(
    input_data: &[f64],
    input_shape: &[usize],
    input_strides: &[usize],
    input_offset: usize,
    output_shape: &[usize],
    axes: &[usize],
    _keep_dims: bool // Préfixer car non utilisé
) -> Result<Vec<f64>, NeuraRustError> {
     // Similaire à f32 mais avec f64::min_value() et données f64
    let output_numel = output_shape.iter().product();
    let mut output_data = vec![f64::min_value(); output_numel];
    let input_rank = input_shape.len();
    let input_numel = input_shape.iter().product();

    let mut reduced_axes_mask = vec![false; input_rank];
    if axes.is_empty() { reduced_axes_mask.fill(true); } else { for &axis in axes { if axis < input_rank { reduced_axes_mask[axis] = true; } } }

    let output_rank = output_shape.len();
    let mut input_indices = vec![0; input_rank];

    for i in 0..input_numel {
        let mut current_linear = i;
        for dim in (0..input_rank).rev() { let sv = input_shape[dim]; if sv > 0 { input_indices[dim] = current_linear % sv; current_linear /= sv; } else { input_indices[dim] = 0; } }

        let mut output_linear_index = 0;
        let mut current_output_stride = 1;
        let mut output_dim_idx = output_rank;
        for input_dim_idx in (0..input_rank).rev() { if !reduced_axes_mask[input_dim_idx] { if output_dim_idx == 0 { return Err(NeuraRustError::InternalError("Output index underflow (f64)".to_string())); } output_dim_idx -= 1; output_linear_index += input_indices[input_dim_idx] * current_output_stride; current_output_stride *= output_shape[output_dim_idx]; } }
        
        if output_linear_index >= output_numel { return Err(NeuraRustError::InternalError(format!("Output index {} OOB ({}) (f64)", output_linear_index, output_numel))); }

        let mut physical_offset = input_offset;
        for dim in 0..input_rank { physical_offset += input_indices[dim] * input_strides[dim]; }
        if physical_offset >= input_data.len() { return Err(NeuraRustError::InternalError(format!("Input offset {} OOB ({}) (f64)", physical_offset, input_data.len()))); }
        
        let value = input_data[physical_offset];
        if value > output_data[output_linear_index] { output_data[output_linear_index] = value; }
    }
    Ok(output_data)
}

/// Performs element-wise max reduction along specified axes.
pub(crate) fn max_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let t_guard = tensor.read_data();

    // --- Device Check ---
    if t_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "max_op".to_string(),
            expected: StorageDevice::CPU,
            actual: t_guard.device,
        });
    }

    // --- DType Check --- 
    let dtype = t_guard.dtype;
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Max operation only supports F32 and F64, got {:?}",
            dtype
        )));
    }

    // --- Process Axes --- (Utiliser l'utilitaire)
    let rank = t_guard.shape.len();
    let axes_vec = process_reduction_axes(rank, axes)?;

    // --- Calculate Output Shape --- (Utiliser l'utilitaire)
    let output_shape = calculate_reduction_output_shape(&t_guard.shape, &axes_vec, keep_dims);

    // --- Autograd Info --- 
    let requires_grad = t_guard.requires_grad;
    let input_node_arc = if requires_grad { Some(tensor.data.clone()) } else { None };

    // --- Dispatch Kernel --- 
    let output_tensor = match dtype {
        DType::F32 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f32()?.as_slice();
            let result_data = max_cpu_f32_kernel(
                input_data_slice,
                &t_guard.shape,
                &t_guard.strides,
                t_guard.offset,
                &output_shape,
                &axes_vec,
                keep_dims,
            )?;
            drop(t_guard);
            Tensor::new(result_data, output_shape)?
        }
        DType::F64 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f64()?.as_slice();
            let result_data = max_cpu_f64_kernel(
                input_data_slice,
                &t_guard.shape,
                &t_guard.strides,
                t_guard.offset,
                &output_shape,
                &axes_vec,
                keep_dims,
            )?;
            drop(t_guard);
            Tensor::new_f64(result_data, output_shape)?
        }
    };

    // --- Autograd Setup --- 
    if requires_grad {
        if let Some(in_arc) = input_node_arc {
             // MaxBackward a besoin de l'output ET de l'input
            let output_node_arc = output_tensor.data.clone(); 
            let grad_fn = MaxBackward {
                input_node: in_arc,
                output_node: output_node_arc,
                // Stocker les axes traités (Vec<usize>) ou None si réduction globale
                axes: if axes_vec.len() == rank { None } else { Some(axes_vec) }, 
                keep_dims,
            };
            let mut output_guard = output_tensor.write_data();
            output_guard.requires_grad = true;
            output_guard.grad_fn = Some(Arc::new(grad_fn));
        } else {
             return Err(NeuraRustError::InternalError(
                "Max op requires grad but input Arc Node unavailable".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// Revenir à l'attribut #[path] pour lier le fichier de test externe
#[cfg(test)]
#[path = "max_test.rs"]
mod tests; 