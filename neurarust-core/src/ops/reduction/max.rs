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
        // Backward of max: grad_input = grad_output * (input == output_expanded)
        
        // 1. Recreate tensors from stored Arcs
        let input_tensor = Tensor { data: self.input_node.clone() };
        let output_tensor = Tensor { data: self.output_node.clone() };
        let input_shape = input_tensor.shape();

        // 2. Prepare output_tensor and grad_output for expansion (handle keep_dims=false)
        //    If keep_dims=false, we need to reshape output/grad_output to have rank of input
        //    by inserting back the reduced axes with size 1.
        let (output_to_expand, grad_output_to_expand) = 
            if !self.keep_dims && self.axes.is_some() {
                let axes_vec = self.axes.as_ref().unwrap(); // We know it's Some
                let mut target_shape_with_ones = input_shape.clone();
                // Set reduced axes to size 1
                for &axis in axes_vec {
                    if axis < target_shape_with_ones.len() {
                        target_shape_with_ones[axis] = 1;
                    } else {
                        // This case should ideally be caught earlier, but handle defensively
                        return Err(NeuraRustError::InternalError(format!(
                            "Invalid axis {} found during MaxBackward reshape for input shape {:?}", 
                            axis, input_shape
                        )));
                    }
                }
                // Reshape both output and grad_output
                let reshaped_output = reshape_op(&output_tensor, target_shape_with_ones.clone())?;
                let reshaped_grad_output = reshape_op(grad_output, target_shape_with_ones)?;
                (reshaped_output, reshaped_grad_output)
            } else {
                // If keep_dims=true or no axes reduced (scalar), shapes are already compatible for expand
                (output_tensor.clone(), grad_output.clone())
            };
        
        // 3. Expand output and grad_output to input shape
        //    expand_op handles the case where shapes might already match.
        let expanded_output = expand_op(&output_to_expand, input_shape.clone())?;
        let expanded_grad_output = expand_op(&grad_output_to_expand, input_shape.clone())?;

        // 4. Create the mask: (input == expanded_output)
        //    equal_op handles broadcasting if needed (though shapes should match here)
        //    and returns 1.0 where equal, 0.0 otherwise.
        let mask = equal_op(&input_tensor, &expanded_output)?;
        
        // 5. Calculate grad_input = expanded_grad_output * mask
        let grad_input = mul_op(&expanded_grad_output, &mask)?;

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Calculates the output shape after reduction and returns the final shape 
/// and a boolean mask indicating which axes were reduced.
fn calculate_reduction_shape_and_mask(
    input_shape: &[usize],
    axes: Option<&[usize]>,
    keep_dims: bool,
) -> Result<(Vec<usize>, Vec<bool>), NeuraRustError> {
    let rank = input_shape.len();
    let mut reduced_axes_mask = vec![false; rank];
    let _axes_to_reduce = match axes {
        Some(ax) => {
            let mut processed_axes = Vec::with_capacity(ax.len());
            for &axis in ax {
                if axis >= rank {
                    return Err(NeuraRustError::InvalidAxis { axis, rank });
                }
                if reduced_axes_mask[axis] {
                    // Avoid reducing the same axis twice
                    continue;
                }
                reduced_axes_mask[axis] = true;
                processed_axes.push(axis);
            }
            processed_axes
        }
        None => { // Reduce all axes
            reduced_axes_mask.fill(true);
            (0..rank).collect()
        }
    };

    let output_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter_map(|(i, &dim)| {
            if reduced_axes_mask[i] {
                if keep_dims { Some(1) } else { None }
            } else {
                Some(dim)
            }
        })
        .collect();

    Ok((output_shape, reduced_axes_mask))
}

/// Kernel for max reduction on CPU for f32.
fn max_cpu_f32_kernel(
    input_data: &[f32],
    input_shape: &[usize],
    input_strides: &[usize],
    input_offset: usize,
    output_shape: &[usize],
    reduced_axes_mask: &[bool], // Mask indicating which axes are reduced
) -> Result<Vec<f32>, NeuraRustError> {
    let output_numel = output_shape.iter().product();
    let mut output_data = vec![f32::min_value(); output_numel];
    let input_rank = input_shape.len();
    let output_rank = output_shape.len(); // Can be different if keep_dims=false

    let mut input_indices = vec![0; input_rank];

    // Iterate through all elements of the input tensor
    let input_numel = input_shape.iter().product();
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
        let mut output_dim_idx = output_rank; // Iterate dims from right-to-left
        for input_dim_idx in (0..input_rank).rev() {
            if !reduced_axes_mask[input_dim_idx] { // If this axis is NOT reduced
                if output_dim_idx == 0 { 
                     // Should not happen if logic is correct
                     return Err(NeuraRustError::InternalError("Output dimension index underflow".to_string()));
                }
                output_dim_idx -= 1; // Move to the next output dimension
                output_linear_index += input_indices[input_dim_idx] * current_output_stride;
                current_output_stride *= output_shape[output_dim_idx];
            } // else: skip reduced axes for output index calculation
        }

        // Calculate physical offset in input buffer
        let input_physical_offset = input_offset + input_indices.iter().zip(input_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        
        if input_physical_offset >= input_data.len() {
            return Err(NeuraRustError::InternalError("Input buffer access out of bounds in max_kernel".to_string()));
        }
        if output_linear_index >= output_numel {
             return Err(NeuraRustError::InternalError("Output buffer access out of bounds in max_kernel".to_string()));
        }

        // Update the max value for the corresponding output element
        let current_max = &mut output_data[output_linear_index];
        let input_val = input_data[input_physical_offset];
        if input_val > *current_max {
            *current_max = input_val;
        }
    }

    Ok(output_data)
}

// --- Forward Operation ---
pub(crate) fn max_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let tensor_data_guard = tensor.read_data();
    let input_requires_grad = tensor_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(tensor.data.clone()) } else { None };
    let device = tensor_data_guard.device;
    let dtype = tensor_data_guard.dtype;

    if device != StorageDevice::CPU || dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation("max_op currently only supports F32 CPU tensors.".to_string()));
    }

    // Calculate output shape and reduction mask
    let (output_shape, reduced_axes_mask) = calculate_reduction_shape_and_mask(&tensor_data_guard.shape, axes, keep_dims)?;

    // Extract necessary data before dropping guard
    let input_buffer_arc = tensor_data_guard.buffer().try_get_cpu_f32()?.clone();
    let input_shape_clone = tensor_data_guard.shape.clone();
    let input_strides_clone = tensor_data_guard.strides.clone();
    let input_offset = tensor_data_guard.offset;
    let axes_clone = axes.map(|a| a.to_vec()); // Clone axes if Some

    drop(tensor_data_guard);

    // --- Calculation --- 
    let result_vec = max_cpu_f32_kernel(
        input_buffer_arc.as_slice(), 
        &input_shape_clone, 
        &input_strides_clone, 
        input_offset, 
        &output_shape, 
        &reduced_axes_mask
    )?;
    let result_buffer_arc = Arc::new(result_vec);

    // --- Create Output Tensor --- 
    let output_td = TensorData::new(
        result_buffer_arc.as_ref().clone(), 
        output_shape
    )?;
    let output_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    // --- Autograd Setup --- 
    if input_requires_grad {
        if let Some(input_arc) = input_node_arc {
            println!("*** Autograd setup for max_op triggered! ***"); 
            let grad_fn = MaxBackward {
                input_node: input_arc,
                output_node: output_tensor.data.clone(), // Store the output node Arc
                axes: axes_clone, // Use the cloned axes
                keep_dims,
            };
            // Acquire write lock and set requires_grad & grad_fn
            let mut output_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::InternalError("Failed to lock output tensor for autograd setup".to_string()))?;
            output_write_guard.grad_fn = Some(Arc::new(grad_fn));
            output_write_guard.requires_grad = true;
            println!("MaxBackward grad_fn set for max result."); // Debug print
        } else {
            // This case should not happen if input_requires_grad is true
             return Err(NeuraRustError::InternalError(
                "MaxOp requires grad but input Arc<TensorData> was not available".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// Revenir à l'attribut #[path] pour lier le fichier de test externe
#[cfg(test)]
#[path = "max_test.rs"]
mod tests; 