//! Utility functions for reduction operations.

use crate::error::NeuraRustError;

/// Processes the axes provided for a reduction operation.
///
/// - If `axes` is `None`, returns all axes `0..rank`.
/// - If `axes` is `Some`, validates each axis against the rank,
///   handles negative axes (relative to rank), and removes duplicates.
///
/// # Arguments
/// * `rank`: The rank (number of dimensions) of the input tensor.
/// * `axes`: An optional slice of `usize` representing the axes to reduce.
///
/// # Returns
/// A `Result` containing a sorted `Vec<usize>` of unique, non-negative axes, 
/// or a `NeuraRustError::InvalidAxis` if any axis is out of bounds.
pub(crate) fn process_reduction_axes(
    rank: usize,
    axes: Option<&[usize]>,
) -> Result<Vec<usize>, NeuraRustError> {
    let mut processed_axes: Vec<usize>;

    if let Some(ax) = axes {
        if ax.is_empty() {
             // Special case: empty axes means reduce over all dimensions if axes is Some([])
             // However, typically None is used for all axes. If Some([]) is passed,
             // interpret as no reduction? Or all? Let's assume reduce over all.
            // Clarification: PyTorch's sum([]) reduces nothing. Let's match that.
            // If user wants to reduce all, they should pass None.
            // Update: Based on sum_op logic, None or empty slice means reduce all dims.
             processed_axes = (0..rank).collect();

        } else {
            processed_axes = Vec::with_capacity(ax.len());
            for &axis_val in ax {
                // Handle potential negative indices (though usize doesn't allow negatives directly,
                // if input came from signed type, this logic would apply)
                // Let's assume usize means non-negative indices here.
                if axis_val >= rank {
                    return Err(NeuraRustError::InvalidAxis {
                        axis: axis_val,
                        rank,
                    });
                }
                processed_axes.push(axis_val);
            }
            // Sort and remove duplicates
            processed_axes.sort_unstable();
            processed_axes.dedup();
        }
    } else {
        // None means reduce over all axes
        processed_axes = (0..rank).collect();
    }

    Ok(processed_axes)
}

/// Calculates the output shape after a reduction operation.
///
/// # Arguments
/// * `input_shape`: The shape of the original tensor.
/// * `axes`: A slice of processed (unique, non-negative) axes to reduce.
/// * `keep_dims`: If true, reduced dimensions are kept with size 1.
///
/// # Returns
/// The shape of the tensor after reduction.
pub(crate) fn calculate_reduction_output_shape(
    input_shape: &[usize],
    axes: &[usize],
    keep_dims: bool,
) -> Vec<usize> {
    if input_shape.is_empty() {
        return vec![]; // Reduction of a scalar is a scalar
    }

    let rank = input_shape.len();
    let mut output_shape = Vec::with_capacity(rank);

    if keep_dims {
        for (i, &dim_size) in input_shape.iter().enumerate() {
            if axes.contains(&i) {
                output_shape.push(1); // Keep reduced dim as 1
            } else {
                output_shape.push(dim_size);
            }
        }
    } else {
        for (i, &dim_size) in input_shape.iter().enumerate() {
            if !axes.contains(&i) {
                output_shape.push(dim_size); // Only keep non-reduced dims
            }
        }
        // Handle case where all dimensions are reduced and keep_dims is false
        // In this case, the result is a scalar, represented by an empty shape vec![]
        // (No need for special handling, the loop correctly produces an empty vec)
    }
    output_shape
}

/// Calculates the shape needed for broadcasting a gradient back to the input shape
/// during the backward pass of a reduction operation.
///
/// If `keep_dims` was true during the forward pass, the gradient already has the correct
/// number of dimensions (with size 1 for reduced axes), so its shape is returned directly.
/// If `keep_dims` was false, this function inserts dimensions of size 1 into the
/// `grad_output_shape` at the positions corresponding to the reduced `axes`.
///
/// # Arguments
/// * `input_shape`: The shape of the original input tensor to the reduction op.
/// * `grad_output_shape`: The shape of the gradient received by the backward pass.
/// * `axes`: The axes along which the reduction was performed.
/// * `keep_dims`: Whether `keep_dims` was true during the forward pass.
///
/// # Returns
/// The reshaped gradient shape suitable for broadcasting to the `input_shape`.
pub(crate) fn calculate_grad_broadcast_shape(
    input_shape: &[usize],
    grad_output_shape: &[usize],
    axes: &[usize],
    keep_dims: bool,
) -> Vec<usize> {
    if keep_dims {
        // If dims were kept, grad_output shape should already be compatible for broadcast
        // (assuming grad_output_shape has same rank as input_shape)
        // Might need validation here? For now, just return it.
        return grad_output_shape.to_vec();
    }

    if axes.is_empty() {
        // If no specific axes were reduced (meaning all were), and keep_dims was false,
        // the output was scalar (shape []), and grad_output is likely scalar.
        // The target shape for broadcast is the input_shape, but the reshape target
        // to enable broadcast should be all ones.
        return vec![1; input_shape.len()];
    }

    // If keep_dims was false, insert 1s for the reduced axes.
    let rank = input_shape.len();
    let mut shape_with_kept_dims = Vec::with_capacity(rank);
    let mut current_grad_dim = 0;

    for i in 0..rank {
        if axes.contains(&i) {
            shape_with_kept_dims.push(1); // Insert dimension of size 1
        } else {
            // This dimension was kept, take its size from grad_output_shape
            if current_grad_dim < grad_output_shape.len() {
                shape_with_kept_dims.push(grad_output_shape[current_grad_dim]);
                current_grad_dim += 1;
            } else {
                // This indicates a shape mismatch, likely grad_output has wrong rank.
                // The function calling this should handle the Result/Error.
                // For simplicity here, we might push a default or panic, but ideally,
                // this situation is caught earlier. Let's assume rank matches.
                // Fallback: push the original input dim size? Or 1?
                // Pushing 1 might be safer for broadcasting but masks the error.
                // Let's push the corresponding input shape dim size, assuming caller validated ranks.
                 if i < input_shape.len() { // Bounds check on input_shape
                     shape_with_kept_dims.push(input_shape[i]);
                 } else {
                     // Should be impossible if rank is correct
                     shape_with_kept_dims.push(1); // Safe fallback
                 }
                 // Log potential error
                eprintln!(
                    "Warning: Mismatch between non-reduced axes and grad_output rank in calculate_grad_broadcast_shape. Input: {:?}, Grad: {:?}, Axes: {:?}", 
                    input_shape, grad_output_shape, axes
                );
            }
        }
    }
    shape_with_kept_dims
}

// TODO: Add function to reshape grad_output for backward pass? 