use crate::autograd::BackwardOp;
 // Non-generic
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use super::utils; // Use super::utils

use std::sync::{Arc, RwLock};
use std::fmt::Debug;
 // For test signatures

/// Backward operation context for the `permute` operation.
///
/// Stores the original input tensor node and the original permutation (`original_axes`)
/// applied during the forward pass. This is needed to compute the inverse permutation
/// for the backward pass.
#[derive(Debug)]
struct PermuteBackward {
    input_node: Arc<RwLock<TensorData>>,
    original_axes: Vec<usize>,
}

impl PermuteBackward {
    /// Computes the inverse permutation of the `original_axes`.
    ///
    /// If the original permutation mapped axis `i` to `original_axes[i]`, the inverse
    /// permutation maps axis `original_axes[i]` back to `i`.
    fn inverse_axes(&self) -> Vec<usize> {
        let mut inverse = vec![0; self.original_axes.len()];
        for (i, &axis) in self.original_axes.iter().enumerate() {
            inverse[axis] = i;
        }
        inverse
    }
}

// --- Backward Operation Implementation ---
impl BackwardOp for PermuteBackward {
    /// Computes the gradient for the permute operation.
    ///
    /// The gradient of a permutation is obtained by permuting the incoming gradient
    /// (`grad_output`) using the *inverse* of the original permutation.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original permute operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with a single element: the gradient
    /// with respect to the original input tensor (which is the inversely permuted `grad_output`).
    /// Returns an error if the permutation operation on the gradient fails.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let inverse_axes = self.inverse_axes();
        
        // Appeler permute_op avec les axes inverses sur le gradient entrant original
        let grad_input = permute_op(grad_output, &inverse_axes)?;

        // Retourner le gradient calculé dans un vecteur
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- Forward Operation ---
/// Permutes the dimensions of the tensor according to the given axes, creating a view.
///
/// This is a crate-internal function, typically called via the `Tensor::permute` method.
/// It rearranges the dimensions of the input tensor based on the `dims` slice,
/// returning a new tensor view without copying data.
///
/// # Arguments
///
/// * `input` - The input tensor.
/// * `dims` - A slice specifying the new order of dimensions. It must be a permutation
///   of `0..rank`, where `rank` is the number of dimensions of the input tensor.
///   `dims[i]` indicates which original dimension should become the new dimension `i`.
///
/// # Returns
///
/// A `Result` containing the permuted `Tensor` view. Returns an error if:
/// *   `dims` is not a valid permutation of `0..rank`.
/// *   Device or autograd operations fail.
///
/// # Example (Conceptual - Use `Tensor::permute` instead)
///
/// ```rust,ignore
/// // Assuming t is a Tensor of shape [2, 3, 4]
/// // use crate::ops::view::permute::permute_op; // Assuming direct access
///
/// // Permute dimensions to [4, 2, 3] (original dims 2, 0, 1)
/// let permuted_view = permute_op(&t, &[2, 0, 1])?;
/// // permuted_view will have shape [4, 2, 3]
/// ```
/// assert_eq!(permuted.get_f32_data().unwrap(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
/// # Ok::<(), NeuraRustError>(())
/// # }
/// // Example ignored as doc-test: illustrative purpose
pub fn permute_op(input: &Tensor, dims: &[usize]) -> Result<Tensor, NeuraRustError> {
    let input_data_guard = input.data.read().map_err(|_| NeuraRustError::LockError {
        lock_type: "read".to_string(),
        reason: "Failed to lock input TensorData for read in permute_op".to_string(),
    })?;

    let rank = input_data_guard.shape.len();
    // Validate permutation dimensions
    utils::validate_permutation(rank, dims)?;

    // Calculate new shape and strides
    let new_shape = utils::permute_shape(&input_data_guard.shape, dims);
    let new_strides = utils::permute_strides(&input_data_guard.strides, dims);
    let offset = input_data_guard.offset;
    let device = input_data_guard.device;
    let buffer_arc = Arc::clone(&input_data_guard.buffer);
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };
    let original_dims_clone = dims.to_vec(); // For backward

    drop(input_data_guard);

    let view_td = TensorData::new_view(buffer_arc, device, offset, new_shape, new_strides)?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    // Autograd setup
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                 lock_type: "write".to_string(),
                 reason: "Failed to lock output TensorData for write (autograd setup in permute_op)".to_string(),
             })?;
             output_data_write_guard.requires_grad = true;
             let backward_op = PermuteBackward {
                 input_node: node_arc,
                 original_axes: original_dims_clone, // Store original permutation
             };
             output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in permute_op".to_string()));
        }
    }

    Ok(output_tensor)
}

// Link the external tests file
#[cfg(test)]
#[path = "permute_test.rs"] mod tests; 