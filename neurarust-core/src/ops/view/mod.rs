// neurarust-core/src/ops/view/mod.rs

//! # Tensor View Operations
//!
//! This module provides operations that create new `Tensor` views without copying the
//! underlying data. These operations manipulate the tensor's metadata (shape, strides, offset)
//! to present a different perspective on the same data buffer.
//!
//! View operations are crucial for efficiency, especially in deep learning, as they avoid
//! unnecessary memory allocations and copies.
//!
//! ## Key Operations:
//! - **[`slice_op`](slice/fn.slice_op.html)**: Extracts a sub-tensor (slice).
//! - **[`transpose_op`](transpose/fn.transpose_op.html)**: Swaps two dimensions.
//! - **[`permute_op`](permute/fn.permute_op.html)**: Rearranges dimensions according to a given permutation.
//! - **[`reshape_op`](reshape/fn.reshape_op.html)**: Changes the shape of the tensor while preserving the number of elements.
//! - **[`expand_op`](fn.expand_op.html)**: Broadcasts singleton dimensions (size 1) to a larger size.
//!
//! ## Autograd Integration:
//! Each view operation (`_op` function) typically has a corresponding `Backward` struct
//! (e.g., `SliceBackward`, `TransposeBackward`) that implements the [`BackwardOp`](../../autograd/trait.BackwardOp.html)
//! trait. These structures store the necessary context (like original shapes or axes)
//! to correctly propagate gradients back through the view operation during the backward pass.
//!
//! For example, the backward pass of a `reshape` operation might involve reshaping the incoming
//! gradient back to the input tensor's original shape. The backward of `expand` requires summing
//! the gradient along the dimensions that were expanded.
//!
//! ## Usage:
//! These `_op` functions are usually called internally by methods on the `Tensor` struct
//! (e.g., [`tensor.slice()`](../../tensor/struct.Tensor.html#method.slice), [`tensor.transpose()`](../../tensor/struct.Tensor.html#method.transpose), etc.),
//! which provide a more user-friendly interface.

pub mod slice;
pub mod transpose;
pub mod permute;
pub mod reshape;
pub mod expand;
pub mod contiguous;

pub mod utils; // Declare the utils module

// Re-exports for easier access (optional)
// Use pub(crate) for ops not meant for direct user call yet
pub(crate) use slice::slice_op;
pub(crate) use transpose::transpose_op;
pub(crate) use permute::permute_op;
pub(crate) use reshape::reshape_op;
// pub use expand::expand_op; // Keep expand_op public for now

// Conserver la définition de SliceArg ici car elle est utilisée par slice_op
pub use slice::SliceArg;

// Le reste du code (implémentations des ops, backward structs, tests)
// a été déplacé dans les modules slice.rs, transpose.rs, etc.

use crate::autograd::BackwardOp;
use crate::autograd::graph::NodeId;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

// --- Expand Operation ---

/// Backward operation for the expand operation.
///
/// Stores the original input tensor node to route gradients back to it.
/// The backward pass involves reducing the incoming gradient back to the
/// original input shape by summing along the dimensions that were expanded.
#[derive(Debug)]
struct ExpandBackward {
    input_node: Arc<RwLock<TensorData>>, // Store original tensor info
    _original_shape: Vec<usize>, // TODO: Use this shape in backward pass
}

impl BackwardOp for ExpandBackward {
    /// Computes the gradient for the expand operation.
    ///
    /// This requires summing the incoming gradient (`grad_output`) along the dimensions
    /// that were expanded during the forward pass to match the original input shape.
    ///
    /// **Note:** This implementation is currently incomplete (`todo!`).
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original expand operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with the gradient for the original input.
    /// Should eventually return the summed gradient. Currently panics.
    fn backward(&self, _grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // TODO: Implement expand backward using a reduction operation (likely sum)
        // Identify expanded dimensions by comparing grad_output.shape() and self._original_shape.
        // Call sum_op along those dimensions, with keep_dims=false.
        // Need to handle the axes correctly for sum_op.
        todo!("Implement Expand backward using reduction (sum_op)");
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Creates a new view of the tensor with singleton dimensions expanded
/// to match the target shape, similar to broadcasting rules.
///
/// This operation does not copy the underlying data. It works by manipulating
/// the tensor's strides: dimensions that are expanded from size 1 have their
/// stride set to 0.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `target_shape` - The desired shape after expansion. Must be compatible with the
///   input tensor's shape according to broadcasting rules:
///   - The target shape's rank must be greater than or equal to the input's rank.
///   - When iterating dimensions from right to left:
///     - If the input dimension size is equal to the target dimension size, it's compatible.
///     - If the input dimension size is 1, it can be expanded to match the target size (stride becomes 0).
///     - If the input dimension size is different from the target dimension size and not 1, it's an error.
///     - If the target shape has more dimensions than the input, the new leading dimensions
///       are treated as being expanded from size 1 (stride becomes 0).
///
/// # Returns
///
/// A `Result` containing the expanded `Tensor` view. Returns an error if:
/// *   The target shape is incompatible with the input shape.
/// *   Device or autograd operations fail.
///
/// # Example
///
/// ```rust,ignore
/// // Assuming t is a Tensor of shape [3, 1] with data [[1], [2], [3]]
/// // use crate::ops::view::expand_op; // Assuming direct access
///
/// let expanded = expand_op(&t, vec![3, 4])?;
/// // expanded will have shape [3, 4]
/// // Data effectively looks like:
/// // [[1, 1, 1, 1],
/// //  [2, 2, 2, 2],
/// //  [3, 3, 3, 3]]
/// // But the underlying data buffer is unchanged.
///
/// // Assuming s is a scalar Tensor (shape []) with value 5
/// let expanded_scalar = expand_op(&s, vec![2, 2])?;
/// // expanded_scalar will have shape [2, 2] with all elements 5.
/// ```
/// assert_eq!(expanded.shape(), &[2, 3, 4]);
/// assert_eq!(expanded.get_f32_data().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
/// # Ok::<(), NeuraRustError>(())
/// # }
/// // Example ignored as doc-test: illustrative purpose
/// ```rust, ignore
/// use neurarust_core::{tensor::Tensor, error::NeuraRustError};
/// use neurarust_core::ops::view::expand_op;
///
pub fn expand_op(tensor: &Tensor, target_shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    let input_guard = tensor.read_data();
    let input_shape = &input_guard.shape;
    let input_strides = &input_guard.strides;
    let input_rank = input_shape.len();
    let target_rank = target_shape.len();

    if target_rank < input_rank {
        return Err(NeuraRustError::ShapeMismatch {
            expected: format!("Rank >= {}", input_rank),
            actual: format!("Rank {}", target_rank),
            operation: "expand_op (target rank must be >= input rank)".to_string(),
        });
    }

    // Calculate new strides and check shape compatibility
    let mut new_strides = vec![0; target_rank];
    for i in (0..target_rank).rev() {
        let input_dim_idx = (i as isize) - (target_rank as isize - input_rank as isize);
        if input_dim_idx >= 0 {
            // Corresponding dimension exists in input
            let input_dim_idx = input_dim_idx as usize;
            let current_input_dim = input_shape[input_dim_idx];
            let current_target_dim = target_shape[i];

            if current_input_dim == current_target_dim {
                new_strides[i] = input_strides[input_dim_idx];
            } else if current_input_dim == 1 {
                // Expand singleton dimension
                if current_target_dim == 0 { // Cannot expand to zero
                     return Err(NeuraRustError::ShapeMismatch { 
                        expected: format!("non-zero target dim"), // expecting non-zero target dim 
                        actual: format!("0"), 
                        operation: "expand_op (cannot expand to size 0)".to_string()
                    });
                }
                new_strides[i] = 0; // Stride is 0 for expanded dim
            } else {
                // Mismatched non-singleton dimension
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("{}", current_input_dim),
                    actual: format!("{}", current_target_dim),
                    operation: format!("expand_op (dimension mismatch at index {})", i),
                });
            }
        } else {
            // This is a new dimension added at the front
            if target_shape[i] == 0 { // Cannot expand to zero
                return Err(NeuraRustError::ShapeMismatch { 
                    expected: format!("non-zero target dim"), // expecting non-zero target dim 
                    actual: format!("0"), 
                    operation: "expand_op (cannot expand new dim to size 0)".to_string()
                });
            }
            new_strides[i] = 0; // Stride is 0 for new dim
        }
    }

    // Clone necessary data before dropping guard
    let buffer = input_guard.buffer.clone();
    let dtype = input_guard.dtype;
    let device = input_guard.device;
    let offset = input_guard.offset;
    let requires_grad = input_guard.requires_grad;
    let input_node_arc = if requires_grad { Some(tensor.data.clone()) } else { None };
    let original_shape_clone = input_guard.shape.clone(); // For backward

    drop(input_guard);

    // Create new TensorData
    let output_td = TensorData {
        buffer,
        dtype,
        device,
        shape: target_shape.clone(),
        strides: new_strides,
        offset,
        requires_grad: requires_grad, // Inherit requires_grad initially
        grad: None,          // New tensor view starts with no grad
        grad_fn: None,       // Will be set below if needed
    };

    let output_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    // Setup autograd
    if requires_grad {
        if let Some(node_arc) = input_node_arc {
            let grad_fn = ExpandBackward {
                input_node: node_arc,
                _original_shape: original_shape_clone,
            };
            output_tensor.write_data().grad_fn = Some(Arc::new(grad_fn));
            // output_tensor requires_grad is already true via inheritance
        } else {
             return Err(NeuraRustError::InternalError(
                "Expand op requires grad but input Arc Node unavailable".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}
