// neurarust-core/src/ops/view/mod.rs

pub mod slice;
pub mod transpose;
pub mod permute;
pub mod reshape;
pub mod expand; // Added expand op

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
#[derive(Debug)]
struct ExpandBackward {
    input_node: Arc<RwLock<TensorData>>, // Store original tensor info
    _original_shape: Vec<usize>,
}

impl BackwardOp for ExpandBackward {
    // Backward of expand requires reducing the gradient back to the original shape.
    // This often involves summing along the expanded dimensions.
    fn backward(&self, _grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // TODO: Implement expand backward using a reduction operation (likely sum)
        // Identify expanded dimensions by comparing grad_output.shape() and self.original_shape.
        // Call sum_axes along those dimensions, with keep_dims=false.
        todo!("Implement Expand backward using reduction (sum_axes)");
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Creates a new view of the tensor with singleton dimensions expanded
/// to match the target shape. Does not copy data.
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
