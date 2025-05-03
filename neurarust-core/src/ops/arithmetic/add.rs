// neurarust-core/src/ops/arithmetic/add.rs

use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use crate::tensor_data::TensorData;
use std::sync::RwLockReadGuard;

// --- Kernel de Calcul ---

/// Noyau de calcul privé pour l'addition élément par élément avec broadcasting.
/// Gère l'itération, le calcul d'indices/offsets et l'opération arithmétique.
fn add_kernel<T>(
    a_guard: &RwLockReadGuard<'_, TensorData<T>>,
    b_guard: &RwLockReadGuard<'_, TensorData<T>>,
    a_data_slice: &[T],
    b_data_slice: &[T],
    output_shape: &[usize],
) -> Result<Vec<T>, NeuraRustError>
where
    T: Add<Output = T>
        + Copy
        + Debug
        + Default
        + Send
        + Sync
        + Zero
        + One
        + AddAssign
        + PartialEq
        + PartialOrd
        + Sum
        + 'static,
{
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    let result_strides = calculate_strides(output_shape);

    let rank_diff_a = output_shape.len().saturating_sub(a_guard.shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_guard.shape.len());

    let mut input_a_coords = vec![0; a_guard.shape.len()];
    let mut input_b_coords = vec![0; b_guard.shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, output_shape);

        for dim_idx in 0..a_guard.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_a = a_guard.get_offset(&input_a_coords);
        let val_a = a_data_slice[offset_a];

        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_b = b_guard.get_offset(&input_b_coords);
        let val_b = b_data_slice[offset_b];

        result_data_vec.push(val_a + val_b);
    }

    Ok(result_data_vec)
}

// --- Forward Operation ---

/// Performs element-wise addition for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// If either input tensor requires gradients, the output tensor will also require gradients
/// and have its `grad_fn` set to an `AddBackward` operation node.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
/// This operation creates a new Tensor with copied data on the same device.
pub fn add_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Add<Output = T>
        + AddAssign
        + Sum
        + PartialOrd
        + Send
        + Sync
        + std::iter::Product
        + Debug
        + Copy
        + Clone
        + Default
        + PartialEq
        + Zero
        + One
        + 'static,
{
    // --- Autograd Setup ---
    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut a_id_maybe: Option<NodeId<T>> = None;
    let mut a_shape_maybe: Option<Vec<usize>> = None;
    let mut b_id_maybe: Option<NodeId<T>> = None;
    let mut b_shape_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        a_id_maybe = Some(a.get_node_id());
        a_shape_maybe = Some(a.shape());
        b_id_maybe = Some(b.get_node_id());
        b_shape_maybe = Some(b.shape());
    }

    // Acquire read locks for inputs
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Cannot add tensors on different devices: {:?} and {:?}",
            a_guard.device, b_guard.device
        )));
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Addition is currently only supported on CPU, not {:?}",
            device
        )));
    }

    // --- Get CPU Data Buffers ---
    let a_data_arc = a_guard.data.cpu_data()?.clone();
    let b_data_arc = b_guard.data.cpu_data()?.clone();
    let a_data_slice = a_data_arc.as_slice();
    let b_data_slice = b_data_arc.as_slice();

    // --- Shape and Broadcasting ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;

    let output_shape =
        broadcast_shapes(a_shape, b_shape).map_err(|_e| {
            NeuraRustError::BroadcastError {
                shape1: a_shape.clone(),
                shape2: b_shape.clone(),
            }
        })?;

    // --- Calculation (Appel au Kernel) ---
    let result_data_vec = add_kernel(
        &a_guard,
        &b_guard,
        a_data_slice,
        b_data_slice,
        &output_shape,
    )?;

    // Drop read locks explicitly (although they drop implicitly at end of scope)
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data_vec, output_shape.clone())?;

    // --- Autograd Linkage (The General Pattern) ---
    if requires_grad {
        let backward_context = AddBackward {
            a_id: a_id_maybe.unwrap(),
            a_shape: a_shape_maybe.unwrap(),
            b_id: b_id_maybe.unwrap(),
            b_shape: b_shape_maybe.unwrap(),
        };
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }

    Ok(result_tensor)
}

/// REMOVED: In-place AddAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.
// impl<'a, T> AddAssign<&'a Tensor<T>> for Tensor<T>
// where
//     T: AddAssign + Copy + Clone,
// {
//     fn add_assign(&mut self, other: &'a Tensor<T>) { ... }
// }

// --- Backward Operation ---

/// Backward operation context for the element-wise addition operation.
/// Stores the NodeIds of the input tensors and their shapes to handle broadcasting.
#[derive(Debug)]
struct AddBackward<T: 'static + Debug + Copy + Send + Sync> {
    // NodeId for input tensor 'a'
    a_id: NodeId<T>,
    // Original shape of input tensor 'a' before broadcasting
    a_shape: Vec<usize>,
    // NodeId for input tensor 'b'
    b_id: NodeId<T>,
    // Original shape of input tensor 'b' before broadcasting
    b_shape: Vec<usize>,
}

// Mark AddBackward as Send + Sync.
// This is unsafe because the struct contains raw pointers (NodeId).
// However, we guarantee that these pointers are valid and accesses are synchronized
// through the RwLocks within the TensorData they point to, managed by the broader
// autograd system (Tensor::backward ensures tensors are kept alive).
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for AddBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for AddBackward<T> {}

// Implement `BackwardOp<T>` for `AddBackward<T>`
impl<T> BackwardOp<T> for AddBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Default
        + Clone
        + Zero
        + One
        + AddAssign
        + Sum
        + Add<Output = T>
        + PartialEq
        + PartialOrd
        + std::iter::Product,
{
    /// Returns the NodeIds of the input tensors involved in the addition.
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.a_id, self.b_id]
    }

    /// Computes the gradients for the input tensors (`a` and `b`) of the addition.
    /// Handles broadcasting by summing the gradient along the broadcasted dimensions.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // For z = a + b, grad_a = grad_output * 1, grad_b = grad_output * 1
        // However, we need to reduce the gradient shape if broadcasting occurred.
        // grad_a shape must match self.a_shape
        // grad_b shape must match self.b_shape
        let grad_a = grad_output.reduce_to_shape(&self.a_shape)?;
        let grad_b = grad_output.reduce_to_shape(&self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}

// --- Tests ---
#[cfg(test)]
#[path = "add_test.rs"]
mod tests;
