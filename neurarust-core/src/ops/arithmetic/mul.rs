use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul};
use std::sync::{Arc, RwLockReadGuard};

// --- Backward Operation Structure ---

/// Backward operation context for multiplication.
#[derive(Debug)]
struct MulBackward<T: 'static + Debug + Copy + Send + Sync> {
    // Stocker les clones est sûr pour Send + Sync
    a: Tensor<T>,
    b: Tensor<T>,
    // Garder les shapes pour le backward
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for MulBackward<T>
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
        + Add<Output = T>
        + Sum
        + Mul<Output = T>
        + PartialEq
        + PartialOrd
        + std::iter::Product,
{
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.a.get_node_id(), self.b.get_node_id()]
    }

    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = grad_output * b
        let grad_a_unreduced = mul_op(grad_output, &self.b)?;
        let grad_a = grad_a_unreduced.reduce_to_shape(&self.a_shape)?;

        // grad_b = grad_output * a
        let grad_b_unreduced = mul_op(grad_output, &self.a)?;
        let grad_b = grad_b_unreduced.reduce_to_shape(&self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}

// --- Kernel de Calcul ---

/// Noyau de calcul privé pour la multiplication élément par élément avec broadcasting.
fn mul_kernel<T>(
    a_guard: &RwLockReadGuard<'_, TensorData<T>>,
    b_guard: &RwLockReadGuard<'_, TensorData<T>>,
    a_data_slice: &[T],
    b_data_slice: &[T],
    output_shape: &[usize],
) -> Result<Vec<T>, NeuraRustError>
where
    // Bounds pour get_offset (Debug) et calcul (Mul, Copy)
    T: Mul<Output = T> + Copy + Debug,
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

        // Indice pour A
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

        // Indice pour B
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

        result_data_vec.push(val_a * val_b); // Opération de multiplication
    }

    Ok(result_data_vec)
}

// --- Forward Operation ---

pub fn mul_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Mul<Output = T>
        + Add<Output = T>
        + AddAssign
        + Sum
        + PartialOrd
        + std::iter::Product
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Default
        + Zero
        + One
        + PartialEq,
{
    // --- Autograd Setup --- (Adapté pour cloner)
    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut a_maybe_clone: Option<Tensor<T>> = None;
    let mut b_maybe_clone: Option<Tensor<T>> = None;
    let mut a_shape_maybe: Option<Vec<usize>> = None;
    let mut b_shape_maybe: Option<Vec<usize>> = None;
    if requires_grad {
        a_maybe_clone = Some(a.clone());
        b_maybe_clone = Some(b.clone());
        a_shape_maybe = Some(a.shape());
        b_shape_maybe = Some(b.shape());
    }

    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Cannot multiply tensors on different devices: {:?} and {:?}",
            a_guard.device, b_guard.device
        )));
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Multiplication is currently only supported on CPU, not {:?}",
            device
        )));
    }

    let a_data_arc = a_guard.data.cpu_data()?.clone();
    let b_data_arc = b_guard.data.cpu_data()?.clone();
    let a_data_slice = a_data_arc.as_slice();
    let b_data_slice = b_data_arc.as_slice();

    // --- Shape and Broadcasting ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    let output_shape = broadcast_shapes(a_shape, b_shape).map_err(|_e| {
        NeuraRustError::BroadcastError {
            shape1: a_shape.clone(),
            shape2: b_shape.clone(),
        }
    })?;

    // --- Calculation (Appel au Kernel) ---
    let result_data_vec = mul_kernel(
        &a_guard,
        &b_guard,
        a_data_slice,
        b_data_slice,
        &output_shape,
    )?;

    // Drop read locks
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data_vec, output_shape.clone())?;

    // --- Autograd Linkage ---
    if requires_grad {
        let backward_context = MulBackward {
            a: a_maybe_clone.unwrap(),
            b: b_maybe_clone.unwrap(),
            a_shape: a_shape_maybe.unwrap(),
            b_shape: b_shape_maybe.unwrap(),
        };
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }

    Ok(result_tensor)
}

/// REMOVED: In-place MulAssign

// --- Tests ---
#[cfg(test)]
#[path = "mul_test.rs"]
mod tests;
