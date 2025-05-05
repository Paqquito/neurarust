//! # Tensor Operations Module (`ops`)
//!
//! This module serves as the central hub for defining and organizing various tensor operations
//! within NeuraRust. Operations are categorized into submodules based on their functionality.
//!
//! ## Structure:
//!
//! - **Submodules:** Operations are grouped logically (e.g., `arithmetic`, `linalg`, `reduction`, `view`).
//! - **`_op` Functions:** Each operation typically has a core function (often named `xxx_op`)
//!   that performs the forward computation and sets up the backward pass for autograd.
//!   These functions are often marked `pub(crate)` as they are primarily intended for internal use,
//!   called by methods defined on the `Tensor` struct itself.
//! - **`Backward` Structs:** Each operation requiring gradient computation has a corresponding
//!   struct (e.g., `AddBackward`, `MatmulBackward`) that implements the
//!   [`BackwardOp`](../autograd/backward_op/trait.BackwardOp.html) trait. This struct stores the necessary
//!   context from the forward pass to compute gradients correctly during backpropagation.
//! - **Traits (`ops::traits`):** May define common traits for operations if needed (currently basic).
//!
//! ## Key Submodules:
//!
//! - [`arithmetic`]: Element-wise arithmetic operations (add, sub, mul, div, etc.).
//! - [`linalg`]: Linear algebra operations (matmul, etc.).
//! - [`nn`]: Operations commonly used in neural networks (activations, etc.).
//! - [`reduction`]: Operations that reduce tensor dimensions (sum, mean, max, etc.).
//! - [`view`]: Operations that create new views of tensors without copying data (reshape, slice, transpose, etc.).
//! - [`dtype`]: Operations related to data type conversion (cast).

// Re-export key traits and types from submodules for easier use.
pub mod traits;

// Import necessary types for helper functions
use crate::tensor::Tensor;
use crate::tensor_data::TensorData; // Import direct
use crate::{DType, StorageDevice}; // Utiliser les re-exports de la racine
use crate::error::NeuraRustError;
use crate::autograd::BackwardOp;
 // Chemin correct
use std::sync::{Arc, RwLock};
// Supprimer l'import des itérateurs pour l'instant car non utilisés par apply_unary_op
// use crate::tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64};

// Declare operation submodules
pub mod activation; // Activation functions (formerly under nn)
pub mod arithmetic;
pub mod comparison;
pub mod linalg;
pub mod loss;       // Loss functions (currently empty)
pub mod math_elem;  // Element-wise math functions (ln, etc.)
pub mod reduction;
pub mod view;

// Re-exports: Make core operation functions easily accessible within the crate
// Using pub(crate) keeps them internal but usable by Tensor methods etc.


// Arithmetic ops are re-exported from ops/arithmetic/mod.rs
// pub(crate) use arithmetic::{add_op, div_op, mul_op, neg_op, sub_op, pow_op};


// math_elem ops are re-exported from ops/math_elem/mod.rs
// pub(crate) use math_elem::{ln_op, exp_op, sqrt_op}; // Assuming exp/sqrt exist later

 // Add min_op later if needed


// -- Removed old re-export that caused visibility errors --
// pub use arithmetic::{add_op, div_op, mul_op, neg_op, sub_op}; // Keep arithmetic ops

// Re-export the main BackwardOp trait for convenience within ops modules?
// Maybe not necessary, full path is clear.
// pub(crate) use crate::autograd::backward_op::BackwardOp;

/// Applies a unary element-wise operation to a tensor.
///
/// Handles DType dispatch (F32, F64), CPU device check, output tensor creation,
/// data iteration (simple contiguous loop), and autograd setup.
///
/// # Arguments
/// * `a`: The input tensor.
/// * `op_f32`: Closure defining the operation for F32: `Fn(f32) -> f32`.
/// * `op_f64`: Closure defining the operation for F64: `Fn(f64) -> f64`.
/// * `backward_builder`: Closure to build the BackwardOp: `FnOnce(Option<Arc<RwLock<TensorData>>>) -> Arc<dyn BackwardOp>`.
/// * `op_name`: Name of the operation for error messages.
///
/// # Returns
/// A `Result` containing the output tensor or a `NeuraRustError`.
///
/// # Note
/// Currently assumes the input tensor `a` is contiguous.
pub(crate) fn apply_unary_op<F32Op, F64Op, B>(
    a: &Tensor,
    op_f32: F32Op,
    op_f64: F64Op,
    backward_builder: B,
    op_name: &str,
) -> Result<Tensor, NeuraRustError>
where
    F32Op: Fn(f32) -> f32,
    F64Op: Fn(f64) -> f64,
    B: FnOnce(Option<Arc<RwLock<TensorData>>>) -> Arc<dyn BackwardOp>,
{
    let a_guard = a.read_data();

    // Device Check
    if a_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: op_name.to_string(),
            expected: StorageDevice::CPU,
            actual: a_guard.device,
        });
    }

    // Contiguous Check (Initial Simplification)
    // TODO: Enhance later to handle non-contiguous inputs, possibly using NdArrayTensorIter
    if !a_guard.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Unary op helper '{}' currently requires contiguous input tensor. Found strides: {:?}", 
            op_name,
            a_guard.strides
        )));
    }

    // Autograd Setup
    let requires_grad = a_guard.requires_grad;
    let a_node_opt = if requires_grad { Some(Arc::clone(&a.data)) } else { None };
    let output_shape = a_guard.shape.clone();
    let numel = a_guard.numel();
    let offset = a_guard.offset;

    // DType Dispatch & Computation
    let output_tensor = match a_guard.dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer().try_get_cpu_f32()?;
            let output_data: Vec<f32> = a_buffer[offset..offset + numel]
                .iter()
                .map(|&val| op_f32(val))
                .collect();
            drop(a_guard);
            Tensor::new(output_data, output_shape)?
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer().try_get_cpu_f64()?;
            let output_data: Vec<f64> = a_buffer[offset..offset + numel]
                .iter()
                .map(|&val| op_f64(val))
                .collect();
            drop(a_guard);
            Tensor::new_f64(output_data, output_shape)?
        }
    };

    // Set Autograd Metadata
    if requires_grad {
        // Pass the optional Arc to the builder
        let grad_fn = backward_builder(a_node_opt);
        let mut output_guard = output_tensor.write_data(); // Use helper
        output_guard.grad_fn = Some(grad_fn);
        output_guard.requires_grad = true;
    }

    Ok(output_tensor)
}
