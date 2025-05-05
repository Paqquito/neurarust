//!
//! # Element-wise Arithmetic Operations
//!
//! This module provides functions for performing basic element-wise arithmetic
//! operations on tensors, such as addition, subtraction, multiplication, division,
//! negation, and exponentiation.
//!
//! These operations typically support broadcasting and automatic differentiation.

// Declare the modules first
pub mod add;
pub mod div;
pub mod mul;
pub mod neg;
pub mod pow;
pub mod sub;

// Re-export specific op functions after declaring modules
pub(crate) use add::add_op;
pub(crate) use div::div_op;
pub(crate) use mul::mul_op;
pub(crate) use sub::sub_op;

// --- Centralized Helper for Binary Ops with Broadcasting --- 

use crate::{
    autograd::BackwardOp,
    device::StorageDevice,
    error::NeuraRustError,
    tensor::{iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64}, utils::broadcast_shapes, Tensor},
    tensor_data::TensorData,
    types::DType,
};
use std::sync::{Arc, RwLock};

/// Applies a binary element-wise operation with broadcasting and autograd support.
///
/// This internal helper centralizes the boilerplate logic for operations like add, mul, etc.
/// It handles device/dtype checks, shape broadcasting, iterator creation, computation,
/// output tensor creation, and autograd graph setup.
///
/// # Arguments
/// * `a`: The first input tensor.
/// * `b`: The second input tensor.
/// * `op_f32`: Closure performing the operation for F32: `|f32, f32| -> f32`.
/// * `op_f64`: Closure performing the operation for F64: `|f64, f64| -> f64`.
/// * `build_backward_op`: Closure to build the specific `BackwardOp` struct for autograd.
///   It receives the necessary context (input nodes, shapes, requires_grad flags).
/// * `op_name`: Static string representing the operation name (for error messages).
///
/// # Returns
/// A `Result` containing the output `Tensor` or a `NeuraRustError`.
pub(crate) fn apply_binary_op_broadcasted<
    OpF32,
    OpF64,
    BuildBackward
>(
    a: &Tensor,
    b: &Tensor,
    op_f32: OpF32,
    op_f64: OpF64,
    build_backward_op: BuildBackward,
    op_name: &'static str,
) -> Result<Tensor, NeuraRustError>
where
    OpF32: Fn(f32, f32) -> f32,
    OpF64: Fn(f64, f64) -> f64,
    BuildBackward: FnOnce(
        Option<Arc<RwLock<TensorData>>>, // a_node_arc
        Option<Arc<RwLock<TensorData>>>, // b_node_arc
        Vec<usize>,                     // a_shape
        Vec<usize>,                     // b_shape
        bool,                           // a_requires_grad
        bool,                           // b_requires_grad
    ) -> Arc<dyn BackwardOp + Send + Sync>,
{
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device and DType Checks ---
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: op_name.to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }
    if a_guard.dtype != b_guard.dtype {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: op_name.to_string(),
            expected: a_guard.dtype,
            actual: b_guard.dtype,
        });
    }
    let dtype = a_guard.dtype;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    // --- Prepare for Autograd --- 
    let requires_grad = a_guard.requires_grad || b_guard.requires_grad;
    let a_node_arc_for_backward = if requires_grad { Some(Arc::clone(&a.data)) } else { None };
    let b_node_arc_for_backward = if requires_grad { Some(Arc::clone(&b.data)) } else { None };
    let a_shape_clone = a_guard.shape.clone();
    let b_shape_clone = b_guard.shape.clone();
    let a_req_grad_clone = a_guard.requires_grad;
    let b_req_grad_clone = b_guard.requires_grad;

    // --- DType Dispatch for Computation using Iterators --- 
    let output_tensor = match dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            
            let iter_a = NdArrayBroadcastingIter::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIter::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            
            let output_data_vec: Vec<f32> = iter_a.zip(iter_b).map(|(va, vb)| op_f32(va, vb)).collect();
            
            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!(
                    "{}: Output vec len {} mismatch with expected numel {}",
                    op_name, output_data_vec.len(), numel
                )));
            }
            
            drop(a_guard); drop(b_guard);
            Tensor::new(output_data_vec, output_shape)?
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;

            let iter_a = NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;

            let output_data_vec: Vec<f64> = iter_a.zip(iter_b).map(|(va, vb)| op_f64(va, vb)).collect();

            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!(
                    "{}: Output vec len {} mismatch with expected numel {}",
                    op_name, output_data_vec.len(), numel
                )));
            }

            drop(a_guard); drop(b_guard);
            Tensor::new_f64(output_data_vec, output_shape)?
        }
    };

    // --- Autograd Setup --- 
    if requires_grad {
         let backward_op_arc = build_backward_op(
             a_node_arc_for_backward,
             b_node_arc_for_backward,
             a_shape_clone,
             b_shape_clone,
             a_req_grad_clone,
             b_req_grad_clone,
         );

        let mut output_guard = output_tensor.write_data();
        output_guard.requires_grad = true;
        output_guard.grad_fn = Some(backward_op_arc);
    }

    Ok(output_tensor)
}
