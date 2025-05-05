use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::ops::view::transpose_op; // Needed for backward
use crate::types::DType;

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// --- MatmulBackward Definition ---

/// Backward operation context for `matmul_op`.
#[derive(Debug)]
struct MatmulBackward {
    // Store Arc<RwLock<TensorData>> for Send + Sync safety
    a_node: Arc<RwLock<TensorData>>,
    b_node: Arc<RwLock<TensorData>>,
    a_requires_grad: bool,
    b_requires_grad: bool,
}

// --- BackwardOp Implementation for MatmulBackward ---

impl BackwardOp for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut input_grads: Vec<Tensor> = Vec::with_capacity(2);

        // Create Tensors by cloning the Arcs stored in self
        let a_tensor = Tensor { data: self.a_node.clone() };
        let b_tensor = Tensor { data: self.b_node.clone() };

        // Calculate grad_a = grad_output @ b.T
        if self.a_requires_grad {
            // Ensure b_transposed is contiguous before matmul
            let b_transposed = transpose_op(&b_tensor, 0, 1)?.contiguous()?;
            let grad_a = matmul_op(grad_output, &b_transposed)?;
            input_grads.push(grad_a);
        } // Else: No grad needed for a

        // Calculate grad_b = a.T @ grad_output
        if self.b_requires_grad {
             // Ensure a_transposed is contiguous before matmul
            let a_transposed = transpose_op(&a_tensor, 0, 1)?.contiguous()?;
            let grad_b = matmul_op(&a_transposed, grad_output)?;
            // Ensure grads are pushed in the correct order relative to `inputs()`
            // If a_grad wasn't pushed, grad_b is the first element.
            // If a_grad was pushed, grad_b is the second.
            // Let's assume the graph expects grads only for inputs that required them.
            
            input_grads.push(grad_b);
        } // Else: No grad needed for b

        Ok(input_grads)
    }

    fn inputs(&self) -> Vec<NodeId> {
        let mut ids = Vec::new();
        // Return the NodeId (raw pointer) for inputs that required grad
        // Order should match the original op's input order (a, then b)
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

// --- matmul_op Implementation (Public API with Autograd) ---

/// Performs 2D matrix multiplication (A @ B).
/// Currently only supports F32 tensors on CPU.
pub fn matmul_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_requires_grad = a.requires_grad();
    let b_requires_grad = b.requires_grad();
    let requires_grad = a_requires_grad || b_requires_grad;

    // Clone Arcs for TensorData if grad is needed
    let a_node_arc = if requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if requires_grad { Some(b.data.clone()) } else { None };

    // Perform the actual matrix multiplication
    let output_tensor = matmul_internal(a, b)?;

    // --- Autograd Integration ---
    if requires_grad {
        // We need both Arcs if any grad is required by MatmulBackward's current structure
        if let (Some(a_arc), Some(b_arc)) = (a_node_arc, b_node_arc) {
             let grad_fn = MatmulBackward {
                a_node: a_arc,
                b_node: b_arc,
                a_requires_grad,
                b_requires_grad,
            };
            let mut output_write_guard = output_tensor.write_data();
            output_write_guard.grad_fn = Some(Arc::new(grad_fn));
            output_write_guard.requires_grad = true;
        } else {
            // This case should not happen if requires_grad is true and we correctly cloned
             return Err(NeuraRustError::InternalError(
                "Matmul requires grad but Arc<TensorData> unavailable".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// --- matmul_kernel (Private Calculation Core) ---

/// Private kernel for matrix multiplication calculation. CPU implementation, generic over T.
fn matmul_kernel<T>(
    m: usize,
    k: usize,
    n: usize,
    a_buffer: &[T],
    a_strides: &[usize],
    a_offset: usize,
    b_buffer: &[T],
    b_strides: &[usize],
    b_offset: usize,
) -> Result<Vec<T>, NeuraRustError>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + Debug
{
    let mut output_data = vec![T::default(); m * n]; // Use T::default()

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default(); // Use T::default()
            for k_idx in 0..k {
                let a_physical_idx = a_offset + i * a_strides[0] + k_idx * a_strides[1];
                let b_physical_idx = b_offset + k_idx * b_strides[0] + j * b_strides[1];

                if a_physical_idx >= a_buffer.len() || b_physical_idx >= b_buffer.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Matmul kernel index out of bounds ({},{},{}) -> A[{}], B[{}]", i, j, k_idx, a_physical_idx, b_physical_idx)
                    ));
                }
                // Use AddAssign and Mul traits for T
                sum += a_buffer[a_physical_idx] * b_buffer[b_physical_idx];
            }
            let output_physical_idx = i * n + j;
            if output_physical_idx >= output_data.len() {
                 return Err(NeuraRustError::InternalError(
                    format!("Matmul kernel output index out of bounds ({},{}) -> Out[{}]", i, j, output_physical_idx)
                ));
            }
            output_data[output_physical_idx] = sum;
        }
    }
    Ok(output_data)
}

// --- matmul_internal Implementation (Core Logic, No Autograd Setup) ---

/// Internal implementation of 2D matrix multiplication without autograd setup.
/// Handles F32 and F64 CPU tensors.
fn matmul_internal(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
         return Err(NeuraRustError::DeviceMismatch {
            operation: "matmul_internal".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device }
         });
    }

    // --- DType Check --- 
    if a_guard.dtype != b_guard.dtype {
         return Err(NeuraRustError::DataTypeMismatch {
            operation: "matmul_internal".to_string(),
            expected: a_guard.dtype,
            actual: b_guard.dtype
        });
    }
    let dtype = a_guard.dtype; // Store dtype for dispatch

    // --- Rank Check ---
    if a_guard.shape.len() != 2 || b_guard.shape.len() != 2 {
        return Err(NeuraRustError::RankMismatch {
            expected: 2, 
            actual: a_guard.shape.len().max(b_guard.shape.len())
        });
    }

    // --- Shape Compatibility Check ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    let m = a_shape[0];
    let k1 = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];
    if k1 != k2 {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "matmul (inner dim)".to_string(),
            expected: format!("Inner dimension matching {}, got ({}, {})", k1, k1, k2),
            actual: format!("Shapes {:?} @ {:?}", a_shape, b_shape)
        });
    }

    // Clone strides and offset BEFORE dropping the guards
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;

    // --- Dispatch based on DType for Kernel Call & Output Creation ---
    let output_shape = vec![m, n];
    let output_tensor = match dtype {
        DType::F32 => {
            let a_buffer_arc = a_guard.buffer().try_get_cpu_f32()?.clone();
            let b_buffer_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
            let a_buffer_slice = a_buffer_arc.as_slice();
            let b_buffer_slice = b_buffer_arc.as_slice();
            drop(a_guard); drop(b_guard); // Drop guards

            let output_data = matmul_kernel(
                m, k1, n,
                a_buffer_slice, &a_strides, a_offset,
                b_buffer_slice, &b_strides, b_offset,
            )?;
            Tensor::new(output_data, output_shape)?
        }
        DType::F64 => {
            let a_buffer_arc = a_guard.buffer().try_get_cpu_f64()?.clone();
            let b_buffer_arc = b_guard.buffer().try_get_cpu_f64()?.clone();
            let a_buffer_slice = a_buffer_arc.as_slice();
            let b_buffer_slice = b_buffer_arc.as_slice();
            drop(a_guard); drop(b_guard); // Drop guards

            let output_data: Vec<f64> = matmul_kernel(
                m, k1, n,
                a_buffer_slice, &a_strides, a_offset,
                b_buffer_slice, &b_strides, b_offset,
            )?;
            Tensor::new_f64(output_data, output_shape)?
        }
    };

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
#[path = "matmul_test.rs"]
mod tests; 