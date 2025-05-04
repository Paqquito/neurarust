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

/// Private kernel for matrix multiplication calculation. F32 CPU implementation.
fn matmul_kernel(
    m: usize,
    k: usize, // Inner dimension (k1 == k2)
    n: usize,
    a_buffer: &[f32],
    a_strides: &[usize],
    a_offset: usize,
    b_buffer: &[f32],
    b_strides: &[usize],
    b_offset: usize,
) -> Result<Vec<f32>, NeuraRustError> {
    let mut output_data = vec![0.0f32; m * n];
    // Output is contiguous, calculate its strides implicitly or pass them?
    // For simplicity, calculate index directly: output_physical_idx = i * n + j;

    for i in 0..m { // Row of output
        for j in 0..n { // Col of output
            let mut sum = 0.0f32;
            for k_idx in 0..k { // Inner dimension
                // Calculate physical index for A[i, k_idx]
                let a_physical_idx = a_offset + i * a_strides[0] + k_idx * a_strides[1];

                // Calculate physical index for B[k_idx, j]
                let b_physical_idx = b_offset + k_idx * b_strides[0] + j * b_strides[1];

                // Bounds check (optional but safer)
                if a_physical_idx >= a_buffer.len() || b_physical_idx >= b_buffer.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Matmul kernel index out of bounds ({},{},{}) -> A[{}], B[{}]", i, j, k_idx, a_physical_idx, b_physical_idx)
                    ));
                }

                sum += a_buffer[a_physical_idx] * b_buffer[b_physical_idx];
            }
            // Calculate physical index for Output[i, j] (Contiguous)
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

/// Internal implementation of 2D matrix multiplication without autograd setup. F32 CPU.
fn matmul_internal(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device and DType Checks ---
    // ... (Assume CPU F32 for now)

    // --- Rank Check ---
    if a_guard.shape.len() != 2 || b_guard.shape.len() != 2 {
        // Correctly return RankMismatch error
        return Err(NeuraRustError::RankMismatch {
            expected: 2, 
            actual: a_guard.shape.len().max(b_guard.shape.len()), // Report the max rank found
        });
    }

    // --- Shape Compatibility Check ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;

    // --- Inner Dimension Check ---
    let m = a_shape[0];
    let k1 = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];
    if k1 != k2 {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "matmul (inner dim)".to_string(),
            // Format shapes into Strings
            expected: format!("[{}, {}]", m, k1), // Or some way to show inner dim match
            actual: format!("[{}, {}]", k2, n),
        });
    }

    // --- Device Check ---
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU
        || a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32
    {
        // Provide more specific error?
        return Err(NeuraRustError::UnsupportedOperation(
            "Matmul currently only supports F32 tensors on CPU".to_string(),
        ));
    }

    // --- Extract data for kernel ---
    let a_buffer_arc = a_guard.buffer().try_get_cpu_f32()?.clone();
    let b_buffer_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    let a_buffer_slice = a_buffer_arc.as_slice();
    let b_buffer_slice = b_buffer_arc.as_slice();

    // Clone strides and offset BEFORE dropping the guards
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;

    // Release guards before calling kernel
    drop(a_guard);
    drop(b_guard);

    // --- Call Kernel ---
    let output_shape = vec![m, n];
    let output_data = matmul_kernel(
        m,
        k1, // k1 == k2
        n,
        a_buffer_slice,
        &a_strides, // Pass as slice
        a_offset,
        b_buffer_slice,
        &b_strides, // Pass as slice
        b_offset,
    )?;

    // --- Create Output Tensor ---
    Tensor::new(output_data, output_shape)
}


// --- Tests ---
#[cfg(test)]
#[path = "matmul_test.rs"]
mod tests; 