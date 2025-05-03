use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::ops::view::transpose_op; // Needed for backward

use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::sync::{Arc, RwLock};

// --- MatmulBackward Definition ---

/// Backward operation context for `matmul_op`.
#[derive(Debug)]
struct MatmulBackward<T>
where
    T: Debug + Copy + Send + Sync + 'static + Default + Zero + One + Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + PartialEq + PartialOrd + std::iter::Sum + std::iter::Product,
{
    a_node: Arc<RwLock<TensorData<T>>>,
    b_node: Arc<RwLock<TensorData<T>>>,
    a_requires_grad: bool,
    b_requires_grad: bool,
}

// --- BackwardOp Implementation for MatmulBackward ---

impl<T> BackwardOp<T> for MatmulBackward<T>
where
    T: Debug + Copy + Send + Sync + 'static + Default + Zero + One + Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + PartialEq + PartialOrd + std::iter::Sum + std::iter::Product,
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let mut input_grads: Vec<Tensor<T>> = Vec::with_capacity(2);

        let a_tensor = Tensor { data: self.a_node.clone() };
        let b_tensor = Tensor { data: self.b_node.clone() };

        // Calculate grad_a = grad_output * b.T
        if self.a_requires_grad {
            let b_transposed = transpose_op(&b_tensor, 0, 1)?;
            let grad_a = matmul_op(grad_output, &b_transposed)?;
            input_grads.push(grad_a);
        }

        // Calculate grad_b = a.T * grad_output
        if self.b_requires_grad {
            let a_transposed = transpose_op(&a_tensor, 0, 1)?;
            let grad_b = matmul_op(&a_transposed, grad_output)?;
            input_grads.push(grad_b);
        }

        Ok(input_grads)
    }

    fn inputs(&self) -> Vec<NodeId<T>> {
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

// --- matmul_op Implementation (Public API with Autograd) ---

/// Performs 2D matrix multiplication (A @ B).
pub fn matmul_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Debug + Copy + Send + Sync + 'static + Default + Zero + One + Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + PartialEq + PartialOrd + std::iter::Sum + std::iter::Product,
{
    let a_requires_grad = a.requires_grad();
    let b_requires_grad = b.requires_grad();
    let requires_grad = a_requires_grad || b_requires_grad;

    // Clone Arcs needed for backward pass *before* calling internal matmul
    let a_node_arc = if requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if requires_grad { Some(b.data.clone()) } else { None };

    // Perform the actual matrix multiplication
    let output_tensor = matmul_internal(a, b)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let (Some(a_arc), Some(b_arc)) = (a_node_arc, b_node_arc) {
            let grad_fn = MatmulBackward {
                a_node: a_arc,
                b_node: b_arc,
                a_requires_grad,
                b_requires_grad,
            };
            output_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
            output_tensor.set_requires_grad(true)?;
        } else {
            return Err(NeuraRustError::InternalError(
                "Input requires_grad but Arc could not be cloned for Matmul".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// --- matmul_kernel (Private Calculation Core) ---

/// Private kernel for matrix multiplication calculation.
/// Assumes shapes are validated and buffers are accessible.
fn matmul_kernel<T>(
    m: usize,
    k: usize, // Inner dimension (k1 == k2)
    n: usize,
    a_buffer: &[T],
    a_strides: &[usize],
    a_offset: usize,
    b_buffer: &[T],
    b_strides: &[usize],
    b_offset: usize,
) -> Result<Vec<T>, NeuraRustError>
where
     T: Debug + Copy + Send + Sync + 'static + Default + Zero + One + Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign,
{
    let mut output_data = vec![T::zero(); m * n];
    // Output is contiguous, calculate its strides implicitly or pass them?
    // For simplicity, calculate index directly: output_physical_idx = i * n + j;

    for i in 0..m { // Row of output
        for j in 0..n { // Col of output
            let mut sum = T::zero();
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

/// Internal implementation of 2D matrix multiplication without autograd setup.
fn matmul_internal<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
     T: Debug + Copy + Send + Sync + 'static + Default + Zero + One + Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign,
{
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Checks --- (Simplified: Assume CPU)
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "Matmul currently only supports CPU".to_string(),
        ));
    }

    // --- Shape Checks ---
    if a_guard.shape.len() != 2 || b_guard.shape.len() != 2 {
        return Err(NeuraRustError::ShapeMismatch {
            expected: vec![2], // Indicating rank 2
            actual: if a_guard.shape.len() != 2 { a_guard.shape.clone() } else { b_guard.shape.clone() },
            operation: "matmul (inputs must be 2D)".to_string(),
        });
    }
    let m = a_guard.shape[0];
    let k1 = a_guard.shape[1];
    let k2 = b_guard.shape[0];
    let n = b_guard.shape[1];

    if k1 != k2 {
         return Err(NeuraRustError::ShapeMismatch {
            expected: vec![m, k1], // Or some way to show inner dim match
            actual: vec![k2, n],
            operation: format!("matmul (inner dimensions must match: {} != {})", k1, k2),
        });
    }

    // --- Extract data for kernel ---
    let a_buffer_arc = a_guard.data.cpu_data()?.clone();
    let b_buffer_arc = b_guard.data.cpu_data()?.clone();
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