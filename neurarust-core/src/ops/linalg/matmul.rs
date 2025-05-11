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

/// Backward pass structure for the 2D matrix multiplication operation (`matmul_op`).
///
/// Stores references to the input tensor nodes (`a_node`, `b_node`) and flags
/// indicating if they require gradients, used for graph traversal and gradient computation.
#[derive(Debug)]
struct MatmulBackward {
    /// Reference counted pointer to the first input tensor's data (`a` in `a @ b`).
    a_node: Arc<RwLock<TensorData>>,
    /// Reference counted pointer to the second input tensor's data (`b` in `a @ b`).
    b_node: Arc<RwLock<TensorData>>,
    /// Flag indicating if the first input (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the second input (`b`) required gradients.
    b_requires_grad: bool,
}

// --- BackwardOp Implementation for MatmulBackward ---

impl BackwardOp for MatmulBackward {
    /// Computes gradients for the matrix multiplication operation \( Z = A \cdot B \).
    ///
    /// Using the chain rule \( \frac{dL}{dX} = \frac{dL}{dZ} \cdot \frac{dZ}{dX} \), the gradients are:
    /// \\[ \frac{dL}{dA} = \frac{dL}{dZ} \cdot \frac{dZ}{dA} = \frac{dL}{dZ} \cdot B^T \\]
    /// \\[ \frac{dL}{dB} = \frac{dL}{dZ} \cdot \frac{dZ}{dB} = A^T \cdot \frac{dL}{dZ} \\]
    ///
    /// Where \( \frac{dL}{dZ} \) is `grad_output`, and \( X^T \) denotes the transpose of matrix \( X \).
    /// The matrix multiplications involved in the gradient computation are performed using `matmul_op`.
    /// Input matrices are transposed and made contiguous before the multiplication for correctness and performance.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        println!("[DEBUG][matmul::backward] Appelé. grad_output.shape={:?}", grad_output.shape());
        let mut input_grads: Vec<Tensor> = Vec::with_capacity(2);

        // Create Tensors by cloning the Arcs stored in self
        let a_tensor = Tensor { data: self.a_node.clone() };
        let b_tensor = Tensor { data: self.b_node.clone() };

        // S'assurer que le gradient est contigu
        let grad_output_contig = grad_output.contiguous()?;

        // Calculate grad_a = grad_output @ b.T
        if self.a_requires_grad {
            let b_transposed = transpose_op(&b_tensor, 0, 1)?.contiguous()?;
            println!("[DEBUG][matmul::backward] Calcul grad_a = grad_output @ b.T, b_transposed.shape={:?}", b_transposed.shape());
            let grad_a = matmul_op(&grad_output_contig, &b_transposed)?;
            println!("[DEBUG][matmul::backward] grad_a.shape={:?}", grad_a.shape());
            input_grads.push(grad_a);
        }

        // Calculate grad_b = a.T @ grad_output
        if self.b_requires_grad {
            let a_transposed = transpose_op(&a_tensor, 0, 1)?.contiguous()?;
            println!("[DEBUG][matmul::backward] Calcul grad_b = a.T @ grad_output, a_transposed.shape={:?}", a_transposed.shape());
            let grad_b = matmul_op(&a_transposed, &grad_output_contig)?;
            println!("[DEBUG][matmul::backward] grad_b.shape={:?}", grad_b.shape());
            input_grads.push(grad_b);
        }

        Ok(input_grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` and `b` of the forward `matmul_op`.
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

/// Performs 2D matrix multiplication of two tensors (`a @ b`).
///
/// Computes the dot product between matrices `a` and `b`. The number of columns
/// in `a` must match the number of rows in `b`.
/// If `a` has shape `[m, k]` and `b` has shape `[k, n]`, the output will have shape `[m, n]`.
///
/// This operation supports automatic differentiation.
/// Currently only implemented for 2D tensors (`RankMismatch` error otherwise) on the CPU
/// with `DType::F32` or `DType::F64` (`UnsupportedOperation` or `DataTypeMismatch` error otherwise).
///
/// # Arguments
/// * `a`: The first input `Tensor` (left matrix).
/// * `b`: The second input `Tensor` (right matrix).
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the matrix product, or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError` for dimension/shape mismatches, device/dtype mismatches, or internal errors.
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

/// **(Internal)** Private kernel for CPU-based 2D matrix multiplication.
///
/// Performs the calculation \( C = A \cdot B \) using nested loops.
/// This is a basic implementation and not optimized for performance (e.g., cache efficiency).
/// It handles potential offsets and strides from the input tensor views.
///
/// # Type Constraints
/// Requires the generic type `T` to implement traits necessary for multiplication and addition
/// (`Copy`, `Default`, `AddAssign`, `Mul<Output = T>`) and `Debug`.
///
/// # Arguments
/// * `m`, `k`, `n`: Dimensions of the multiplication (A: [m, k], B: [k, n], Output: [m, n]).
/// * `a_buffer`, `b_buffer`: Slices containing the underlying data for A and B.
/// * `a_strides`, `b_strides`: Strides for accessing elements in A and B.
/// * `a_offset`, `b_offset`: Starting offsets within the data buffers.
///
/// # Returns
/// A `Result` containing a `Vec<T>` with the computed output matrix data (row-major), or a `NeuraRustError` if an index is out of bounds.
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

/// **(Internal)** Internal implementation of 2D matrix multiplication without autograd setup.
///
/// This function performs checks (device, dtype, rank, shape compatibility) and then
/// calls the appropriate `matmul_kernel` based on the `DType`.
/// It assumes inputs are 2D tensors.
///
/// # Arguments
/// * `a`: The first input `Tensor`.
/// * `b`: The second input `Tensor`.
///
/// # Returns
/// A `Result` containing the output `Tensor` or a `NeuraRustError` from checks or the kernel.
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
            drop(a_guard); drop(b_guard);

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
            drop(a_guard); drop(b_guard);

            let output_data: Vec<f64> = matmul_kernel(
                m, k1, n,
                a_buffer_slice, &a_strides, a_offset,
                b_buffer_slice, &b_strides, b_offset,
            )?;
            Tensor::new_f64(output_data, output_shape)?
        }
        DType::I32 => {
            return Err(NeuraRustError::UnsupportedOperation(
                "Matrix multiplication not supported for I32 type".to_string(),
            ))
        }
        DType::I64 => {
            return Err(NeuraRustError::UnsupportedOperation(
                "Matrix multiplication not supported for I64 type".to_string(),
            ))
        }
        DType::Bool => {
            return Err(NeuraRustError::UnsupportedOperation(
                "Matrix multiplication not supported for Bool type".to_string(),
            ))
        }
    };

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
#[path = "matmul_test.rs"]
mod tests; 