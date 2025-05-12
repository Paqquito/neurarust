// neurarust-core/src/ops/arithmetic/add.rs

use crate::{
    autograd::backward_op::BackwardOp,
    error::NeuraRustError,
    ops::traits::NeuraNumeric,
    tensor::Tensor,
    tensor_data::TensorData,
};
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

/// Generic kernel for element-wise addition.
fn add_kernel<T: NeuraNumeric>(a: T, b: T) -> T {
    a + b
}

// --- Backward Operation Structure ---

/// Backward pass structure for the element-wise addition operation.
///
/// Stores references to the input tensor nodes (`a_node`, `b_node`), their original
/// shapes (`a_shape`, `b_shape`), and flags indicating if they require gradients.
/// The original shapes are crucial for reducing the output gradient back to the
/// correct input shapes if broadcasting occurred during the forward pass.
#[derive(Debug)]
struct AddBackward {
    /// Reference counted pointer to the first input tensor's data (`a`).
    a_node: Arc<RwLock<TensorData>>,
    /// Reference counted pointer to the second input tensor's data (`b`).
    b_node: Arc<RwLock<TensorData>>,
    /// Original shape of the first input tensor (`a`).
    a_shape: Vec<usize>,
    /// Original shape of the second input tensor (`b`).
    b_shape: Vec<usize>,
    /// Flag indicating if the first input tensor (`a`) required gradients.
    a_requires_grad: bool,
    /// Flag indicating if the second input tensor (`b`) required gradients.
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for AddBackward {
    /// Computes gradients for the addition operation \( z = a + b \).
    ///
    /// The gradient \( \frac{dL}{dz} \) received (`grad_output`) is passed back to both
    /// inputs `a` and `b` because the local derivatives \( \frac{dz}{da} = 1 \) and
    /// \( \frac{dz}{db} = 1 \).
    ///
    /// If broadcasting occurred, `grad_output` is reduced to the original shapes of `a` and `b`
    /// using [`reduce_gradient_to_shape`] before being returned.
    ///
    /// # Arguments
    /// * `grad_output`: The gradient tensor flowing back, corresponding to the output `z`.
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of gradient tensors corresponding to the inputs
    /// `a` and `b` (in that order) that required gradients. If an input did not require
    /// gradients, its corresponding gradient is not computed or returned.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::with_capacity(2);
        //println!("[ADD_BACKWARD] backward called. grad_output shape: {:?}, dtype: {:?}", grad_output.shape(), grad_output.dtype());
        //println!("[ADD_BACKWARD] a_requires_grad: {}, a_shape: {:?}, b_requires_grad: {}, b_shape: {:?}", self.a_requires_grad, self.a_shape, self.b_requires_grad, self.b_shape);

        if self.a_requires_grad {
            let grad_a = grad_output.reduce_to_shape(&self.a_shape)?;
            //println!("[ADD_BACKWARD] Calculated grad_a shape: {:?}, dtype: {:?}", grad_a.shape(), grad_a.dtype());
            grads.push(grad_a);
        }
        if self.b_requires_grad {
            let grad_b = grad_output.reduce_to_shape(&self.b_shape)?;
            //println!("[ADD_BACKWARD] Calculated grad_b shape: {:?}, dtype: {:?}", grad_b.shape(), grad_b.dtype());
            grads.push(grad_b);
        }
        //println!("[ADD_BACKWARD] Returning grads. Count: {}", grads.len());
        Ok(grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `a` and `b` of the forward `add_op`.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        // Return IDs of inputs that required grad, in the order (a, b)
        let mut ids = Vec::new();
        //println!("[ADD_BACKWARD] inputs() called.");
        if self.a_requires_grad { 
            let ptr_a = Arc::as_ptr(&self.a_node);
            //println!("[ADD_BACKWARD] Adding a_node to inputs. Ptr: {:?}", ptr_a);
            ids.push(ptr_a);
        }
        if self.b_requires_grad { 
            let ptr_b = Arc::as_ptr(&self.b_node);
            //println!("[ADD_BACKWARD] Adding b_node to inputs. Ptr: {:?}", ptr_b);
            ids.push(ptr_b);
        }
        //println!("[ADD_BACKWARD] Returning input pointers. Count: {}", ids.len());
        ids
    }
}

// --- Forward Operation ---

/// Performs element-wise addition between two tensors, supporting broadcasting and autograd.
///
/// # Supported Types
/// - `DType::F32`
/// - `DType::F64`
/// - `DType::I32`
/// - `DType::I64`
///
/// # Unsupported Types
/// - `DType::Bool` (returns an `UnsupportedOperation` error)
///
/// # Arguments
/// * `a` - First input tensor (addend).
/// * `b` - Second input tensor (addend).
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise sum, or a `NeuraRustError`.
///
/// # Errors
/// - `DeviceMismatch` if tensors are not on the CPU.
/// - `DataTypeMismatch` if the DTypes do not match.
/// - `BroadcastError` if the shapes are not broadcast-compatible.
/// - `UnsupportedOperation` if the DType is not supported.
/// - `InternalError` for internal errors.
///
/// # Example
/// ```
/// use neurarust_core::{Tensor, DType};
/// use neurarust_core::ops::arithmetic::add_op;
/// let a = Tensor::new_i32(vec![1, 2, 3], vec![3]).unwrap();
/// let b = Tensor::new_i32(vec![4, 5, 6], vec![3]).unwrap();
/// let c = add_op(&a, &b).unwrap();
/// assert_eq!(c.get_i32_data().unwrap(), vec![5, 7, 9]);
/// assert_eq!(c.dtype(), DType::I32);
/// ```
pub fn add_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    crate::ops::arithmetic::apply_binary_op_broadcasted(
        a,
        b,
        |va, vb| add_kernel::<f32>(va, vb),
        |va, vb| add_kernel::<f64>(va, vb),
        |va, vb| va + vb, // I32
        |va, vb| va + vb, // I64
        |a_node_opt, b_node_opt, a_shape, b_shape, a_req, b_req| {
            let a_node = a_node_opt.expect("a_node must be Some if requires_grad is true");
            let b_node = b_node_opt.expect("b_node must be Some if requires_grad is true");
            Arc::new(AddBackward {
                a_node,
                b_node,
                a_shape,
                b_shape,
                a_requires_grad: a_req,
                b_requires_grad: b_req,
            })
        },
        "add_op",
    )
}

// Re-enable the test module link
#[cfg(test)]
#[path = "add_test.rs"]
mod tests;
