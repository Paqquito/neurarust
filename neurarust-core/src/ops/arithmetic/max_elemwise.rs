use crate::{
    autograd::{backward_op::BackwardOp, graph::NodeId},
    error::NeuraRustError,
    ops::{comparison::ge_op, traits::NeuraNumeric},
    tensor::Tensor,
    tensor_data::TensorData,
    types::DType,
    ops::arithmetic::mul_op,
};
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

/// Kernel for element-wise maximum.
#[inline]
fn max_elemwise_kernel<T>(a: T, b: T) -> T
where
    T: NeuraNumeric + PartialOrd,
{
    a.max(b)
}

// --- Backward Operation Structure ---
#[derive(Debug)]
struct MaxElementwiseBackward {
    a_node: Option<Arc<RwLock<TensorData>>>,
    b_node: Option<Arc<RwLock<TensorData>>>,
    a_clone: Tensor, // Store clones needed for gradient calculation
    b_clone: Tensor,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    a_requires_grad: bool,
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for MaxElementwiseBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut result_grads = Vec::new();

        // Calcul du masque : a >= b (désormais un tenseur Bool)
        let mask_a_ge_b_bool = ge_op(&self.a_clone, &self.b_clone)?;
        // On caste le masque en F32 pour le backward
        let mask_a_ge_b = crate::ops::dtype::cast_op(&mask_a_ge_b_bool, DType::F32)?;

        if self.a_requires_grad {
            let grad_a_unreduced = mul_op(grad_output, &mask_a_ge_b)?;
            let grad_a = grad_a_unreduced.reduce_to_shape(&self.a_shape)?;
            result_grads.push(grad_a);
        }

        // Gradient pour b : grad_output * (a < b) = grad_output * (1.0 - (a >= b))
        if self.b_requires_grad {
            let ones = crate::tensor::Tensor::new(vec![1.0; mask_a_ge_b.numel()], mask_a_ge_b.shape().to_vec())?;
            let mask_a_lt_b = crate::ops::arithmetic::sub_op(&ones, &mask_a_ge_b)?;
            let grad_b_unreduced = mul_op(grad_output, &mask_a_lt_b)?;
            let grad_b = grad_b_unreduced.reduce_to_shape(&self.b_shape)?;
            result_grads.push(grad_b);
        }

        Ok(result_grads)
    }

    fn inputs(&self) -> Vec<NodeId> {
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(self.a_node.as_ref().unwrap())); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(self.b_node.as_ref().unwrap())); }
        ids
    }
}

/// Computes the element-wise maximum between two tensors, supporting broadcasting and autograd.
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
/// * `a` - First input tensor.
/// * `b` - Second input tensor.
///
/// # Returns
/// A `Result` containing a new `Tensor` representing the element-wise maximum, or a `NeuraRustError`.
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
/// use neurarust_core::ops::arithmetic::max_elemwise;
/// let a = Tensor::new_i32(vec![1, 5, 3], vec![3]).unwrap();
/// let b = Tensor::new_i32(vec![4, 2, 6], vec![3]).unwrap();
/// let c = max_elemwise::max_elemwise_op(&a, &b).unwrap();
/// assert_eq!(c.get_i32_data().unwrap(), vec![4, 5, 6]);
/// assert_eq!(c.dtype(), DType::I32);
/// ```
pub fn max_elemwise_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_clone = a.clone();
    let b_clone = b.clone();

    // Use the centralized helper for binary ops
    crate::ops::arithmetic::apply_binary_op_broadcasted(
        a,
        b,
        // F32 kernel
        |va, vb| max_elemwise_kernel::<f32>(va, vb),
        // F64 kernel
        |va, vb| max_elemwise_kernel::<f64>(va, vb),
        // I32
        |va, vb| va.max(vb),
        // I64
        |va, vb| va.max(vb),
        // Build backward op closure
        move |a_node_opt, b_node_opt, a_shape, b_shape, a_req, b_req| {
            Arc::new(MaxElementwiseBackward {
                a_node: a_node_opt,
                b_node: b_node_opt,
                a_clone, // Move clones into the closure
                b_clone,
                a_shape,
                b_shape,
                a_requires_grad: a_req,
                b_requires_grad: b_req,
            })
        },
        "max_elemwise_op", // Operation name
    )
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::create;
    use crate::utils::testing::check_tensor_near;
    use crate::autograd::grad_check::{GradCheckError, check_grad};

    #[test]
    fn test_max_elemwise_forward_simple() -> Result<(), NeuraRustError> {
        let a = create::from_vec_f32(vec![1.0, 5.0, 3.0], vec![3])?;
        let b = create::from_vec_f32(vec![2.0, 5.0, 2.0], vec![3])?;
        let result = max_elemwise_op(&a, &b)?;
        check_tensor_near(&result, &[3], &[2.0, 5.0, 3.0], 1e-7);
        Ok(())
    }

    #[test]
    fn test_max_elemwise_forward_broadcast() -> Result<(), NeuraRustError> {
        let a = create::from_vec_f32(vec![3.0], vec![1])?;
        let b = create::from_vec_f32(vec![1.0, 4.0, 3.0, 2.0], vec![2, 2])?;
        let result = max_elemwise_op(&a, &b)?;
        check_tensor_near(&result, &[2, 2], &[3.0, 4.0, 3.0, 3.0], 1e-7);
        Ok(())
    }

    #[test]
    fn test_max_elemwise_forward_f64() -> Result<(), NeuraRustError> {
        let a = create::from_vec_f64(vec![1.0, 5.0, 3.0], vec![3])?;
        let b = create::from_vec_f64(vec![2.0, 5.0, 2.0], vec![3])?;
        let result = max_elemwise_op(&a, &b)?;
        let data = result.get_f64_data()?;
        assert_eq!(result.shape(), &[3]);
        assert_eq!(data, vec![2.0, 5.0, 3.0]);
        Ok(())
    }
    
    #[test]
    fn test_max_elemwise_backward_simple() -> Result<(), GradCheckError> {
        let a = create::from_vec_f32(vec![1.0, 6.0, 3.0], vec![3])?;
        a.set_requires_grad(true)?;
        let b = create::from_vec_f32(vec![2.0, 5.0, 4.0], vec![3])?;
        b.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| max_elemwise_op(&inputs[0], &inputs[1]);
        
        let grad_output = create::ones_like(&a)?; 
        
        check_grad(func, &[a, b], &grad_output, 1e-3, 1e-4, 1e-3)
    }
    
    #[test]
    fn test_max_elemwise_backward_broadcast() -> Result<(), GradCheckError> {
        let a = create::from_vec_f32(vec![3.5], vec![1])?;
        a.set_requires_grad(true)?;
        let b = create::from_vec_f32(vec![1.0, 4.0, 3.0, 2.0], vec![2, 2])?;
        b.set_requires_grad(true)?;
        
        let func = |inputs: &[Tensor]| max_elemwise_op(&inputs[0], &inputs[1]);
        
        let output_shape = vec![2, 2];
        let grad_output = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], output_shape)?;
        
        check_grad(func, &[a, b], &grad_output, 1e-3, 1e-4, 1e-3)
    }
} 