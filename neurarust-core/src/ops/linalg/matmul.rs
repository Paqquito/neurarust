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

    // --- Calculation ---
    let output_shape = vec![m, n];
    let mut output_data = vec![T::zero(); m * n];
    let output_strides = TensorData::<T>::calculate_contiguous_strides(&output_shape);

    let a_buffer = a_guard.data.cpu_data()?.clone();
    let b_buffer = b_guard.data.cpu_data()?.clone();

    for i in 0..m { // Row of output
        for j in 0..n { // Col of output
            let mut sum = T::zero();
            for k_idx in 0..k1 { // Inner dimension
                // Calculate physical index for A[i, k_idx]
                let a_coords = [i, k_idx]; // Logical coordinates
                let mut a_physical_idx = a_guard.offset;
                for dim in 0..a_guard.shape.len() {
                    a_physical_idx += a_coords[dim] * a_guard.strides[dim];
                }

                // Calculate physical index for B[k_idx, j]
                let b_coords = [k_idx, j]; // Logical coordinates
                let mut b_physical_idx = b_guard.offset;
                 for dim in 0..b_guard.shape.len() {
                    b_physical_idx += b_coords[dim] * b_guard.strides[dim];
                }

                sum += a_buffer[a_physical_idx] * b_buffer[b_physical_idx];
            }
            // Calculate physical index for Output[i, j] (Contiguous)
            let output_physical_idx = i * output_strides[0] + j * output_strides[1];
            output_data[output_physical_idx] = sum;
        }
    }

    Tensor::new(output_data, output_shape)
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::Tensor;
    use approx::assert_relative_eq;

    // Helper for f64 tests
    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_matmul_forward() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let output = matmul_op(&a, &b).unwrap();
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected_data = vec![19.0, 22.0, 43.0, 50.0];
        let output_data = output.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(output.shape(), vec![2, 2]);
        output_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e, epsilon = 1e-7));
    }

     #[test]
    fn test_matmul_forward_non_square() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::<f64>::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let output = matmul_op(&a, &b).unwrap();
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        let expected_data = vec![58.0, 64.0, 139.0, 154.0];
        let output_data = output.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(output.shape(), vec![2, 2]);
        output_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(o, e)| assert_relative_eq!(*o, *e, epsilon = 1e-7));
    }

     #[test]
    fn test_matmul_shape_mismatch_inner() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0], vec![3, 1]).unwrap(); // Inner dim mismatch (2 vs 3)
        let result = matmul_op(&a, &b);
        assert!(result.is_err());
        if let Err(NeuraRustError::ShapeMismatch { .. }) = result {
            // Correct error type
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }

     #[test]
    fn test_matmul_shape_mismatch_rank() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]).unwrap(); // Rank mismatch (2 vs 3)
        let result = matmul_op(&a, &b);
         assert!(result.is_err());
        if let Err(NeuraRustError::ShapeMismatch { operation, .. }) = result {
             assert!(operation.contains("inputs must be 2D"));
        } else {
            panic!("Expected ShapeMismatch error due to rank");
        }
    }


    #[test]
    fn test_matmul_backward_simple() {
        let a = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = create_tensor_f64_with_grad(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // Use a closure for check_grad
        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &inputs[1]);

        // Numerical gradient check requires f64
        let output = matmul_op(&a, &b).unwrap();
        let output_grad = Tensor::<f64>::ones(output.shape().clone()).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

        assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_matmul_backward_non_square() {
        let a = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = create_tensor_f64_with_grad(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &inputs[1]);

        let output = matmul_op(&a, &b).unwrap();
        let output_grad = Tensor::<f64>::ones(output.shape().clone()).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

        assert!(grad_check_result.is_ok(), "Gradient check failed for non-square: {:?}", grad_check_result.err());
    }


     #[test]
    fn test_matmul_backward_only_a_grad() {
        let a = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f64>::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap(); // b does not require grad

        // Closure captures b
        let func = |inputs: &[Tensor<f64>]| matmul_op(&inputs[0], &b);

        let output = matmul_op(&a, &b).unwrap();
        let output_grad = Tensor::<f64>::ones(output.shape().clone()).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        // check_grad only receives 'a' as the tensor to check gradient for
        let grad_check_result = check_grad(func, &[a.clone()], &output_grad, epsilon, tolerance);

        assert!(grad_check_result.is_ok(), "Gradient check failed for only a: {:?}", grad_check_result.err());

        // Also check that b.grad() is None
        let b_read = b.read_data();
        assert!(b_read.grad.is_none());
    }

    #[test]
    fn test_matmul_backward_only_b_grad() {
        let a = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(); // a does not require grad
        let b = create_tensor_f64_with_grad(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // Closure captures a
        let func = |inputs: &[Tensor<f64>]| matmul_op(&a, &inputs[0]);

        let output = matmul_op(&a, &b).unwrap();
        let output_grad = Tensor::<f64>::ones(output.shape().clone()).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        // check_grad only receives 'b' as the tensor to check gradient for
        let grad_check_result = check_grad(func, &[b.clone()], &output_grad, epsilon, tolerance);

        assert!(grad_check_result.is_ok(), "Gradient check failed for only b: {:?}", grad_check_result.err());

        // Also check that a.grad() is None
        let a_read = a.read_data();
        assert!(a_read.grad.is_none());
    }
} 