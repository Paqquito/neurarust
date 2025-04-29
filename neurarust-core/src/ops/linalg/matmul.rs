// Ce module contiendra les opérations d'algèbre linéaire comme matmul.

use crate::tensor::Tensor;
use crate::autograd::{BackwardOp};
use crate::tensor_data::TensorData; // Use correct path
use num_traits::{Zero, One};
use std::ops::{Add, Mul, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::cmp::PartialEq;
use std::collections::HashMap;
use std::default::Default;

/// Represents the Matrix Multiplication operation.
#[derive(Debug)]
struct MatMulBackward<T> {
    // Need to store references to inputs for backward pass
    // Using clones might be easier than weak refs if we need their data directly
    a: Tensor<T>, 
    b: Tensor<T>,
    _phantom: PhantomData<T>,
}

impl<T> MatMulBackward<T> {
    fn new(a: &Tensor<T>, b: &Tensor<T>) -> Self 
    where 
        T: Clone // Clone is needed to store inputs
    {
        Self {
            a: a.clone(),
            b: b.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T> BackwardOp<T> for MatMulBackward<T> 
where
    // Added PartialEq, Default to existing bounds
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + One + Zero + Add<Output=T> + Debug + PartialEq + Default,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        let a_weak = self.a.get_weak_ref();
        let b_weak = self.b.get_weak_ref();
        let a_ptr_opt = a_weak.upgrade().map(|rc| Rc::as_ptr(&rc));
        let b_ptr_opt = b_weak.upgrade().map(|rc| Rc::as_ptr(&rc));
        println!("[MatMulBackward] backward called. a_ptr: {:?}, b_ptr: {:?}", a_ptr_opt, b_ptr_opt);

        let needs_grad_a = a_weak.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        println!("[MatMulBackward] needs_grad_a: {}", needs_grad_a);
        if needs_grad_a {
            let b_transposed = self.b.transpose();
            let grad_a = matmul(upstream_grad, &b_transposed);
            grad_a.set_requires_grad(false);
            println!("[MatMulBackward] Accumulating grad for A (ptr: {:?})", a_ptr_opt);
            crate::autograd::accumulate_gradient(gradients, &a_weak, grad_a);
        }
        
        let needs_grad_b = b_weak.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        println!("[MatMulBackward] needs_grad_b: {}", needs_grad_b);
        if needs_grad_b {
             let a_transposed = self.a.transpose();
             let grad_b = matmul(&a_transposed, upstream_grad); 
             grad_b.set_requires_grad(false);
             println!("[MatMulBackward] Accumulating grad for B (ptr: {:?})", b_ptr_opt);
             crate::autograd::accumulate_gradient(gradients, &b_weak, grad_b);
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        // Use get_weak_ref, no need to clone weak refs
        vec![self.a.get_weak_ref(), self.b.get_weak_ref()]
    }
}

/// Performs matrix multiplication C = A @ B.
/// Currently supports only 2D tensors (matrices).
/// A: [M, K], B: [K, N] -> C: [M, N]
pub fn matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    // Added PartialEq to existing bounds
    T: Mul<Output = T> + AddAssign + Zero + One + Default + Clone + Copy + Debug + 'static + Add<Output=T> + PartialEq,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    // --- Basic Shape Checks (2D only for now) ---
    assert_eq!(a_shape.len(), 2, "MatMul input A must be 2D, got shape: {:?}", a_shape);
    assert_eq!(b_shape.len(), 2, "MatMul input B must be 2D, got shape: {:?}", b_shape);
    let m = a_shape[0];
    let k_a = a_shape[1];
    let k_b = b_shape[0];
    let n = b_shape[1];
    assert_eq!(k_a, k_b, 
        "MatMul dimension mismatch: A shape {:?} vs B shape {:?}. Inner dimensions ({} and {}) must match.",
        a_shape, b_shape, k_a, k_b);

    let output_shape = vec![m, n];
    let mut output_data = vec![T::zero(); m * n];

    let a_data = a.borrow_tensor_data();
    let b_data = b.borrow_tensor_data();

    // --- Naive MatMul Calculation (C[i,j] = sum(A[i,k]*B[k,j])) ---
    for i in 0..m {         
        for j in 0..n {     
            let mut sum = T::zero();
            for k in 0..k_a { 
                // A[i, k] -> index i * k_a + k
                // B[k, j] -> index k * n + j  <-- Corrected index for B
                let a_val = a_data.data[i * k_a + k];
                let b_val = b_data.data[k * n + j]; 
                sum += a_val * b_val;
            }
            // C[i, j] -> index i * n + j
            output_data[i * n + j] = sum;
        }
    }
    
    drop(a_data);
    drop(b_data);

    // --- Autograd Setup ---
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = Tensor::new(output_data, output_shape);

    if requires_grad {
        output_tensor.set_requires_grad(true);
        let matmul_backward: Rc<dyn BackwardOp<T>> = Rc::new(MatMulBackward::new(a, b));
        output_tensor.set_grad_fn(Some(matmul_backward));
    }

    output_tensor
}

#[cfg(test)]
mod tests {
    // Explicitly import the matmul function using its full path
    use crate::ops::linalg::matmul;
    // Import necessary tensor components
    use crate::tensor::Tensor;
    use num_traits::{Zero, One}; // Keep for helpers
    use std::fmt::Debug;
    use std::ops::{AddAssign, Mul, Add};
    use std::default::Default;
    use std::cmp::PartialEq;
    use std::collections::HashMap; // For backward tests later

    // Removed Float import as tests use concrete types or simple comparisons

    // Helper function for creating tensors in tests, updated bounds
    // Ensure these bounds match what matmul function and backward require
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Mul<Output=T> + Default + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Mul<Output=T> + Default + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }
    
    #[test]
    fn test_matmul_forward_2x2() {
        let a = create_test_tensor(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = create_test_tensor(vec![5.0f64, 6.0, 7.0, 8.0], vec![2, 2]);
        // Use the explicitly imported function
        let result = matmul(&a, &b);
        let expected_data = vec![19.0, 22.0, 43.0, 50.0];
        let expected_shape = vec![2, 2];
        assert_eq!(result.shape(), expected_shape);
        assert_eq!(result.borrow_tensor_data().data.to_vec(), expected_data);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_matmul_forward_2x3_3x2() {
        let a = create_test_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = create_test_tensor(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        // Use the explicitly imported function
        let result = matmul(&a, &b);
        let expected_data = vec![58.0, 64.0, 139.0, 154.0];
        let expected_shape = vec![2, 2];
        assert_eq!(result.shape(), expected_shape);
        assert_eq!(result.borrow_tensor_data().data.to_vec(), expected_data);
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic(expected = "MatMul dimension mismatch")]
    fn test_matmul_dimension_mismatch() {
        let a = create_test_tensor(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let b_wrong_k = create_test_tensor(vec![5.0f64, 6.0, 7.0, 8.0, 9.0, 10.0], vec![3, 2]);
        // Use the explicitly imported function
        let _result = matmul(&a, &b_wrong_k);
    }

    #[test]
    #[should_panic(expected = "MatMul input A must be 2D")]
    fn test_matmul_first_arg_not_2d() {
        let a_1d = create_test_tensor(vec![1.0f64, 2.0], vec![2]);
        let b = create_test_tensor(vec![5.0f64, 6.0], vec![2, 1]);
        // Use the explicitly imported function
        let _result = matmul(&a_1d, &b);
    }
    
    #[test]
    #[should_panic(expected = "MatMul input B must be 2D")]
    fn test_matmul_second_arg_not_2d() {
        let a = create_test_tensor(vec![1.0f64, 2.0], vec![1, 2]);
        let b_1d = create_test_tensor(vec![5.0f64, 6.0], vec![2]);
        // Use the explicitly imported function
        let _result = matmul(&a, &b_1d);
    }

    #[test]
    fn test_matmul_propagate_requires_grad() {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = create_test_tensor::<f32>(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        // Use the explicitly imported function
        let result = matmul(&a, &b);
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let c = create_test_tensor::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let d = create_test_tensor_with_grad::<f32>(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        // Use the explicitly imported function
        let result2 = matmul(&c, &d);
        assert!(result2.requires_grad());
        assert!(result2.grad_fn().is_some());

        let e = create_test_tensor::<f32>(vec![1.0, 2.0], vec![1, 2]);
        let f = create_test_tensor::<f32>(vec![3.0, 4.0], vec![2, 1]);
        // Use the explicitly imported function
        let result3 = matmul(&e, &f);
        assert!(!result3.requires_grad());
        assert!(result3.grad_fn().is_none());
    }
    
    // TODO: Add autograd tests (backward simple, backward chain)
}