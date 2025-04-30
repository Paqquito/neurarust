// Ce module contiendra les opérations d'algèbre linéaire comme matmul.

use crate::tensor::Tensor;
use crate::autograd::{BackwardOp, accumulate_gradient};
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
use crate::error::NeuraRustError;
 // Import add/mul for backward
use std::iter::Sum;
use crate::tensor::utils::reduce_gradient;

/// Represents the Matrix Multiplication operation.
#[derive(Debug)]
struct MatMulBackward<T> {
    // Need to store references to inputs for backward pass
    // Using clones might be easier than weak refs if we need their data directly
    a: Tensor<T>, 
    b: Tensor<T>,
    a_ref: Weak<RefCell<TensorData<T>>>, 
    b_ref: Weak<RefCell<TensorData<T>>>, 
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T> MatMulBackward<T> {
    /* Field is never used
    fn new(a: &Tensor<T>, b: &Tensor<T>) -> Self 
    where T: Clone + 'static 
    {
        MatMulBackward {
            input_a_shape: a.shape(),
            input_b_shape: b.shape(),
            input_a_ref: a.get_weak_ref(),
            input_b_ref: b.get_weak_ref(),
            input_a_val: a.clone(), // Clone needed tensors
            input_b_val: b.clone(),
            _phantom: PhantomData,
        }
    }
    */
}

impl<T> BackwardOp<T> for MatMulBackward<T> 
where
    // Added PartialEq, Default to existing bounds
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + One + Zero + Add<Output=T> + Debug + PartialEq + Default + Sum,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        let grad_a_unreduced = matmul(upstream_grad, &self.b.transpose())
            .expect("Matmul backward failed calculating grad_a");
        let grad_a = reduce_gradient(&grad_a_unreduced, &self.a_shape);
        
        let grad_b_unreduced = matmul(&self.a.transpose(), upstream_grad)
             .expect("Matmul backward failed calculating grad_b");
        let grad_b = reduce_gradient(&grad_b_unreduced, &self.b_shape);
        grad_b.set_requires_grad(false);

        if let Some(a_rc) = self.a_ref.upgrade() {
             if a_rc.borrow().requires_grad {
                 accumulate_gradient(gradients, &self.a_ref, grad_a);
             }
        }
         if let Some(b_rc) = self.b_ref.upgrade() {
             if b_rc.borrow().requires_grad {
                 accumulate_gradient(gradients, &self.b_ref, grad_b);
             }
         }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.a_ref.clone(), self.b_ref.clone()]
    }
}

/// Performs matrix multiplication C = A @ B.
/// Currently supports only 2D tensors (matrices).
/// A: [M, K], B: [K, N] -> C: [M, N]
pub fn matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + One + Zero + Add<Output=T> + Debug + PartialEq + Default + Sum,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Validation
    if a_shape.len() != 2 || b_shape.len() != 2 {
        // Use correct error variant IncompatibleShapes
        return Err(NeuraRustError::IncompatibleShapes {
            shape1: a_shape.clone(),
            shape2: b_shape.clone(),
            // Add context if needed, or rely on the default message
            // detail: format!("Matmul inputs must be 2D. Got A: {:?}, B: {:?}", a_shape, b_shape)
        });
    }
    if a_shape[1] != b_shape[0] {
         // Use correct error variant IncompatibleShapes
         return Err(NeuraRustError::IncompatibleShapes {
            shape1: a_shape.clone(),
            shape2: b_shape.clone(),
             // detail: format!("Matmul inner dimensions mismatch: {} != {}", a_shape[1], b_shape[0])
        });
    }

    let m = a_shape[0];
    let k = a_shape[1]; // == b_shape[0]
    let n = b_shape[1];

    let output_shape = vec![m, n];
    let mut output_data = vec![T::zero(); m * n];

    let a_data = a.data(); // Borrow starts here
    let b_data = b.data(); // Borrow starts here

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            output_data[i * n + j] = sum;
        }
    } // Borrows of a_data and b_data end here
    
    // Move drops after the borrows end
    drop(a_data);
    drop(b_data);

    let a_ref_for_backward = a.get_weak_ref();
    let b_ref_for_backward = b.get_weak_ref();
    let a_shape_clone = a_shape.clone();
    let b_shape_clone = b_shape.clone();
    let a_clone = a.clone(); 
    let b_clone = b.clone();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let result = Tensor::new(output_data, output_shape)?; 

    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = MatMulBackward {
            a_ref: a_ref_for_backward,
            b_ref: b_ref_for_backward,
            a_shape: a_shape_clone,
            b_shape: b_shape_clone,
            a: a_clone,
            b: b_clone,
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn))); 
    }
    Ok(result)
}

impl<T> Tensor<T> {
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Mul<Output=T> + Default + Sum + 'static,
    {
        matmul(self, other)
            .unwrap_or_else(|e| panic!("Tensor matmul failed: {:?}", e))
    }
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
    use crate::error::NeuraRustError; // Import NeuraRustError
    use std::iter::Sum; // Import Sum for test helpers
     // For backward tests later

    // Removed Float import as tests use concrete types or simple comparisons

    // Helper function for creating tensors in tests, updated bounds
    // Ensure these bounds match what matmul function and backward require
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Mul<Output=T> + Default + Sum + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape).expect("Tensor creation failed in test")
    }
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Mul<Output=T> + Default + Sum + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
         let tensor = Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed");
         tensor
    }
    
    #[test]
    fn test_matmul_forward() {
        let a = create_test_tensor::<i32>(vec![1, 2, 3, 4], vec![2, 2]);
        let b = create_test_tensor::<i32>(vec![5, 6, 7, 8], vec![2, 2]);

        let result = matmul(&a, &b);
        assert!(result.is_ok());
        let result_tensor = result.unwrap();

        let expected_data = vec![19, 22, 43, 50];
        assert_eq!(result_tensor.data().to_vec(), expected_data);
        assert_eq!(result_tensor.shape(), vec![2, 2]);
        assert!(!result_tensor.requires_grad());

        // Test incompatible shapes - use correct error variant IncompatibleShapes
        let c = create_test_tensor::<i32>(vec![1, 2], vec![1, 2]); // a=[2,2], c=[1,2] -> k mismatch (2 != 1)
        let result_err = matmul(&a, &c);
        assert!(result_err.is_err());
        assert!(matches!(result_err.err().unwrap(), NeuraRustError::IncompatibleShapes { .. }));
        
        let d = create_test_tensor::<i32>(vec![1,2,3,4,5,6], vec![2, 3]); // d=[2,3]
        let e = create_test_tensor::<i32>(vec![1,2,3,4,5,6], vec![2, 3]); // e=[2,3] -> k mismatch (3 != 2)
        let result_err2 = matmul(&d, &e);
        assert!(result_err2.is_err());
        assert!(matches!(result_err2.err().unwrap(), NeuraRustError::IncompatibleShapes { .. }));

        // Test the Tensor method
        let result_method = a.matmul(&b); // Should still work
        assert_eq!(result_method.data().to_vec(), expected_data);
    }

    #[test]
    fn test_matmul_backward() {
        type TestType = f32;
        let a = create_test_tensor_with_grad::<TestType>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = create_test_tensor_with_grad::<TestType>(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let output = matmul(&a, &b).expect("Matmul failed in backward test setup");
        
        let loss = output.sum();
        loss.backward(None);

        let grad_a_opt = a.grad();
        assert!(grad_a_opt.is_some(), "Gradient A missing");
        let grad_a = grad_a_opt.as_ref().unwrap();
        assert_eq!(grad_a.shape(), vec![2, 2]);
        assert_eq!(grad_a.data().to_vec(), vec![11.0, 15.0, 11.0, 15.0]);

        let grad_b_opt = b.grad();
        assert!(grad_b_opt.is_some(), "Gradient B missing");
        let grad_b = grad_b_opt.as_ref().unwrap();
         assert_eq!(grad_b.shape(), vec![2, 2]);
        assert_eq!(grad_b.data().to_vec(), vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_matmul_propagate_requires_grad() {
        let a_grad = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1, 1]);
        let b_grad = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1, 1]);
        let a_no_grad = create_test_tensor::<f32>(vec![3.0], vec![1, 1]);
        let b_no_grad = create_test_tensor::<f32>(vec![4.0], vec![1, 1]);

        let res1 = matmul(&a_grad, &b_grad).unwrap();
        assert!(res1.requires_grad());
        assert!(res1.grad_fn().is_some());

        let res2 = matmul(&a_grad, &b_no_grad).unwrap();
        assert!(res2.requires_grad());
        assert!(res2.grad_fn().is_some());

        let res3 = matmul(&a_no_grad, &b_grad).unwrap();
        assert!(res3.requires_grad());
        assert!(res3.grad_fn().is_some());

        let res4 = matmul(&a_no_grad, &b_no_grad).unwrap();
        assert!(!res4.requires_grad());
        assert!(res4.grad_fn().is_none());
    }
}