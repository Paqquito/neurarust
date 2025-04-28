// Ce module contiendra les opérations d'algèbre linéaire comme matmul.

use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData; // Use correct path
use num_traits::{Zero, One};
use std::ops::{Add, Mul, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::cmp::PartialEq;
use std::collections::HashMap;

#[derive(Debug)]
struct MatmulBackward<T> {
    input_a: Tensor<T>, // Need clones of inputs for matmul gradient calculation
    input_b: Tensor<T>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MatmulBackward<T> 
where
    // Ensure bounds cover matmul (Mul+AddAssign+Zero+...) and transpose requirements
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + One + Zero + Add<Output=T> + Debug + PartialEq,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        // Calculate gradient for input A: dL/dA = dL/dC @ B.T
        // Need to check if input A requires gradient before computing
        let needs_grad_a = self.input_a_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        if needs_grad_a {
            let input_b_transposed = self.input_b.transpose();
            let grad_a = upstream_grad.matmul(&input_b_transposed);
            crate::autograd::accumulate_gradient(gradients, &self.input_a_ref, grad_a);
        }
        
        // Calculate gradient for input B: dL/dB = A.T @ dL/dC
        // Need to check if input B requires gradient before computing
        let needs_grad_b = self.input_b_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        if needs_grad_b {
             let input_a_transposed = self.input_a.transpose();
             let grad_b = input_a_transposed.matmul(upstream_grad); // Use original upstream_grad
             crate::autograd::accumulate_gradient(gradients, &self.input_b_ref, grad_b);
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_a_ref.clone(), self.input_b_ref.clone()]
    }
}

impl<T> Tensor<T> {
    /// Performs matrix multiplication (matmul) between two 2D tensors (matrices).
    ///
    /// Calculates `self * other`.
    /// `self` must have shape `(M, K)` and `other` must have shape `(K, N)`.
    /// The resulting tensor will have shape `(M, N)`.
    ///
    /// Requires the element type `T` to implement `Add<Output = T>`, `Mul<Output = T>`, `Zero`, and `Copy`.
    /// Uses a naive triple-loop algorithm for now.
    ///
    /// # Panics
    /// - Panics if either tensor is not 2-dimensional.
    /// - Panics if the inner dimensions (`K`) do not match.
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: Copy + Mul<Output=T> + AddAssign + Zero + Clone + 'static + One + Add<Output=T> + Debug + PartialEq,
    {
        let a_td = self.borrow_tensor_data();
        let b_td = other.borrow_tensor_data();

        assert!(a_td.shape.len() == 2 && b_td.shape.len() == 2, "Matmul requires 2D tensors.");
        assert_eq!(a_td.shape[1], b_td.shape[0], "Matmul dimension mismatch: A shape {:?} B shape {:?}", a_td.shape, b_td.shape);

        let m = a_td.shape[0];
        let k = a_td.shape[1]; // Must be equal to b_td.shape[0]
        let n = b_td.shape[1];

        let mut data = vec![T::zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    // Try using Add instead of AddAssign
                    sum = sum + (a_td.data[i * k + l] * b_td.data[l * n + j]);
                }
                data[i * n + j] = sum;
            }
        }

        let new_shape = vec![m, n];
        
        // Keep borrows alive until here if needed, but we copied shape dims
        drop(a_td);
        drop(b_td);

        let result = Tensor::new(data, new_shape);

        // Check if requires_grad and setup backward pass
        let requires_grad = self.requires_grad() || other.requires_grad();
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = MatmulBackward {
                input_a: self.clone(), // Clone inputs needed for backward
                input_b: other.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
                _phantom: PhantomData,
            };
             // Now we can assign grad_fn as T satisfies the bounds
             result.data.borrow_mut().grad_fn = Some(Rc::new(grad_fn)); 
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::{Zero, One}; // Keep One for helpers if needed
    use std::ops::{AddAssign, Mul, Add}; // Add std::ops::Add here too

    // Updated helper with full bounds needed for matmul
    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where T: Copy + Mul<Output=T> + AddAssign + Zero + Clone + 'static + One + Add<Output=T> + std::fmt::Debug + PartialEq
    {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_matmul_forward_2x2() {
        let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected_data = vec![19.0, 22.0, 43.0, 50.0];
        let result = a.matmul(&b);
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), vec![2, 2]);
    }
    
     #[test]
    fn test_matmul_forward_2x3_3x2() {
        let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]] = [[58, 64], [139, 154]]
        let expected_data = vec![58.0, 64.0, 139.0, 154.0];
        let result = a.matmul(&b);
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_propagate_requires_grad() { 
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul(&b);
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let c = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let d = create_test_tensor_with_grad::<f32>(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result2 = c.matmul(&d);
        assert!(result2.requires_grad());
        assert!(result2.grad_fn().is_some());

        let e = Tensor::new(vec![1.0_f32, 2.0], vec![1, 2]);
        let f = Tensor::new(vec![3.0_f32, 4.0], vec![2, 1]);
        let result3 = e.matmul(&f);
        assert!(!result3.requires_grad());
        assert!(result3.grad_fn().is_none());
     }
    
    #[test] 
    #[should_panic(expected = "Matmul dimension mismatch")] 
    fn test_matmul_dimension_mismatch() { 
        let a = Tensor::<f32>::new(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::<f32>::new(vec![3.0, 4.0], vec![1, 2]); // Should be 2xN
         let _ = a.matmul(&b);
     }
    
    #[test] 
    #[should_panic(expected = "Matmul requires 2D tensors")] 
    fn test_matmul_first_arg_not_2d() { 
        let a = Tensor::<f32>::new(vec![1.0, 2.0], vec![2]); 
        let b = Tensor::<f32>::new(vec![3.0, 4.0], vec![2, 1]);
        let _ = a.matmul(&b);
     }
    
    #[test] 
    #[should_panic(expected = "Matmul requires 2D tensors")] 
    fn test_matmul_second_arg_not_2d() { 
        let a = Tensor::<f32>::new(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::<f32>::new(vec![3.0, 4.0], vec![2]); 
        let _ = a.matmul(&b);
     }
    
    // test_matmul_2x2 and test_matmul_2x3_3x2 are now forward tests
    // #[test] fn test_matmul_2x2() { /* Placeholder -> Replaced by forward test */ }
    // #[test] fn test_matmul_2x3_3x2() { /* Placeholder -> Replaced by forward test */ }

    #[test]
    fn test_matmul_backward() {
        let a = create_test_tensor_with_grad(vec![1.0_f32, 2.0], vec![1, 2]); // Shape(1, 2)
        let b = create_test_tensor_with_grad(vec![3.0_f32, 4.0], vec![2, 1]); // Shape(2, 1)
        
        // c = a @ b = [[1*3 + 2*4]] = [[11.0]] -> Shape(1, 1)
        let c = a.matmul(&b);
        assert_eq!(c.data().to_vec(), vec![11.0]);
        assert_eq!(c.shape(), vec![1, 1]);
        assert!(c.requires_grad());

        // Use the global backward function on the result tensor
        // Need a scalar loss for backward(None)
        let loss = c.sum(); // loss = 11.0, shape []
        loss.backward(None); // Upstream grad for c will be [[1.0]]

        // Check gradients obtained via Tensor::grad()
        let grad_a = a.grad().expect("Grad A missing");
        let grad_b = b.grad().expect("Grad B missing");

        // Expected gradients:
        // grad_A = dLoss/dA = (dLoss/dC * dC/dA)
        // dLoss/dC = [[1.0]] (Shape 1,1)
        // B.T = [[3.0, 4.0]] (Shape 1,2)
        // grad_A = [[1.0]] @ [[3.0, 4.0]] = [[1*3, 1*4]] = [[3.0, 4.0]] -> Shape(1, 2)
        let expected_grad_a = vec![3.0_f32, 4.0];
        assert_eq!(grad_a.shape(), vec![1, 2]);
        assert_eq!(grad_a.data().to_vec(), expected_grad_a);
        
        // grad_B = dLoss/dB = (dLoss/dC * dC/dB)
        // A.T = [[1.0], [2.0]] (Shape 2,1)
        // dLoss/dC = [[1.0]] (Shape 1,1)
        // grad_B = [[1.0], [2.0]] @ [[1.0]] = [[1.0*1], [2.0*1]] = [[1.0], [2.0]] -> Shape(2, 1)
        let expected_grad_b = vec![1.0_f32, 2.0];
        assert_eq!(grad_b.shape(), vec![2, 1]);
        assert_eq!(grad_b.data().to_vec(), expected_grad_b);
    }
    
    #[test]
    fn test_matmul_backward_larger() {
         let a = create_test_tensor_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
         let b = create_test_tensor_with_grad(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
         // c = [[19, 22], [43, 50]]
         let c = a.matmul(&b);

         // To make backward work on non-scalar, we need to provide upstream grad
         // Or sum the result to get a scalar loss
         let loss = c.sum(); // loss = 19+22+43+50 = 134
         loss.backward(None);
         
         // Expected gradients (dL/dc = [[1, 1], [1, 1]] from sum):
         // grad_A = dL/dc @ B.T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
         let expected_grad_a = vec![11.0_f32, 15.0, 11.0, 15.0];
         // grad_B = A.T @ dL/dc = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
         let expected_grad_b = vec![4.0_f32, 4.0, 6.0, 6.0];
         
        let grad_a = a.grad().expect("Grad for A missing");
        assert_eq!(grad_a.data().to_vec(), expected_grad_a);
        assert_eq!(grad_a.shape(), vec![2, 2]);

        let grad_b = b.grad().expect("Grad for B missing");
        assert_eq!(grad_b.data().to_vec(), expected_grad_b);
        assert_eq!(grad_b.shape(), vec![2, 2]);
    }
} 