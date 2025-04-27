pub mod transpose;
pub mod matmul;

// Placeholder for Matmul implementation (forward and backward)
// This file might contain the matmul forward function itself,
// or just the MatmulBackward struct if matmul is a method on Tensor.

use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{AddAssign, Mul};
use std::rc::{Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use num_traits::{Zero, One}; // Add One here

// --- Matmul Backward Operation --- 

// Struct definition remains
struct MatmulBackward<T> {
    input_a: Tensor<T>,
    input_b: Tensor<T>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

// impl BackwardOp will be updated later using transpose
impl<T> BackwardOp<T> for MatmulBackward<T> 
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + One + Zero, // Use imported One
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // TODO: Implement using transpose
        // grad_A = upstream_grad @ B.T
        // grad_B = A.T @ upstream_grad
        println!("MatmulBackward: backward called (gradient accumulation pending - requires transpose and matmul)");
        
        // Placeholder logic for gradient calculation (requires matmul)
        // let grad_a = matmul(upstream_grad, &self.input_b.transpose());
        // let grad_b = matmul(&self.input_a.transpose(), upstream_grad);
        
        // Placeholder accumulation logic
        if let Some(input_a_rc) = self.input_a_ref.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                println!("MatmulBackward: Would accumulate grad for A");
                // if let Some(ref mut grad) = input_a_td.grad { *grad += &grad_a; }
                // else { input_a_td.grad = Some(grad_a); }
            }
        }
        if let Some(input_b_rc) = self.input_b_ref.upgrade() {
            let mut input_b_td = input_b_rc.borrow_mut();
            if input_b_td.requires_grad {
                 println!("MatmulBackward: Would accumulate grad for B");
                // if let Some(ref mut grad) = input_b_td.grad { *grad += &grad_b; }
                // else { input_b_td.grad = Some(grad_b); }
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_a_ref.clone(), self.input_b_ref.clone()]
    }
}

// Matmul forward function or method implementation needed here or in tensor.rs

// ... Tests ...
#[cfg(test)]
mod tests {
    // ... tests remain the same for now ...
     use crate::Tensor;
    use num_traits::Zero;

    fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }
    #[test] fn test_matmul_propagate_requires_grad() { 
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let _ = (&a, &b);
     }
    #[test] #[should_panic] fn test_matmul_dimension_mismatch() { 
        let a = Tensor::<f32>::new(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::<f32>::new(vec![3.0, 4.0], vec![1, 2]);
         let _ = (&a, &b);
     }
    #[test] #[should_panic] fn test_matmul_first_arg_not_2d() { 
        let a = Tensor::<f32>::new(vec![1.0, 2.0], vec![2]); 
        let b = Tensor::<f32>::new(vec![3.0, 4.0], vec![2, 1]);
        let _ = (&a, &b);
     }
    #[test] #[should_panic] fn test_matmul_second_arg_not_2d() { 
        let a = Tensor::<f32>::new(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::<f32>::new(vec![3.0, 4.0], vec![2]); 
        let _ = (&a, &b);
     }
    #[test] fn test_matmul_2x2() { /* Placeholder */ }
    #[test] fn test_matmul_2x3_3x2() { /* Placeholder */ }
} 