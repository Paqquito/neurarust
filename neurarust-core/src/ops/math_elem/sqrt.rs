use crate::tensor::Tensor;
use crate::tensor_data::TensorData; // TensorData is pub
use crate::autograd::{self, BackwardOp}; // Import autograd module too
use num_traits::{Float, Zero, One};
use std::fmt::{Debug}; // Import fmt for PhantomData Debug
use std::cell::{RefCell, Ref};
use std::rc::{Rc, Weak};
use std::collections::HashMap;
use std::ops::{AddAssign, Mul};
use std::default::Default;
use std::marker::PhantomData;


/// Represents the element-wise square root operation.
#[derive(Debug)]
pub struct SqrtOp<T> {
    input: Tensor<T>, // Keep input tensor for backward pass
    _phantom: PhantomData<T>,
}

impl<T> SqrtOp<T> 
where 
    // Base requirements for forward: Float for sqrt, Debug for PhantomData
    T: Float + Debug + 'static 
{
    /// Creates a new SqrtOp and performs the forward pass.
    pub fn forward(input: &Tensor<T>) -> Tensor<T> 
    where
        // Bounds needed for autograd setup + accumulate_gradient compatibility
        T: Clone + Zero + One + AddAssign + Default + Copy + Mul<Output = T>
    {
        // Calculate output data safely using Ref::map
        let output_data: Vec<T> = Ref::map(input.borrow_tensor_data(), |td| &td.data)
            .iter().map(|&x| x.sqrt()).collect();
        let output_shape = input.shape();
        let requires_grad = input.requires_grad();

        let output_tensor = Tensor::new(output_data, output_shape);

        if requires_grad {
            output_tensor.set_requires_grad(true);
            // Create Rc directly, it handles the trait object conversion
            let sqrt_op: Rc<dyn BackwardOp<T>> = Rc::new(Self {
                input: input.clone(), 
                _phantom: PhantomData,
            });
            output_tensor.set_grad_fn(Some(sqrt_op));
        }

        output_tensor
    }
}

impl<T> BackwardOp<T> for SqrtOp<T>
where
    // Bounds needed for calculation + accumulate_gradient
    T: Float + Debug + 'static + Mul<Output = T> + AddAssign + Default + Zero + One + Clone + Copy, 
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        
        // 1. Calculate 1 / (2 * sqrt(input))
        let two = T::one() + T::one(); 
        let input_data_ref = Ref::map(self.input.borrow_tensor_data(), |td| &td.data);
        let sqrt_input_data: Vec<T> = input_data_ref.iter().map(|&x| x.sqrt()).collect();
        let grad_factor_data: Vec<T> = sqrt_input_data.iter().map(|&sqrt_x| {
            if sqrt_x == T::zero() { T::zero() } else { T::one() / (two * sqrt_x) }
        }).collect();
        
        let grad_factor = Tensor::new(grad_factor_data, self.input.shape());
        
        // 2. Multiply by upstream gradient element-wise
        let upstream_data_ref = Ref::map(upstream_grad.borrow_tensor_data(), |td| &td.data);
        let grad_factor_ref = Ref::map(grad_factor.borrow_tensor_data(), |td| &td.data);
        let local_grad_data: Vec<T> = upstream_data_ref.iter()
            .zip(grad_factor_ref.iter())
            .map(|(&up, &fact)| up * fact) // Requires T: Mul + Copy
            .collect();
        let local_grad = Tensor::new(local_grad_data, self.input.shape());

        // 3. Accumulate gradient using the helper function
        let input_weak_ref = self.input.get_weak_ref();
        autograd::accumulate_gradient(gradients, &input_weak_ref, local_grad);
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input.get_weak_ref()]
    }
}

// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    // Helper for float comparison
    fn assert_approx_eq_float<T: Float + Debug>(calc: T, exp: T, tolerance: T) {
        assert!((calc - exp).abs() <= tolerance, "Assertion failed: {:?} != {:?} within tolerance {:?}", calc, exp, tolerance);
    }
    
    // Re-add forward tests if they were removed accidentally
    #[test]
    fn test_sqrt_forward() {
        let data = vec![1.0f64, 4.0, 9.0, 16.0];
        let shape = vec![2, 2];
        let t1 = Tensor::new(data.clone(), shape.clone());
        
        let t_sqrt = SqrtOp::forward(&t1);
        
        let expected_data = vec![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(Ref::map(t_sqrt.borrow_tensor_data(), |d| &d.data).to_vec(), expected_data);
        assert_eq!(t_sqrt.shape(), shape);
        assert!(!t_sqrt.requires_grad()); 
    }
    
    #[test]
    fn test_sqrt_forward_requires_grad() {
        let data = vec![1.0f64, 4.0, 9.0, 16.0];
        let shape = vec![2, 2];
        let t1 = Tensor::new(data.clone(), shape.clone());
        t1.set_requires_grad(true);
        
        let t_sqrt = SqrtOp::forward(&t1);
        
        let expected_data = vec![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(Ref::map(t_sqrt.borrow_tensor_data(), |d| &d.data).to_vec(), expected_data);
        assert_eq!(t_sqrt.shape(), shape);
        assert!(t_sqrt.requires_grad());
        assert!(t_sqrt.grad_fn().is_some()); 
    }

    #[test]
    fn test_sqrt_backward_simple() {
        type TestType = f64;
        let t1 = Tensor::new(vec![4.0 as TestType, 9.0], vec![2]);
        t1.set_requires_grad(true); 
        let t_sqrt = SqrtOp::forward(&t1);
        
        let upstream_grad = Tensor::new(vec![1.0 as TestType, 1.0], vec![2]);
        
        let mut gradients = HashMap::new();
        let sqrt_op = t_sqrt.grad_fn().unwrap(); 
        sqrt_op.backward(&upstream_grad, &mut gradients);
        
        let expected_grad_data = vec![0.25, 1.0 / 6.0];
        
        // Get the pointer key used by accumulate_gradient
        let t1_weak_ref = t1.get_weak_ref();
        let t1_rc = t1_weak_ref.upgrade().expect("Failed to upgrade weak ref in test");
        let t1_ptr = Rc::as_ptr(&t1_rc);
        
        assert!(gradients.contains_key(&t1_ptr));
        let calculated_grad = gradients.get(&t1_ptr).unwrap();
        
        assert_eq!(calculated_grad.shape(), t1.shape());
        Ref::map(calculated_grad.borrow_tensor_data(), |d| &d.data).iter().zip(expected_grad_data.iter()).for_each(|(&calc, &exp)| {
            assert_approx_eq_float(calc, exp, 1e-9 as TestType);
        });
    }
    
    #[test]
    fn test_sqrt_backward_chain() {
        type TestType = f64;
        let t1 = Tensor::new(vec![4.0 as TestType], vec![1]);
        t1.set_requires_grad(true); 
        let t_sqrt = SqrtOp::forward(&t1); // sqrt(4) = 2
        
        // Simplify the operation: Use clone instead of add
        // If this passes, the issue is likely in AddBackward or its interaction
        let t_final = t_sqrt.clone(); 
        
        t_final.backward(None); 
        
        // Expected gradient: d(clone)/d(t_sqrt) * d(t_sqrt)/d(t1)
        // d(clone)/d(t_sqrt) = 1 (assuming clone op correctly implemented for autograd or doesn't interfere)
        // d(sqrt)/d(t1) = 1 / (2 * sqrt(t1)) = 1 / (2 * 2) = 0.25
        // Total gradient = 1 * 0.25 = 0.25
        let grad = t1.grad().unwrap();
        assert_approx_eq_float(Ref::map(grad.borrow_tensor_data(), |d| &d.data)[0], 0.25, 1e-9 as TestType);
    }

    #[test]
    fn test_sqrt_propagate_requires_grad() {
        let t1 = Tensor::new(vec![4.0f64], vec![1]);
        t1.set_requires_grad(true); 
        let t2 = Tensor::new(vec![9.0f64], vec![1]); 
        
        let t_sqrt1 = SqrtOp::forward(&t1);
        let t_sqrt2 = SqrtOp::forward(&t2);
        
        assert!(t_sqrt1.requires_grad());
        assert!(t_sqrt1.grad_fn().is_some());
        assert!(!t_sqrt2.requires_grad());
        assert!(t_sqrt2.grad_fn().is_none());
    }
} 