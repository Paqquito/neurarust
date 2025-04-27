use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Add, AddAssign};
use std::iter::Sum as IterSum; // To avoid ambiguity with our Sum trait/struct
use num_traits::{Zero, One}; // Need Zero for sum init, One for grad init, AddAssign for grad accum
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

// --- Forward Operation --- 

// We could implement this as a method on Tensor<T>
impl<T> Tensor<T> {
    /// Calculates the sum of all elements in the tensor, returning a scalar tensor.
    pub fn sum(&self) -> Tensor<T>
    where
        T: Add<Output = T> + IterSum + Zero + Copy + Clone + 'static + AddAssign + One,
    {
        let input_td = self.borrow_tensor_data();
        let sum_val = input_td.data.iter().copied().sum::<T>();
        let result_shape = vec![1]; // Scalar output
        let result_data = vec![sum_val];
        
        let input_shape_clone = input_td.shape.clone(); // Needed for backward
        let input_numel = input_td.numel(); // Needed for backward
        drop(input_td);

        let requires_grad = self.requires_grad();
        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = SumBackward {
                input_ref: self.get_weak_ref(),
                input_shape: input_shape_clone,
                input_numel,
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct SumBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    input_shape: Vec<usize>,
    input_numel: usize,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SumBackward<T>
where
    T: Clone + Copy + One + AddAssign + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        upstream_grad.set_requires_grad(false); // Ensure upstream grad doesn't require grad
        
        if let Some(input_rc) = self.input_ref.upgrade() {
            let mut input_td = input_rc.borrow_mut();
            if input_td.requires_grad {
                // Gradient of sum is 1 distributed across all input elements,
                // scaled by the upstream gradient.
                assert_eq!(upstream_grad.numel(), 1, "Upstream grad for sum must be scalar.");
                let grad_val = upstream_grad.data()[0]; // Get the scalar gradient value
                
                // Create the local gradient tensor (same shape as input)
                let grad_data = vec![grad_val; self.input_numel];
                let local_grad = Tensor::new(grad_data, self.input_shape.clone());
                // local_grad already has requires_grad=false from Tensor::new

                // Accumulate gradient
                if let Some(existing_grad) = input_td.grad.as_mut() {
                    *existing_grad += &local_grad; // Use AddAssign
                } else {
                    input_td.grad = Some(local_grad); // Assign the new grad (requires_grad=false)
                }
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::AddAssign;

    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq + Zero + AddAssign + One + Copy + std::iter::Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero + AddAssign + One + Copy + std::iter::Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_sum_forward() {
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = t1.sum();
        assert_eq!(result.data(), vec![21.0_f32]);
        assert_eq!(result.shape(), vec![1]);
        assert!(!result.requires_grad());
    }
    
     #[test]
    fn test_sum_forward_scalar() {
        let t1 = create_test_tensor(vec![-5.0_f32], vec![1]);
        let result = t1.sum();
        assert_eq!(result.data(), vec![-5.0_f32]);
        assert_eq!(result.shape(), vec![1]);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_sum_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0], vec![2]);
        let result = t1.sum();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let t2 = create_test_tensor(vec![3.0_f32], vec![1]);
        let result2 = t2.sum();
        assert!(!result2.requires_grad());
        assert!(result2.grad_fn().is_none());
    }

    #[test]
    fn test_sum_backward() {
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t1.sum(); // result = 10.0

        assert!(t1.grad().is_none());
        result.backward(); // Upstream grad is implicitly 1.0

        let grad_t1 = t1.grad();
        assert!(grad_t1.is_some());
        let grad_t1_tensor = grad_t1.unwrap();
        // Expected grad is 1.0 distributed across the original shape
        assert_eq!(grad_t1_tensor.data(), vec![1.0_f32, 1.0, 1.0, 1.0]);
        assert_eq!(grad_t1_tensor.shape(), vec![2, 2]);
    }
    
    #[test]
    fn test_sum_backward_accumulation() {
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0], vec![2]);
        let t2 = create_test_tensor_with_grad(vec![3.0_f32, 4.0], vec![2]);
        
        let sum1 = t1.sum(); // 3.0
        let sum2 = t2.sum(); // 7.0
        let final_sum = &sum1 + &sum2; // 10.0

        final_sum.backward();

        // grad(t1) = grad(final_sum) * d(final_sum)/d(sum1) * d(sum1)/dt1 = 1.0 * 1.0 * [1.0, 1.0]
        assert_eq!(t1.grad().unwrap().data(), vec![1.0_f32, 1.0]);
        // grad(t2) = grad(final_sum) * d(final_sum)/d(sum2) * d(sum2)/dt2 = 1.0 * 1.0 * [1.0, 1.0]
        assert_eq!(t2.grad().unwrap().data(), vec![1.0_f32, 1.0]);
    }
} 