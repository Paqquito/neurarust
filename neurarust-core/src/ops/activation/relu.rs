use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Mul, AddAssign};
use num_traits::{Zero, One}; // Remove PartialOrd from here
use std::cmp::PartialOrd; // Import from std::cmp
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::default::Default;
use std::ops::Neg;
use std::iter::Sum;
use std::collections::HashMap;

// --- Forward Operation --- 

impl<T> Tensor<T> {
    /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
    /// ReLU(x) = max(0, x)
    pub fn relu(&self) -> Tensor<T>
    where
        T: Zero + PartialOrd + Copy + Clone + 'static + AddAssign + One + Mul<Output=T> + Debug + Default + Sum + Neg<Output=T>,
    {
        let input_td = self.borrow_tensor_data();
        
        // Apply ReLU function: max(0, x)
        let result_data: Vec<T> = input_td.data.iter().map(|&x| {
            if x > T::zero() { x } else { T::zero() }
        }).collect();
        
        let result_shape = input_td.shape.clone();
        drop(input_td); // Drop borrow before potentially cloning self

        let requires_grad = self.requires_grad();
        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = ReluBackward {
                input: self.clone(), // Store original input for gradient calculation
                _phantom: PhantomData,
            };
            result.data.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

#[derive(Debug)]
struct ReluBackward<T> {
    input: Tensor<T>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for ReluBackward<T>
where
    T: PartialOrd + Zero + Clone + Copy + Mul<Output = T> + Debug + AddAssign + One + Sum + 'static + Default + Neg<Output=T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input.get_weak_ref().upgrade() {
            if input_rc.borrow().requires_grad {
                // grad = upstream_grad * (input > 0)
                let input_data = self.input.data();
                let mask_data: Vec<T> = input_data.iter().map(|&x| if x > T::zero() { T::one() } else { T::zero() }).collect();
                let mask_tensor = Tensor::new(mask_data, self.input.shape());
                
                let local_gradient = upstream_grad * &mask_tensor;

                // Use the centralized version
                crate::autograd::accumulate_gradient(gradients, &self.input.get_weak_ref(), local_gradient);
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input.get_weak_ref()]
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::cmp::PartialOrd; // Import PartialOrd here too
    use std::ops::{AddAssign, Mul};
    use std::iter::Sum; // Keep for helper if needed

    // Helper function with necessary bounds for ReLU tests
    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where T: Zero + PartialOrd + One + Mul<Output=T> + AddAssign + Copy + Clone + 'static + std::fmt::Debug + PartialEq + Sum
    {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_relu_forward() {
        let data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let shape = vec![5];
        let t = Tensor::new(data, shape.clone());
        let result = t.relu();
        
        let expected_data = vec![0.0_f32, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), shape);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_relu_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad(vec![-1.0_f32, 1.0], vec![2]);
        let result = t1.relu();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let t2 = Tensor::new(vec![3.0_f32], vec![1]);
        let result2 = t2.relu();
        assert!(!result2.requires_grad());
        assert!(result2.grad_fn().is_none());
    }

    #[test]
    fn test_relu_backward() {
        let t1 = create_test_tensor_with_grad(vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let result = t1.relu(); // result = [0.0, 0.0, 0.0, 1.0, 2.0]
        
        // Sum the result to get a scalar for backward()
        let loss = result.sum(); 

        assert!(t1.grad().is_none());
        loss.backward(None); // Upstream grad is implicitly 1.0 for sum, then flows back

        let grad_t1 = t1.grad();
        assert!(grad_t1.is_some());
        let grad_t1_tensor = grad_t1.unwrap();
        
        // Expected grad: upstream(1.0) * mask([0, 0, 0, 1, 1]) = [0.0, 0.0, 0.0, 1.0, 1.0]
        let expected_grad_data = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(grad_t1_tensor.data().to_vec(), expected_grad_data);
        assert_eq!(grad_t1_tensor.shape(), vec![5]);
    }
    
     #[test]
    fn test_relu_backward_chain() {
        // Test ReLU within a chain: loss = sum(relu(x * 2))
        let x = create_test_tensor_with_grad(vec![-1.0_f32, 1.0, 2.0], vec![3]);
        let two = Tensor::new(vec![2.0_f32, 2.0, 2.0], vec![3]); 
        
        let y = &x * &two;  // y = [-2.0, 2.0, 4.0]
        let z = y.relu(); // z = [0.0, 2.0, 4.0]
        let loss = z.sum();  // loss = 6.0

        assert!(x.grad().is_none());
        loss.backward(None);

        // Gradients:
        // dLoss/dz = [1, 1, 1] (from sum)
        // dLoss/dy = dLoss/dz * d(relu)/dy = [1, 1, 1] * [0, 1, 1] = [0, 1, 1]
        // dLoss/dx = dLoss/dy * dy/dx = [0, 1, 1] * [2, 2, 2] = [0, 2, 2]
        let expected_grad_x = vec![0.0_f32, 2.0, 2.0];
        assert_eq!(x.grad().unwrap().data().to_vec(), expected_grad_x);
    }
} 