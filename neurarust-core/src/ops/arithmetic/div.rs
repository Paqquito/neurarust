use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Div, Mul, Neg, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

// --- Forward Operation --- 

/// Implements element-wise division for two Tensors.
impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static, // Add all needed for backward
{
    type Output = Tensor<T>;

    fn div(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise division.");

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a / b) 
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = DivBackward {
                input_a_val: self.clone(),
                input_b_val: other.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct DivBackward<T> {
    input_a_val: Tensor<T>,
    input_b_val: Tensor<T>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for DivBackward<T>
where
    // Requires Div, Mul, Neg, AddAssign, Copy, Clone, 'static
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // Calculate gradients: dA = dC * (1/B), dB = dC * (-A / B^2)

        // Calculate grad_a = upstream_grad / B
        let grad_a = upstream_grad / &self.input_b_val;

        // Calculate grad_b = upstream_grad * (-A / B^2)
        let b_squared = &self.input_b_val * &self.input_b_val;
        let a_div_b_squared = &self.input_a_val / &b_squared;
        let neg_a_div_b_squared = -&a_div_b_squared; // Apply Neg to a reference
        let grad_b = upstream_grad * &neg_a_div_b_squared;

        // Accumulate gradients
        if let Some(input_a_rc) = self.input_a_ref.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                if let Some(ref mut grad) = input_a_td.grad {
                    *grad += &grad_a; // Accumulate gradient for A
                } else {
                    input_a_td.grad = Some(grad_a);
                }
            }
        }
        if let Some(input_b_rc) = self.input_b_ref.upgrade() {
             let mut input_b_td = input_b_rc.borrow_mut();
             if input_b_td.requires_grad {
                 if let Some(ref mut grad) = input_b_td.grad {
                    *grad += &grad_b; // Accumulate gradient for B
                 } else {
                    input_b_td.grad = Some(grad_b);
                 }
             }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_a_ref.clone(), self.input_b_ref.clone()]
    }
}

// --- Tests --- 

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::Zero;

    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
     fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0_f32, 12.0, 21.0, 32.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let expected_data = vec![2.0_f32, 2.0, 3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = &t1 / &t2;

        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic] 
    fn test_div_tensors_int_div_by_zero() {
        let t1 = create_test_tensor(vec![10_i32], vec![1]);
        let t2 = create_test_tensor(vec![0_i32], vec![1]);
        let _result = &t1 / &t2;
    }

    #[test]
    #[should_panic]
    fn test_div_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0_f32, 6.0], vec![1, 2]);
        let _result = &t1 / &t2;
    }

    #[test]
    fn test_div_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor::<f32>(vec![2.0], vec![1]);
        let res = &t1 / &t2;
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = &t2 / &t3; // t3 requires grad
        assert!(res2.requires_grad());

        let res3 = &t1 / &t3; // both require grad
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_div_backward() {
        let a = create_test_tensor_with_grad(vec![6.0_f32], vec![1]);
        let b = create_test_tensor_with_grad(vec![2.0_f32], vec![1]);
        let c = &a / &b; // c = 3.0

        assert!(c.requires_grad());
        assert!(c.grad_fn().is_some());

        c.backward(); // Initial gradient (dC/dC) = 1.0

        // Expected gradients:
        // grad_a = dC/dA = 1/B = 1/2 = 0.5
        // grad_b = dC/dB = -A/B^2 = -6 / (2*2) = -6 / 4 = -1.5
        let expected_grad_a = vec![0.5_f32];
        let expected_grad_b = vec![-1.5_f32];

        assert_eq!(a.grad().unwrap().data(), expected_grad_a, "Gradient for A mismatch");
        assert_eq!(b.grad().unwrap().data(), expected_grad_b, "Gradient for B mismatch");
    }
} 