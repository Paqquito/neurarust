use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use std::ops::{Sub, Neg, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

// --- Forward Operation --- 

/// Implements element-wise subtraction for two Tensors.
impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Sub<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static,
{
    type Output = Tensor<T>;

    fn sub(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise subtraction.");

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = SubBackward {
                input_a: self.get_weak_ref(),
                input_b: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct SubBackward<T> {
    input_a: Weak<RefCell<crate::tensor::TensorData<T>>>,
    input_b: Weak<RefCell<crate::tensor::TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // Accumulate gradient for Input A (dL/dA = upstream_grad)
        if let Some(input_a_rc) = self.input_a.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                if let Some(existing_grad_a) = input_a_td.grad.as_mut() {
                    *existing_grad_a += upstream_grad;
                } else {
                    input_a_td.grad = Some(Tensor::new(upstream_grad.data(), upstream_grad.shape()));
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input A in SubBackward.");
        }

        // Accumulate gradient for Input B (dL/dB = -upstream_grad)
        if let Some(input_b_rc) = self.input_b.upgrade() {
            let mut input_b_td = input_b_rc.borrow_mut();
            if input_b_td.requires_grad {
                let neg_upstream_grad = -upstream_grad;
                if let Some(existing_grad_b) = input_b_td.grad.as_mut() {
                    *existing_grad_b += &neg_upstream_grad; 
                } else {
                    input_b_td.grad = Some(neg_upstream_grad);
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input B in SubBackward.");
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<crate::tensor::TensorData<T>>>> {
        vec![self.input_a.clone(), self.input_b.clone()]
    }
}

// --- Tests --- 

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::Zero;

    // Helpers might need to be moved to a common test utils module later
    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = create_test_tensor(vec![6_i32, 8, 10, 12], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![1_i32, 2, 3, 4];
        let expected_shape = vec![2, 2];
        let result = &t1 - &t2;

        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 - &t2;
    }

    #[test]
    fn test_sub_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let res = &t2 - &t1;
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = &t1 - &t3; // t3 requires grad
        assert!(res2.requires_grad()); 

        let res3 = &t2 - &t3; // both require grad
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_sub_backward() {
        let a = create_test_tensor_with_grad::<f32>(vec![10.0, 20.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![3.0, 8.0], vec![2]);

        let c = &a - &b;
        assert!(c.requires_grad());
        let grad_fn_option = c.0.borrow().grad_fn.clone();
        assert!(grad_fn_option.is_some());
        let grad_fn = grad_fn_option.unwrap();

        assert!(a.borrow_grad().is_none());
        assert!(b.borrow_grad().is_none());

        let upstream_grad = Tensor::new(vec![1.0, -1.0], vec![2]);

        grad_fn.backward(&upstream_grad);

        {
            let grad_a = a.borrow_grad();
            let grad_b = b.borrow_grad();
            assert!(grad_a.is_some());
            assert!(grad_b.is_some());
            let expected_grad_a_data = vec![1.0, -1.0];
            let expected_grad_b_data = vec![-1.0, 1.0]; 
            let expected_shape = vec![2];
            assert_eq!(grad_a.as_ref().unwrap().data(), expected_grad_a_data, "Grad A data mismatch");
            assert_eq!(grad_a.as_ref().unwrap().shape(), expected_shape, "Grad A shape mismatch");
            assert_eq!(grad_b.as_ref().unwrap().data(), expected_grad_b_data, "Grad B data mismatch");
            assert_eq!(grad_b.as_ref().unwrap().shape(), expected_shape, "Grad B shape mismatch");
        }

        let upstream_grad_2 = Tensor::new(vec![0.5, 0.5], vec![2]);
        grad_fn.backward(&upstream_grad_2);

        let grad_a_accum = a.borrow_grad();
        let grad_b_accum = b.borrow_grad();
        let expected_accum_grad_a_data = vec![1.5, -0.5]; 
        let expected_accum_grad_b_data = vec![-1.5, 0.5];
        let expected_accum_shape = vec![2];

        assert_eq!(grad_a_accum.as_ref().unwrap().data(), expected_accum_grad_a_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().data(), expected_accum_grad_b_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad B shape mismatch");
    }
} 