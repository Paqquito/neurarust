use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Mul, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

// --- Forward Operation --- 

/// Implements element-wise multiplication (Hadamard product) for two Tensors.
impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static,
{
    type Output = Tensor<T>;

    fn mul(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for element-wise multiplication.");

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let result_data: Vec<T> = self_td.data.iter()
            .zip(other_td.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, self_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = MulBackward {
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

struct MulBackward<T> {
    input_a_val: Tensor<T>,
    input_b_val: Tensor<T>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for MulBackward<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // NOTE: This backward pass does not yet support broadcasting.
        // grad_a and grad_b are calculated assuming shapes match.
        // reduce_gradient would be needed here once broadcasting is added to Mul forward.
        let grad_a = upstream_grad * &self.input_b_val;
        let grad_b = upstream_grad * &self.input_a_val; 

        // Accumulate gradient for Input A
        if let Some(input_a_rc) = self.input_a_ref.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                if let Some(existing_grad_tensor) = input_a_td.grad.as_mut() {
                    let mut existing_data = existing_grad_tensor.borrow_tensor_data_mut();
                    let new_grad_data = grad_a.borrow_tensor_data();
                    assert_eq!(existing_data.data.len(), new_grad_data.data.len());
                    for (existing, new) in existing_data.data.iter_mut().zip(new_grad_data.data.iter()) {
                        *existing += *new;
                    }
                } else {
                    // Use Tensor::new for initial grad
                    input_a_td.grad = Some(Tensor::new(grad_a.data(), grad_a.shape()));
                }
            }
        } 

        // Accumulate gradient for Input B
        if let Some(input_b_rc) = self.input_b_ref.upgrade() {
            let mut input_b_td = input_b_rc.borrow_mut();
            if input_b_td.requires_grad {
                if let Some(existing_grad_tensor) = input_b_td.grad.as_mut() {
                   let mut existing_data = existing_grad_tensor.borrow_tensor_data_mut();
                   let new_grad_data = grad_b.borrow_tensor_data();
                   assert_eq!(existing_data.data.len(), new_grad_data.data.len());
                   for (existing, new) in existing_data.data.iter_mut().zip(new_grad_data.data.iter()) {
                       *existing += *new;
                   }
                } else {
                    // Use Tensor::new for initial grad
                    input_b_td.grad = Some(Tensor::new(grad_b.data(), grad_b.shape()));
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
    fn test_mul_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![5_i32, 12, 21, 32];
        let expected_shape = vec![2, 2];
        let result = &t1 * &t2;
        
        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic]
    fn test_mul_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6], vec![1, 2]);
        let _result = &t1 * &t2;
    }

    #[test]
    fn test_mul_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let res = &t1 * &t2;
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = &t3 * &t1; // t3 requires grad
        assert!(res2.requires_grad());

        let res3 = &t2 * &t3; // both require grad
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_mul_backward() {
        let a = create_test_tensor_with_grad::<f32>(vec![2.0, 3.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0], vec![2]);

        let c = &a * &b;
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
            let expected_grad_a_data = vec![4.0, -5.0];
            let expected_grad_b_data = vec![2.0, -3.0]; 
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
        let expected_accum_grad_a_data = vec![6.0, -2.5]; 
        let expected_accum_grad_b_data = vec![3.0, -1.5];
        let expected_accum_shape = vec![2];

        assert_eq!(grad_a_accum.as_ref().unwrap().data(), expected_accum_grad_a_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().data(), expected_accum_grad_b_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad B shape mismatch");
    }
} 