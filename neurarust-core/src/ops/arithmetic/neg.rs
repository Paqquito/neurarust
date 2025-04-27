use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Neg, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

// --- Forward Operation --- 

/// Implements unary negation for a Tensor.
impl<'a, T> Neg for &'a Tensor<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static, // Add all needed for backward
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let self_td = self.borrow_tensor_data();
        let result_data: Vec<T> = self_td.data.iter().map(|&x| -x).collect();
        let result_shape = self_td.shape.clone();
        
        drop(self_td);
        
        let requires_grad = self.requires_grad();
        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = NegBackward {
                input_ref: self.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct NegBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for NegBackward<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static, 
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            let mut input_td = input_rc.borrow_mut();
            if input_td.requires_grad {
                let grad_neg = -upstream_grad; // dA = dC * (-1)
                if let Some(ref mut grad) = input_td.grad {
                    *grad += &grad_neg;
                } else {
                    input_td.grad = Some(grad_neg);
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
    use crate::tensor_data::TensorData;
    use num_traits::{Zero, One};
    use std::ops::AddAssign;
    use std::iter::Sum as IterSum;

    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq + Zero + AddAssign + One + Copy + IterSum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
     fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero + AddAssign + One + Copy + IterSum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_neg_tensor() {
        let t1 = create_test_tensor(vec![1.0_f32, -2.0, 3.0, -4.0], vec![2, 2]);
        let expected_data = vec![-1.0_f32, 2.0, -3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = -&t1;

        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_neg_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        let res = -&t1;
        assert!(res.requires_grad());
        assert!(res.grad_fn().is_some());

        let t2 = create_test_tensor::<f32>(vec![2.0], vec![1]);
        let res2 = -&t2;
        assert!(!res2.requires_grad());
        assert!(res2.grad_fn().is_none());
    }

    #[test]
    fn test_neg_backward() {
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0, 3.0], vec![3]);
        let t2 = create_test_tensor_with_grad(vec![4.0_f32, 5.0, 6.0], vec![3]);

        let t3 = &t1 + &t2;
        let t4 = -&t3;

        let loss = t4.sum();

        assert!(loss.requires_grad());
        assert_eq!(loss.data(), vec![-21.0_f32]);
        assert!(loss.grad_fn().is_some());

        assert!(t1.grad().is_none());
        assert!(t2.grad().is_none());
        assert!(t3.grad().is_none());
        assert!(t4.grad().is_none());

        loss.backward();

        let expected_grad = vec![-1.0_f32, -1.0, -1.0];
        let expected_grad_t4 = vec![1.0_f32, 1.0, 1.0];

        assert_eq!(loss.grad().expect("Grad for loss missing").data(), vec![1.0_f32]);
        assert_eq!(t4.grad().expect("Grad for t4 missing").data(), expected_grad_t4);
        assert_eq!(t3.grad().expect("Grad for t3 missing").data(), expected_grad);
        assert_eq!(t1.grad().expect("Grad for t1 missing").data(), expected_grad);
        assert_eq!(t2.grad().expect("Grad for t2 missing").data(), expected_grad);
    }
} 