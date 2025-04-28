use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Neg, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::collections::HashMap;
use num_traits::Zero;

// --- Forward Operation --- 

/// Implements unary negation for a Tensor.
impl<'a, T> Neg for &'a Tensor<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static + Debug + Zero,
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
            result.borrow_tensor_data_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

#[derive(Debug)]
struct NegBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

// Helper function to accumulate gradient
fn accumulate_gradient<T>(
    gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>,
    input_weak_ref: &Weak<RefCell<TensorData<T>>>,
    local_gradient: Tensor<T>,
)
where
    T: AddAssign + Clone + Debug + Zero + Copy + 'static,
{
    if let Some(input_rc) = input_weak_ref.upgrade() {
        let input_ptr = Rc::as_ptr(&input_rc);
        gradients.entry(input_ptr)
            .and_modify(|existing_grad| { *existing_grad += &local_gradient; })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for NegBackward<T>
where
    T: Neg<Output = T> + AddAssign + Clone + Debug + Zero + Copy + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                let local_gradient = -upstream_grad;
                accumulate_gradient(gradients, &self.input_ref, local_gradient);
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
    use std::iter::Sum as IterSum;
    use std::collections::HashMap;
    use std::rc::Rc;

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

        assert_eq!(result.data().to_vec(), expected_data);
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
        let t1 = create_test_tensor_with_grad::<f32>(vec![2.0, -3.0], vec![2]);
        let loss = -(&t1); // Loss = -t1

        let grad_fn = loss.grad_fn().clone().unwrap();
        let upstream_grad = Tensor::new(vec![1.0, 1.0], vec![2]); 
        let mut gradients = HashMap::new();
        gradients.insert(Rc::as_ptr(&loss.data), upstream_grad.clone());

        // Appel correct à grad_fn.backward
        grad_fn.backward(&upstream_grad, &mut gradients);

        // Récupérer le gradient calculé depuis le HashMap
        let grad_t1 = gradients.get(&Rc::as_ptr(&t1.data)).expect("Grad t1 missing");

        assert_eq!(grad_t1.data().to_vec(), vec![-1.0, -1.0]);
        assert_eq!(grad_t1.shape(), vec![2]);

        // Test accumulation
        let upstream_grad2 = Tensor::new(vec![0.5, -0.5], vec![2]);
        grad_fn.backward(&upstream_grad2, &mut gradients);
        let grad_t1_accum = gradients.get(&Rc::as_ptr(&t1.data)).expect("Accum Grad t1 missing");
        assert_eq!(grad_t1_accum.data().to_vec(), vec![-1.5, -0.5]); 
    }

    #[test]
    fn test_neg_tensor_backward() { // Assurer que ce test est correct
        let t1 = create_test_tensor_with_grad::<f32>(vec![2.0, -3.0], vec![2]);
        t1.zero_grad();
        let loss = -(&t1); 

        // Appel correct à Tensor::backward avec un gradient amont
        loss.backward(Some(&Tensor::new(vec![1.0, 1.0], vec![2]))); 

        // Vérifier t1.grad() directement
        let grad_t1 = t1.grad().expect("Grad t1 missing after Tensor::backward");
        assert_eq!(grad_t1.data().to_vec(), vec![-1.0, -1.0]);
        assert_eq!(grad_t1.shape(), vec![2]);
    }

    #[test]
    fn test_neg_backward_no_grad() {
        let t1 = Tensor::new(vec![2.0_f32, -3.0], vec![2]);
        t1.set_requires_grad(false); // Ne requiert pas de gradient
        let t2 = -&t1;

        assert!(!t2.requires_grad()); // Ne devrait pas requérir de gradient
        assert!(t2.borrow_tensor_data().grad_fn.is_none()); // Ne devrait pas avoir de grad_fn

        // Essayer d'appeler backward devrait paniquer ou échouer, 
        // mais ici on vérifie juste qu'aucun grad_fn n'est créé.
        // Si on forçait un backward (ce qui ne devrait pas arriver), on ne devrait pas avoir de grad dans t1
    }
} 