use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{index_to_coord};
use std::ops::{Neg, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use std::collections::HashMap;
use num_traits::Zero;
use crate::error::NeuraRustError;

// --- Forward Operation --- 

/// Performs unary negation for a Tensor.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn neg<T>(a: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static + Debug + Zero,
{
    let a_td = a.borrow_tensor_data();
    let a_shape = a_td.shape.clone();
    let a_strides = a_td.strides.clone();
    let numel = a_shape.iter().product();
    
    // Create result data vec
    let mut result_data = Vec::with_capacity(numel);
    
    // Iterate using logical indices and strides
    for i in 0..numel {
        // Note: Since output shape == input shape, we can use input strides/shape 
        // for index_to_coord to get the logical multi-index.
        let multi_index = index_to_coord(i, &a_strides, &a_shape);
        let offset = a_td.get_offset(&multi_index); // Calculate offset using strides
        result_data.push(-a_td.data[offset]); // Access data via offset and negate
    }

    // No need to clone shape again, already cloned
    let result_shape = a_shape; 
    
    drop(a_td);
    
    // Use `?` for Tensor::new error
    let result = Tensor::new(result_data, result_shape)?;

    let requires_grad = a.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = NegBackward {
            input_ref: a.get_weak_ref(),
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }
    Ok(result)
}

// --- std::ops::Neg implementation (calls the fallible function) ---
impl<'a, T> Neg for &'a Tensor<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + 'static + Debug + Zero,
{
    type Output = Tensor<T>;

    /// Panics if negation fails (highly unlikely, only tensor creation could fail).
    /// Use `neurarust::ops::arithmetic::neg` for fallible negation.
    fn neg(self) -> Self::Output {
        neg(self).unwrap_or_else(|e| panic!("Tensor negation failed: {:?}", e))
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
            .and_modify(|existing_grad| { 
                assert_eq!(existing_grad.shape(), local_gradient.shape());
                *existing_grad += &local_gradient; 
            })
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
                // Use fallible neg, expect success in backward pass context
                let local_gradient = neg(upstream_grad)
                    .expect("Internal error: Backward negation failed");
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
    use super::*; // Import the new `neg` function
    use crate::Tensor;
    
    use num_traits::{Zero, One};
    use std::ops::AddAssign;
    use std::iter::Sum as IterSum;
    use std::collections::HashMap;
    use std::rc::Rc;
    

    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Neg<Output = T> + Clone + Debug + PartialEq + Zero + AddAssign + One + Copy + IterSum + 'static
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
     fn create_test_tensor_with_grad<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Neg<Output = T> + Clone + Debug + PartialEq + Zero + AddAssign + One + Copy + IterSum + 'static
    {
        Tensor::new_with_grad(data, shape).expect("Test tensor_with_grad creation failed")
    }

    #[test]
    fn test_neg_tensor() {
        let t1 = create_test_tensor(vec![1.0_f32, -2.0, 3.0, -4.0], vec![2, 2]);
        let expected_data = vec![-1.0_f32, 2.0, -3.0, 4.0];
        let expected_shape = vec![2, 2];
        
        // Use fallible neg function
        let result = neg(&t1);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();

        assert_eq!(res_tensor.data().to_vec(), expected_data);
        assert_eq!(res_tensor.shape(), expected_shape);
        assert!(!res_tensor.requires_grad());
    }

    #[test]
    fn test_neg_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        // Use fallible neg
        let res = neg(&t1).unwrap();
        assert!(res.requires_grad());
        assert!(res.grad_fn().is_some());

        let t2 = create_test_tensor::<f32>(vec![2.0], vec![1]);
        let res2 = neg(&t2).unwrap();
        assert!(!res2.requires_grad());
        assert!(res2.grad_fn().is_none());
    }

    #[test]
    fn test_neg_backward() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![2.0, -3.0], vec![2]);
        
        // Use fallible neg
        let loss = neg(&t1).expect("Neg failed in backward test setup");

        let grad_fn = loss.grad_fn().clone().expect("Grad fn missing");
        // Create upstream gradient using fallible new
        let upstream_grad = Tensor::new(vec![1.0, 1.0], vec![2])
            .expect("Failed to create upstream grad"); 
        let mut gradients = HashMap::new();
        gradients.insert(Rc::as_ptr(&loss.data), upstream_grad.clone());

        grad_fn.backward(&upstream_grad, &mut gradients);

        let grad_t1 = gradients.get(&Rc::as_ptr(&t1.data)).expect("Grad t1 missing");
        assert_eq!(grad_t1.data().to_vec(), vec![-1.0, -1.0]);
        assert_eq!(grad_t1.shape(), vec![2]);

        let upstream_grad2 = Tensor::new(vec![0.5, -0.5], vec![2])
            .expect("Failed to create upstream grad 2");
        grad_fn.backward(&upstream_grad2, &mut gradients);
        let grad_t1_accum = gradients.get(&Rc::as_ptr(&t1.data)).expect("Accum Grad t1 missing");
        assert_eq!(grad_t1_accum.data().to_vec(), vec![-1.5, -0.5]); 
    }

    #[test]
    fn test_neg_tensor_backward() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![2.0, -3.0], vec![2]);
        t1.zero_grad();
        // Use fallible neg
        let loss = neg(&t1).expect("Neg failed in backward test setup"); 

        // Create upstream gradient using fallible new
        let upstream = Tensor::new(vec![1.0, 1.0], vec![2])
            .expect("Failed to create upstream tensor");
        loss.backward(Some(&upstream)); 

        let grad_t1 = t1.grad().expect("Grad t1 missing after Tensor::backward");
        assert_eq!(grad_t1.data().to_vec(), vec![-1.0, -1.0]);
        assert_eq!(grad_t1.shape(), vec![2]);
    }

    #[test]
    fn test_neg_backward_no_grad() {
        let t1 = create_test_tensor::<f32>(vec![2.0_f32, -3.0], vec![2]);
        // Use fallible neg
        let t2 = neg(&t1).expect("Neg failed for non-grad tensor");

        assert!(!t2.requires_grad());
        assert!(t2.borrow_tensor_data().grad_fn.is_none());
    }
} 