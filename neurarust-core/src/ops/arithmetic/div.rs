use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, coord_to_index_broadcasted, reduce_gradient};
use std::ops::{Div, Mul, Neg, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::collections::HashMap;
use crate::error::NeuraRustError;
use crate::ops::arithmetic::mul::mul;

// --- Forward Operation --- 

/// Performs element-wise division for two Tensors with broadcasting.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn div<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + PartialEq,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    let result_shape = broadcast_shapes(&a_shape, &b_shape)
        .map_err(|_e| NeuraRustError::BroadcastError { 
            shape1: a_shape.clone(), 
            shape2: b_shape.clone(), 
        })?;

    let a_td = a.borrow_tensor_data();
    let b_td = b.borrow_tensor_data();

    let numel_result = result_shape.iter().product();
    let mut result_data = Vec::with_capacity(numel_result);
    let strides_a = calculate_strides(&a_shape);
    let strides_b = calculate_strides(&b_shape);
    let result_strides = calculate_strides(&result_shape);

    for i in 0..numel_result {
        let multi_index = index_to_coord(i, &result_strides, &result_shape);
        let index_a = coord_to_index_broadcasted(&multi_index, &a_shape, &strides_a);
        let index_b = coord_to_index_broadcasted(&multi_index, &b_shape, &strides_b);
        
        let divisor = b_td.data[index_b];
        if divisor == T::zero() {
            return Err(NeuraRustError::DivisionByZero);
        }
        result_data.push(a_td.data[index_a] / divisor); 
    }

    drop(a_td);
    drop(b_td);

    let result = Tensor::new(result_data, result_shape.clone())?;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = DivBackward {
            input_a_shape: a_shape.clone(),
            input_b_shape: b_shape.clone(),
            input_a_ref: a.get_weak_ref(),
            input_b_ref: b.get_weak_ref(),
            input_a_val: a.clone(), 
            input_b_val: b.clone(),
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn))); 
    }
    Ok(result)
}

// --- std::ops::Div implementation (calls the fallible function) ---
impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + PartialEq,
{
    type Output = Tensor<T>;

    /// Panics if division fails (e.g., incompatible shapes, division by zero).
    /// Use `neurarust::ops::arithmetic::div` for fallible division.
    fn div(self, other: &'b Tensor<T>) -> Self::Output {
        div(self, other).unwrap_or_else(|e| panic!("Tensor division failed: {:?}", e))
    }
}

// --- Backward Operation --- 

#[derive(Debug)]
struct DivBackward<T> {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    input_a_val: Tensor<T>,
    input_b_val: Tensor<T>,
    _phantom: PhantomData<T>,
}

// Copier helper
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
                 assert_eq!(existing_grad.shape(), local_gradient.shape(), "Grad shape mismatch");
                *existing_grad += &local_gradient; 
            })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for DivBackward<T>
where
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + PartialEq,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        let needs_grad_a = self.input_a_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        let needs_grad_b = self.input_b_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);

        if needs_grad_a || needs_grad_b {
            let grad_clone = upstream_grad.clone();
            
            if needs_grad_a {
                // Use fallible div, expect success in backward pass context
                let grad_a_unreduced = div(&grad_clone, &self.input_b_val)
                    .expect("Internal error: Backward division failed for grad_a (div by b)"); 
                let grad_a = reduce_gradient(&grad_a_unreduced, &self.input_a_shape); 
                accumulate_gradient(gradients, &self.input_a_ref, grad_a);
            }
            
            if needs_grad_b {
                // Use fallible mul/div, expect success in backward pass context
                let b_squared = mul(&self.input_b_val, &self.input_b_val)
                     .expect("Internal error: Backward division failed (b*b)");
                let a_div_b_squared = div(&self.input_a_val, &b_squared)
                     .expect("Internal error: Backward division failed (a/b^2)");
                // Assuming Neg impl (-) doesn't fail
                let neg_a_div_b_squared = -&a_div_b_squared;         
                let grad_b_unreduced = mul(&grad_clone, &neg_a_div_b_squared)
                     .expect("Internal error: Backward division failed (grad*(-a/b^2))");
                let grad_b = reduce_gradient(&grad_b_unreduced, &self.input_b_shape);
                accumulate_gradient(gradients, &self.input_b_ref, grad_b);
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
    use super::*; // Import the new `div` function
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Div, Mul, Neg, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    
    
    use crate::error::NeuraRustError;

    // Update helpers to handle Result from Tensor::new
    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T>
    where 
        T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + PartialEq
    { 
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    fn create_test_tensor_with_grad<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T>
    where 
        T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + PartialEq
    { 
        let tensor = Tensor::new(data, shape).expect("Test tensor_with_grad creation failed (new)");
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0_f32, 12.0, 21.0, 32.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let expected_data = vec![2.0_f32, 2.0, 3.0, 4.0];
        let expected_shape = vec![2, 2];
        
        // Use fallible div function
        let result = div(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();

        assert_eq!(res_tensor.data().to_vec(), expected_data, "Data mismatch");
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
        assert!(!res_tensor.requires_grad());
    }

    #[test]
    fn test_div_tensors_div_by_zero() {
        let t1 = create_test_tensor(vec![10_i32], vec![1]);
        let t2 = create_test_tensor(vec![0_i32], vec![1]);
        
        let result = div(&t1, &t2);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::DivisionByZero));
    }

    #[test]
    fn test_div_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
        
        let result = div(&t1, &t_non_broadcast);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::BroadcastError { .. }));
    }

    #[test]
    fn test_div_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor::<f32>(vec![2.0], vec![1]);
        let res = div(&t1, &t2).unwrap();
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = div(&t2, &t3).unwrap(); 
        assert!(res2.requires_grad());

        let res3 = div(&t1, &t3).unwrap();
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_div_backward() {
        let a = create_test_tensor_with_grad(vec![6.0_f32], vec![1]);
        let b = create_test_tensor_with_grad(vec![2.0_f32], vec![1]);
        
        let c = div(&a, &b).expect("Div failed in backward test setup");

        assert!(c.requires_grad());
        
        c.backward(None);

        let grad_a = a.grad().expect("Grad A missing");
        let grad_b = b.grad().expect("Grad B missing");

        let expected_grad_a = vec![1.0 / 2.0];
        let expected_grad_b = vec![-6.0 / (2.0*2.0)];

        assert_eq!(grad_a.data().to_vec(), expected_grad_a, "Gradient for A mismatch");
        assert_eq!(grad_a.shape(), vec![1]);
        assert_eq!(grad_b.data().to_vec(), expected_grad_b, "Gradient for B mismatch");
        assert_eq!(grad_b.shape(), vec![1]);
    }
    
    // TODO: Update broadcasting tests to use the new `div` function and check Results.
}