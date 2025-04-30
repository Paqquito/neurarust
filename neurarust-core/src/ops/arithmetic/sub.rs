use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, coord_to_index_broadcasted, reduce_gradient};
use std::ops::{Sub, Neg, AddAssign, Add, SubAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::collections::HashMap;
use crate::error::NeuraRustError;

// --- Forward Operation --- 

/// Performs element-wise subtraction for two Tensors with broadcasting.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn sub<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Sub<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Use `map_err` for broadcast_shapes error
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
        result_data.push(a_td.data[index_a] - b_td.data[index_b]);
    }

    drop(a_td);
    drop(b_td);

    // Use `?` for Tensor::new error
    let result = Tensor::new(result_data, result_shape.clone())?;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = SubBackward {
            input_a_shape: a_shape.clone(), // Keep original shapes for backward
            input_b_shape: b_shape.clone(),
            input_a: a.get_weak_ref(),
            input_b: b.get_weak_ref(),
            _phantom: PhantomData,
        };
        // Use set_grad_fn method
        result.set_grad_fn(Some(Rc::new(grad_fn))); 
    }
    Ok(result)
}

// --- std::ops::Sub implementation (calls the fallible function) ---
impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Sub<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    type Output = Tensor<T>;

    /// Panics if subtraction fails (e.g., incompatible shapes).
    /// Use `neurarust::ops::arithmetic::sub` for fallible subtraction.
    fn sub(self, other: &'b Tensor<T>) -> Self::Output {
        sub(self, other).unwrap_or_else(|e| panic!("Tensor subtraction failed: {:?}", e))
    }
}

/// Implements in-place element-wise subtraction (`-=`) for Tensor -= &Tensor.
/// NOTE: Currently does NOT support broadcasting.
impl<'a, T> SubAssign<&'a Tensor<T>> for Tensor<T>
where
    T: SubAssign + Copy + Clone,
{
    fn sub_assign(&mut self, other: &'a Tensor<T>) {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for SubAssign.");

        let mut self_td_mut = self.borrow_tensor_data_mut();
        let other_td = other.borrow_tensor_data();

        self_td_mut.data.iter_mut()
            .zip(other_td.data.iter())
            .for_each(|(a, &b)| *a -= b); // Requires T: SubAssign
    }
}

// --- Backward Operation --- 

#[derive(Debug)]
struct SubBackward<T> {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_a: Weak<RefCell<TensorData<T>>>,
    input_b: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

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
                // TODO: This assert should ideally return an error
                assert_eq!(existing_grad.shape(), local_gradient.shape(), "Gradient shape mismatch");
                *existing_grad += &local_gradient; // Requires AddAssign
            })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + Default + Debug + 'static + Add<Output = T> + Zero + One + Sum<T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        let needs_grad_a = self.input_a.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        let needs_grad_b = self.input_b.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        
        if needs_grad_a || needs_grad_b {
            let grad_clone = upstream_grad.clone();
            
            if needs_grad_a {
                // Assuming reduce_gradient handles internal errors or panics safely for now
                let grad_a = reduce_gradient(&grad_clone, &self.input_a_shape);
                accumulate_gradient(gradients, &self.input_a, grad_a);
            }
    
            if needs_grad_b {
                let grad_b_unreduced = reduce_gradient(&grad_clone, &self.input_b_shape);
                // Assuming Neg trait impl (-) doesn't panic/fail unexpectedly
                let grad_b = -&grad_b_unreduced; 
                accumulate_gradient(gradients, &self.input_b, grad_b);
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_a.clone(), self.input_b.clone()]
    }
}

// --- Tests --- 

#[cfg(test)]
mod tests {
    use super::*; // Import new `sub` function
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Sub, AddAssign, Neg};
    use std::fmt::Debug;
    use std::iter::Sum;
    
    
    use crate::error::NeuraRustError;
    use crate::tensor::ones;

    // Update test helpers to handle Result from Tensor::new
    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Sub<Output=T> + Neg<Output=T> + Default + Sum + 'static
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    fn create_test_tensor_with_grad<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Sub<Output=T> + Neg<Output=T> + Default + Sum + 'static
    {
        let tensor = Tensor::new(data, shape).expect("Test tensor_with_grad creation failed (new)");
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = create_test_tensor(vec![6_i32, 8, 10, 12], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![1_i32, 2, 3, 4];
        let expected_shape = vec![2, 2];
        
        // Use fallible sub function
        let result = sub(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();

        assert_eq!(res_tensor.data().to_vec(), expected_data, "Data mismatch");
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
        assert!(!res_tensor.requires_grad());
    }

    #[test]
    // Remove should_panic, check for specific error
    fn test_sub_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast_c = create_test_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        
        // Use fallible sub function
        let result = sub(&t1, &t_non_broadcast_c);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::BroadcastError { .. }));
    }

    #[test]
    fn test_sub_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        // Use fallible sub
        let res = sub(&t2, &t1).unwrap();
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = sub(&t1, &t3).unwrap();
        assert!(res2.requires_grad()); 

        let res3 = sub(&t2, &t3).unwrap();
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_sub_backward() -> Result<(), NeuraRustError> {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0, 6.0], vec![3]);
        let result = sub(&a, &b)?;

        // Provide upstream gradient of ones
        let upstream_grad = ones(result.shape()).expect("Failed to create upstream grad");
        result.backward(Some(&upstream_grad));

        let grad_a = a.grad().expect("Grad a missing");
        let grad_b = b.grad().expect("Grad b missing");

        let expected_grad_a_data = vec![1.0, 1.0, 1.0];
        let expected_grad_b_data = vec![-1.0, -1.0, -1.0];
        let expected_shape = vec![3];

        assert_eq!(grad_a.data().to_vec(), expected_grad_a_data, "Grad A data mismatch");
        assert_eq!(grad_a.shape(), expected_shape, "Grad A shape mismatch");
        assert_eq!(grad_b.data().to_vec(), expected_grad_b_data, "Grad B data mismatch");
        assert_eq!(grad_b.shape(), expected_shape, "Grad B shape mismatch");

        Ok(())
    }
    
    // TODO: Update broadcasting tests to use the new `sub` function and check Results.
    // test_sub_broadcast_scalar
    // test_sub_broadcast_vector
    // test_sub_broadcast_column_vector
    // test_sub_broadcast_backward_scalar
} 