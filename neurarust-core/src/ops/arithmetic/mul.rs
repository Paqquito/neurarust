use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, reduce_gradient};
use std::ops::{Mul, AddAssign, Neg, MulAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::collections::HashMap;
use crate::error::NeuraRustError;
use crate::tensor::ones;

// --- Forward Operation --- 

/// Performs element-wise multiplication (Hadamard product) for two Tensors with broadcasting.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn mul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + Neg<Output=T>,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Use map_err for broadcast error
    let result_shape = broadcast_shapes(&a_shape, &b_shape)
        .map_err(|_e| NeuraRustError::BroadcastError { 
            shape1: a_shape.clone(), 
            shape2: b_shape.clone(), 
        })?;

    let a_td = a.borrow_tensor_data();
    let b_td = b.borrow_tensor_data();

    let numel_result = result_shape.iter().product();
    let mut result_data = Vec::with_capacity(numel_result);
    let result_strides = calculate_strides(&result_shape);
    let rank_diff_a = result_shape.len().saturating_sub(a_td.shape.len());
    let rank_diff_b = result_shape.len().saturating_sub(b_td.shape.len());
    
    let mut input_a_coords = vec![0; a_td.shape.len()];
    let mut input_b_coords = vec![0; b_td.shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, &result_shape);
        
        for dim_idx in 0..a_td.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_td.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        let offset_a = a_td.get_offset(&input_a_coords);
        
        for dim_idx in 0..b_td.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
             input_b_coords[dim_idx] = if b_td.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        let offset_b = b_td.get_offset(&input_b_coords);

        result_data.push(a_td.data[offset_a] * b_td.data[offset_b]);
    }

    drop(a_td);
    drop(b_td);

    // Use `?` for Tensor::new error
    let result = Tensor::new(result_data, result_shape.clone())?;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = MulBackward {
            input_a_shape: a_shape.clone(),
            input_b_shape: b_shape.clone(),
            input_a_ref: a.get_weak_ref(),
            input_b_ref: b.get_weak_ref(),
            input_a_val: a.clone(), // Clone tensors for backward pass
            input_b_val: b.clone(),
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }
    Ok(result)
}

/// Implements in-place element-wise multiplication (`*=`) for Tensor *= &Tensor.
/// NOTE: Currently does NOT support broadcasting.
impl<'a, T> MulAssign<&'a Tensor<T>> for Tensor<T>
where
    T: MulAssign + Copy + Clone, 
{
    fn mul_assign(&mut self, other: &'a Tensor<T>) {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for MulAssign.");

        let mut self_td_mut = self.borrow_tensor_data_mut();
        let other_td = other.borrow_tensor_data();

        self_td_mut.data.iter_mut()
            .zip(other_td.data.iter())
            .for_each(|(a, &b)| *a *= b); // Requires T: MulAssign
    }
}

// --- Backward Operation --- 

#[derive(Debug)]
struct MulBackward<T> {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    input_a_val: Tensor<T>,
    input_b_val: Tensor<T>,
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
                // TODO: Handle potential shape mismatch error
                assert_eq!(existing_grad.shape(), local_gradient.shape());
                *existing_grad += &local_gradient; 
            })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for MulBackward<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + Neg<Output=T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        let needs_grad_a = self.input_a_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        let needs_grad_b = self.input_b_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);

        if needs_grad_a || needs_grad_b {
            let grad_clone = upstream_grad.clone();
            
            if needs_grad_a {
                // Handle potential error from the multiplication operation
                let grad_a_unreduced = mul(&grad_clone, &self.input_b_val)
                    .expect("Internal error: Backward multiplication failed for grad_a");
                let grad_a = reduce_gradient(&grad_a_unreduced, &self.input_a_shape);
                accumulate_gradient(gradients, &self.input_a_ref, grad_a);
            }
            
            if needs_grad_b {
                // Handle potential error from the multiplication operation
                let grad_b_unreduced = mul(&grad_clone, &self.input_a_val)
                    .expect("Internal error: Backward multiplication failed for grad_b");
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
    use super::*; // Import the new `mul` function
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Mul, AddAssign, Neg};
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::collections::HashMap;
    use std::rc::Rc;
    use crate::error::NeuraRustError;

    // Update helpers to handle Result from Tensor::new
    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Mul<Output = T> + Default + Sum + Neg<Output=T> + 'static
    { 
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    fn create_test_tensor_with_grad<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Mul<Output = T> + Default + Sum + Neg<Output=T> + 'static
    {
        let tensor = Tensor::new(data, shape).expect("Test tensor_with_grad creation failed (new)");
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_mul_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![5_i32, 12, 21, 32];
        let expected_shape = vec![2, 2];
        
        // Use fallible mul function
        let result = mul(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        
        assert_eq!(res_tensor.data().to_vec(), expected_data, "Data mismatch");
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
        assert!(!res_tensor.requires_grad());
    }

    #[test]
    // Remove should_panic, check for specific error
    fn test_mul_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        
        // Use fallible mul function
        let result = mul(&t1, &t_non_broadcast);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::BroadcastError { .. }));
    }

    #[test]
    fn test_mul_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        // Use fallible mul
        let res = mul(&t1, &t2).unwrap();
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = mul(&t3, &t1).unwrap();
        assert!(res2.requires_grad());

        let res3 = mul(&t2, &t3).unwrap();
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_mul_backward() -> Result<(), NeuraRustError> {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0, 6.0], vec![3]);
        let result = mul(&a, &b)?;

        // Provide upstream gradient of ones
        let upstream_grad = ones(result.shape()).expect("Failed to create upstream grad");
        result.backward(Some(&upstream_grad));

        let grad_a = a.grad().expect("Grad a missing");
        let grad_b = b.grad().expect("Grad b missing");

        let expected_grad_a_data = vec![4.0, 5.0, 6.0];
        let expected_grad_b_data = vec![1.0, 2.0, 3.0];
        let expected_shape = vec![3];

        assert_eq!(grad_a.data().to_vec(), expected_grad_a_data, "Grad A data mismatch");
        assert_eq!(grad_a.shape(), expected_shape, "Grad A shape mismatch");
        assert_eq!(grad_b.data().to_vec(), expected_grad_b_data, "Grad B data mismatch");
        assert_eq!(grad_b.shape(), expected_shape, "Grad B shape mismatch");
        Ok(())
    }
    
    // TODO: Update broadcasting tests to use the new `mul` function and check Results.
    // test_mul_broadcast_scalar
    // test_mul_broadcast_vector
    // test_mul_broadcast_backward_scalar
} 