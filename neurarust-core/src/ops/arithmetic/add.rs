// neurarust-core/src/ops/arithmetic/add.rs

use crate::tensor::Tensor;
use crate::autograd::{BackwardOp, accumulate_gradient};
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, reduce_gradient};
use std::ops::{Add, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::collections::HashMap;
use std::default::Default;
use crate::error::NeuraRustError;
use crate::tensor::ones;

// --- Forward Operation --- 

/// Performs element-wise addition for two Tensors with broadcasting.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn add<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Add<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Use `map_err` to convert the String error from broadcast_shapes
    let result_shape = broadcast_shapes(&a_shape, &b_shape)
        .map_err(|e| NeuraRustError::BroadcastError { 
            shape1: a_shape.clone(), // Or provide more context from `e` if needed
            shape2: b_shape.clone(), 
            // message: e // Or include the original message
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

        result_data.push(a_td.data[offset_a] + b_td.data[offset_b]);
    }

    drop(a_td);
    drop(b_td);

    let result = Tensor::new(result_data, result_shape.clone())?; 

    let requires_grad = a.requires_grad() || b.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = AddBackward {
            input_a_shape: a_shape.clone(),
            input_b_shape: b_shape.clone(),
            input_a: a.clone(), 
            input_b: b.clone(), 
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }
    Ok(result)
}

/// Implements in-place element-wise addition (`+=`) for Tensor += &Tensor.
/// NOTE: Does not support broadcasting.
impl<'a, T> AddAssign<&'a Tensor<T>> for Tensor<T>
where
    T: AddAssign + Copy + Clone, // Copy needed to read from `other`
{
    fn add_assign(&mut self, other: &'a Tensor<T>) {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(self_shape, other_shape, "Tensor shapes must match for AddAssign.");

        let mut self_td_mut = self.borrow_tensor_data_mut();
        let other_td = other.borrow_tensor_data();

        self_td_mut.data.iter_mut()
            .zip(other_td.data.iter())
            .for_each(|(a, &b)| *a += b); // Requires T: AddAssign + Copy
    }
}

// --- Backward Operation --- 

/// Backward operation for addition
#[derive(Debug)]
struct AddBackward<T: 'static> {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_a: Tensor<T>,
    input_b: Tensor<T>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for AddBackward<T>
where 
    T: AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        let grad_clone = upstream_grad.clone();
        let weak_a = self.input_a.get_weak_ref();
        let weak_b = self.input_b.get_weak_ref();

        if weak_a.upgrade().map_or(false, |rc| rc.borrow().requires_grad) {
            let grad_a = reduce_gradient(&grad_clone, &self.input_a_shape);
            grad_a.set_requires_grad(false);
            accumulate_gradient(gradients, &weak_a, grad_a);
        }

        if weak_b.upgrade().map_or(false, |rc| rc.borrow().requires_grad) {
            let grad_b = reduce_gradient(&grad_clone, &self.input_b_shape);
            grad_b.set_requires_grad(false);
            accumulate_gradient(gradients, &weak_b, grad_b);
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_a.get_weak_ref(), self.input_b.get_weak_ref()]
    }
}

// --- Tests --- 

#[cfg(test)]
mod tests {
    use super::*; 
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Add, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    use crate::error::NeuraRustError;
    use crate::tensor::utils::broadcast_shapes; // Import broadcast_shapes if needed for direct testing
    use crate::tensor::ones; // Import ones

    // Helpers remain the same
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + Sum>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> { 
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + Sum + 'static>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> { 
        let tensor = Tensor::new(data, shape).expect("Test tensor_with_grad creation failed (new)");
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_add_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![6_i32, 8, 10, 12];
        let expected_shape = vec![2, 2];
        
        let result = add(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        
        assert_eq!(res_tensor.data().to_vec(), expected_data);
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
        assert!(!res_tensor.requires_grad());

        // Remove the test for the Add trait for now
        // let res_op = &t1 + &t2;
        // assert_eq!(res_op.data().to_vec(), expected_data);
        // assert_eq!(res_op.shape(), expected_shape);
    }

    #[test]
    fn test_add_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        
        let result = add(&t1, &t_non_broadcast);
        assert!(result.is_err());
        // Check the specific error type
        match result.err().unwrap() {
            NeuraRustError::BroadcastError { shape1, shape2 } => {
                assert_eq!(shape1, vec![2, 2]);
                assert_eq!(shape2, vec![2, 3]);
            },
            _ => panic!("Incorrect error type returned"),
        }

        // Remove the panic test for the Add trait impl
        // let panic_result = std::panic::catch_unwind(|| { &t1 + &t_non_broadcast });
        // assert!(panic_result.is_err());
    }

    #[test]
    fn test_add_assign_ok() {
        let mut t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![6_i32, 8, 10, 12];
        
        t1 += &t2; // Use AddAssign

        assert_eq!(t1.data().to_vec(), expected_data, "Data mismatch after AddAssign");
        assert_eq!(t1.shape(), vec![2, 2], "Shape mismatch after AddAssign");
    }

    #[test]
    #[should_panic(expected = "Tensor shapes must match for AddAssign")]
    fn test_add_assign_shape_mismatch() {
        let mut t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_wrong_shape = create_test_tensor(vec![5_i32, 6, 7], vec![3]);
        t1 += &t_wrong_shape; // Should panic
    }

    #[test]
    fn test_add_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]); 
        let t3 = create_test_tensor::<f32>(vec![3.0], vec![1]);

        // Use fallible add
        let res1 = add(&t1, &t2).unwrap();
        assert!(res1.requires_grad());

        let res2 = add(&t1, &t3).unwrap();
        assert!(!res2.requires_grad());

        let t1_grad = create_test_tensor_with_grad::<f32>(vec![4.0], vec![1]);
        let res3 = add(&t1_grad, &t2).unwrap(); 
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_add_backward() -> Result<(), NeuraRustError> {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0, 6.0], vec![3]);
        let result = add(&a, &b)?;

        // Provide upstream gradient of ones using the standalone ones function
        let upstream_grad = ones(result.shape()).expect("Failed to create upstream grad");
        result.backward(Some(&upstream_grad));

        let grad_a = a.grad().expect("Grad a missing");
        assert_eq!(grad_a.data().to_vec(), vec![1.0, 1.0, 1.0]);
        let grad_b = b.grad().expect("Grad b missing");
        assert_eq!(grad_b.data().to_vec(), vec![1.0, 1.0, 1.0]);
        Ok(())
    }
} 