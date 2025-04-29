// neurarust-core/src/ops/arithmetic/add.rs

use crate::tensor::Tensor;
use crate::autograd::{BackwardOp, accumulate_gradient};
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, coord_to_index_broadcasted, reduce_gradient};
use std::ops::{Add, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::collections::HashMap;
use std::default::Default;

// --- Forward Operation --- 

/// Implements element-wise addition for two Tensors with broadcasting.
///
/// Performs `&tensor1 + &tensor2`.
/// The shapes of the two tensors must be identical.
/// Requires the element type `T` to implement `Add<Output = T>`, `AddAssign` (for grad), `Copy` and `Clone`.
impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Add<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    type Output = Tensor<T>;

    fn add(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        let result_shape = broadcast_shapes(&self_shape, &other_shape)
            .expect(&format!("Shapes {:?} and {:?} cannot be broadcasted for addition.", self_shape, other_shape));

        let self_td = self.borrow_tensor_data(); 
        let other_td = other.borrow_tensor_data();
        
        // Calculer la shape et les strides du résultat
        let numel_result = result_shape.iter().product();
        let mut result_data = Vec::with_capacity(numel_result);
        let result_strides = calculate_strides(&result_shape);
        let rank_diff_a = result_shape.len().saturating_sub(self_td.shape.len());
        let rank_diff_b = result_shape.len().saturating_sub(other_td.shape.len());
        
        let mut input_a_coords = vec![0; self_td.shape.len()];
        let mut input_b_coords = vec![0; other_td.shape.len()];

        for i in 0..numel_result {
            let output_coords = index_to_coord(i, &result_strides, &result_shape);
            
            // Calculer les coordonnées et l'offset pour l'input A
            for dim_idx in 0..self_td.shape.len() {
                let output_coord_idx = rank_diff_a + dim_idx;
                input_a_coords[dim_idx] = if self_td.shape[dim_idx] == 1 {
                    0
                } else {
                    output_coords[output_coord_idx]
                };
            }
            let offset_a = self_td.get_offset(&input_a_coords);
            
            // Calculer les coordonnées et l'offset pour l'input B
            for dim_idx in 0..other_td.shape.len() {
                let output_coord_idx = rank_diff_b + dim_idx;
                 input_b_coords[dim_idx] = if other_td.shape[dim_idx] == 1 {
                    0
                 } else {
                    output_coords[output_coord_idx]
                 };
            }
            let offset_b = other_td.get_offset(&input_b_coords);

            result_data.push(self_td.data[offset_a] + other_td.data[offset_b]);
        }

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = AddBackward {
                input_a_shape: self_shape.clone(),
                input_b_shape: other_shape.clone(),
                input_a: self.clone(),
                input_b: other.clone(),
                _phantom: PhantomData,
            };
            result.set_grad_fn(Some(Rc::new(grad_fn)));
        }
        result
    }
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
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Add, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    
    

    // Bounds mis à jour pour correspondre à l'impl Add
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_add_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![6_i32, 8, 10, 12];
        let expected_shape = vec![2, 2];
        let result = &t1 + &t2;
        
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic(expected = "cannot be broadcasted")]
    fn test_add_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        let _result = &t1 + &t_non_broadcast;
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

        let res1 = &t1 + &t2;
        assert!(res1.requires_grad());

        let res2 = &t1 + &t3;
        assert!(!res2.requires_grad());

        let t1_grad = create_test_tensor_with_grad::<f32>(vec![4.0], vec![1]);
        let res3 = &t1_grad + &t2; 
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_add_backward() {
        let a = create_test_tensor_with_grad::<f32>(vec![2.0, 3.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0], vec![2]);

        let c = &a + &b; // c = [6.0, 8.0]
        assert!(c.requires_grad());
        let grad_fn_option = c.borrow_tensor_data().grad_fn.clone(); 
        assert!(grad_fn_option.is_some());
        // let grad_fn = grad_fn_option.unwrap(); // Not needed anymore

        // Ensure grads are initially None
        assert!(a.grad().is_none());
        assert!(b.grad().is_none());
        
        // Sum the result to get a scalar loss, then call backward
        let loss = c.sum(); // loss = 14.0 (scalar)
        loss.backward(None); // Call backward on the scalar loss

        // Check gradients AFTER backward call
        let grad_a_opt = a.grad();
        let grad_b_opt = b.grad();
        assert!(grad_a_opt.is_some());
        assert!(grad_b_opt.is_some());

        // Borrow the unwrapped gradients
        let grad_a = grad_a_opt.as_ref().unwrap();
        let grad_b = grad_b_opt.as_ref().unwrap();

        let expected_grad_a_data = vec![1.0_f32, 1.0]; // dLoss/da = dLoss/dc * dc/da = 1.0 * 1.0
        let expected_grad_b_data = vec![1.0_f32, 1.0]; // dLoss/db = dLoss/dc * dc/db = 1.0 * 1.0
        let expected_shape = vec![2];

        assert_eq!(grad_a.data().to_vec(), expected_grad_a_data);
        assert_eq!(grad_a.shape(), expected_shape);
        assert_eq!(grad_b.data().to_vec(), expected_grad_b_data);
        assert_eq!(grad_b.shape(), expected_shape);
    }
} 