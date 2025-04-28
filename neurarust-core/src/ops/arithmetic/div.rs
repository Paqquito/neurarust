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

// --- Forward Operation --- 

/// Implements element-wise division for two Tensors with broadcasting.
impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum,
{
    type Output = Tensor<T>;

    fn div(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        let result_shape = broadcast_shapes(&self_shape, &other_shape)
            .expect(&format!("Shapes {:?} and {:?} cannot be broadcasted for division.", self_shape, other_shape));

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        // Broadcasted computation
        let numel_result = result_shape.iter().product();
        let mut result_data = Vec::with_capacity(numel_result);
        let strides_a = calculate_strides(&self_shape);
        let strides_b = calculate_strides(&other_shape);
        let result_strides = calculate_strides(&result_shape);

        for i in 0..numel_result {
            let multi_index = index_to_coord(i, &result_strides, &result_shape);
            let index_a = coord_to_index_broadcasted(&multi_index, &self_shape, &strides_a);
            let index_b = coord_to_index_broadcasted(&multi_index, &other_shape, &strides_b);
            // TODO: Add check for division by zero?
            result_data.push(self_td.data[index_a] / other_td.data[index_b]); // Division
        }

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = DivBackward {
                input_a_shape: self_shape.clone(),
                input_b_shape: other_shape.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
                // Store cloned inputs needed for backward calculation
                input_a_val: self.clone(), 
                input_b_val: other.clone(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct DivBackward<T> {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_a_ref: Weak<RefCell<TensorData<T>>>,
    input_b_ref: Weak<RefCell<TensorData<T>>>,
    // Store values needed for grad calculation: A and B
    input_a_val: Tensor<T>,
    input_b_val: Tensor<T>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for DivBackward<T>
where
    // Restore full bounds needed for reduce_gradient
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        let grad_clone = upstream_grad.clone();
        grad_clone.set_requires_grad(false);

        // Restore reduce_gradient calls 
        // NOTE: Still assumes Mul/Div ops handle broadcasting internally, which they don't yet.
        // Placeholder comment remains valid.
        // TODO: Update this backward pass once Mul/Div support broadcasting.

        // Calculate grad_a = upstream_grad / B
        let grad_a_reduced = &grad_clone / &self.input_b_val; 
        let grad_a = reduce_gradient(&grad_a_reduced, &self.input_a_shape); // Restore reduce
        
        // Calculate grad_b = upstream_grad * (-A / B^2)
        let b_squared = &self.input_b_val * &self.input_b_val;
        let a_div_b_squared = &self.input_a_val / &b_squared; 
        let neg_a_div_b_squared = -&a_div_b_squared; 
        let grad_b_reduced = &grad_clone * &neg_a_div_b_squared; 
        let grad_b = reduce_gradient(&grad_b_reduced, &self.input_b_shape); // Restore reduce

        // Accumulate gradients
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
    use num_traits::{Zero, One};
    use std::ops::{Div, Mul, Neg, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    use crate::autograd::BackwardOp;

    // Add necessary bounds to helpers
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Div<Output = T> + Mul<Output=T> + Neg<Output=T> + Default + Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
     fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Div<Output = T> + Mul<Output=T> + Neg<Output=T> + Default + Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0_f32, 12.0, 21.0, 32.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let expected_data = vec![2.0_f32, 2.0, 3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = &t1 / &t2;

        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic] 
    fn test_div_tensors_int_div_by_zero() {
        let t1 = create_test_tensor(vec![10_i32], vec![1]);
        let t2 = create_test_tensor(vec![0_i32], vec![1]);
        let _result = &t1 / &t2;
    }

    #[test]
    #[should_panic(expected = "cannot be broadcasted")]
    fn test_div_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
        let _result = &t1 / &t_non_broadcast;
    }

    #[test]
    fn test_div_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor::<f32>(vec![2.0], vec![1]);
        let res = &t1 / &t2;
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = &t2 / &t3; // t3 requires grad
        assert!(res2.requires_grad());

        let res3 = &t1 / &t3; // both require grad
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_div_backward() {
        let a = create_test_tensor_with_grad(vec![6.0_f32], vec![1]);
        let b = create_test_tensor_with_grad(vec![2.0_f32], vec![1]);
        let c = &a / &b; // c = 3.0

        assert!(c.requires_grad());
        assert!(c.grad_fn().is_some());

        c.backward(); // Initial gradient (dC/dC) = 1.0

        // Expected gradients:
        // grad_a = dC/dA = 1/B = 1/2 = 0.5
        // grad_b = dC/dB = -A/B^2 = -6 / (2*2) = -6 / 4 = -1.5
        let expected_grad_a = vec![0.5_f32];
        let expected_grad_b = vec![-1.5_f32];

        assert_eq!(a.grad().unwrap().data(), expected_grad_a, "Gradient for A mismatch");
        assert_eq!(b.grad().unwrap().data(), expected_grad_b, "Gradient for B mismatch");
    }

    // NEW BROADCASTING TESTS (Forward only for now)
    #[test]
    fn test_div_broadcast_scalar() {
        let t1 = create_test_tensor(vec![10.0_f32, 20.0, 30.0, 40.0], vec![2, 2]);
        let s = create_test_tensor(vec![10.0_f32], vec![]); 
        let expected_data = vec![1.0, 2.0, 3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = &t1 / &s;
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);

        // s / t1
        let s2 = create_test_tensor(vec![100.0_f32], vec![]); 
        let t2 = create_test_tensor(vec![10.0_f32, 20.0, 50.0, 100.0], vec![2, 2]);
        let expected_data_rev = vec![10.0, 5.0, 2.0, 1.0];
        let result_rev = &s2 / &t2;
        assert_eq!(result_rev.data(), expected_data_rev);
        assert_eq!(result_rev.shape(), expected_shape);
    }

    #[test]
    fn test_div_broadcast_vector() {
        let t1 = create_test_tensor(vec![10_i32, 20, 30, 40, 50, 60], vec![2, 3]);
        let v = create_test_tensor(vec![10_i32, 10, 10], vec![3]); // Row vector
        let expected_data = vec![1, 2, 3, 4, 5, 6];
        let expected_shape = vec![2, 3];
        let result = &t1 / &v;
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    // TODO: Add backward broadcasting tests for Div (once Mul also supports broadcasting)
}