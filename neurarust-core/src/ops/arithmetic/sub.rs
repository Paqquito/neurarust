use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, coord_to_index_broadcasted, reduce_gradient};
use std::ops::{Sub, Neg, AddAssign, Add};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;

// --- Forward Operation --- 

/// Implements element-wise subtraction for two Tensors with broadcasting.
impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Sub<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    type Output = Tensor<T>;

    fn sub(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();

        let result_shape = broadcast_shapes(&self_shape, &other_shape)
            .expect(&format!("Shapes {:?} and {:?} cannot be broadcasted for subtraction.", self_shape, other_shape));

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

        let numel_result = result_shape.iter().product();
        let mut result_data = Vec::with_capacity(numel_result);
        let strides_a = calculate_strides(&self_shape);
        let strides_b = calculate_strides(&other_shape);
        let result_strides = calculate_strides(&result_shape);

        for i in 0..numel_result {
            let multi_index = index_to_coord(i, &result_strides, &result_shape);
            let index_a = coord_to_index_broadcasted(&multi_index, &self_shape, &strides_a);
            let index_b = coord_to_index_broadcasted(&multi_index, &other_shape, &strides_b);
            result_data.push(self_td.data[index_a] - other_td.data[index_b]);
        }

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, result_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = SubBackward {
                input_a_shape: self_shape.clone(),
                input_b_shape: other_shape.clone(),
                input_a: self.get_weak_ref(),
                input_b: other.get_weak_ref(),
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct SubBackward<T> {
    input_a_shape: Vec<usize>,
    input_b_shape: Vec<usize>,
    input_a: Weak<RefCell<TensorData<T>>>,
    input_b: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + Default + Debug + 'static + Add<Output = T> + Zero + One + Sum<T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        let grad_clone = upstream_grad.clone();
        grad_clone.set_requires_grad(false);

        if let Some(input_a_rc) = self.input_a.upgrade() {
            let mut input_a_td = input_a_rc.borrow_mut();
            if input_a_td.requires_grad {
                let grad_a = reduce_gradient(&grad_clone, &self.input_a_shape);
                if let Some(existing_grad_tensor) = input_a_td.grad.as_mut() {
                    let mut existing_data = existing_grad_tensor.borrow_tensor_data_mut();
                    let new_grad_data = grad_a.borrow_tensor_data();
                    assert_eq!(existing_data.data.len(), new_grad_data.data.len());
                    for (existing, new) in existing_data.data.iter_mut().zip(new_grad_data.data.iter()) {
                        *existing += *new;
                    }
                } else {
                    input_a_td.grad = Some(Tensor::new(grad_a.data(), grad_a.shape()));
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input A in SubBackward.");
        }

        if let Some(input_b_rc) = self.input_b.upgrade() {
            let mut input_b_td = input_b_rc.borrow_mut();
            if input_b_td.requires_grad {
                let grad_b_unneg = reduce_gradient(&grad_clone, &self.input_b_shape);
                let grad_b = -&grad_b_unneg;
                if let Some(existing_grad_tensor) = input_b_td.grad.as_mut() {
                    let mut existing_data = existing_grad_tensor.borrow_tensor_data_mut();
                    let new_grad_data = grad_b.borrow_tensor_data();
                    assert_eq!(existing_data.data.len(), new_grad_data.data.len());
                    for (existing, new) in existing_data.data.iter_mut().zip(new_grad_data.data.iter()) {
                        *existing += *new; 
                    }
                } else {
                    input_b_td.grad = Some(Tensor::new(grad_b.data(), grad_b.shape()));
                }
            }
        } else {
             eprintln!("Warning: Weak ref upgrade failed for input B in SubBackward.");
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_a.clone(), self.input_b.clone()]
    }
}

// --- Tests --- 

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::{Zero, One};
    use crate::tensor::utils::broadcast_shapes;
    use crate::autograd::BackwardOp;
    use std::ops::{Sub, AddAssign, Neg};
    use std::fmt::Debug;
    use std::iter::Sum;

    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Sub<Output=T> + Neg<Output=T> + Default + Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Sub<Output=T> + Neg<Output=T> + Default + Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = create_test_tensor(vec![6_i32, 8, 10, 12], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![1_i32, 2, 3, 4];
        let expected_shape = vec![2, 2];
        let result = &t1 - &t2;

        assert_eq!(result.data(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic(expected = "cannot be broadcasted")]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast_c = create_test_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        let _result = &t1 - &t_non_broadcast_c;
    }

    #[test]
    fn test_sub_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let res = &t2 - &t1;
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = &t1 - &t3;
        assert!(res2.requires_grad()); 

        let res3 = &t2 - &t3;
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_sub_backward() {
        let a = create_test_tensor_with_grad::<f32>(vec![10.0, 20.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![3.0, 8.0], vec![2]);

        let c = &a - &b;
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
            let expected_grad_a_data = vec![1.0, -1.0];
            let expected_grad_b_data = vec![-1.0, 1.0]; 
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
        let expected_accum_grad_a_data = vec![1.5, -0.5]; 
        let expected_accum_grad_b_data = vec![-1.5, 0.5];
        let expected_accum_shape = vec![2];

        assert_eq!(grad_a_accum.as_ref().unwrap().data(), expected_accum_grad_a_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().data(), expected_accum_grad_b_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.as_ref().unwrap().shape(), expected_accum_shape, "Accum Grad B shape mismatch");
    }

    #[test]
    fn test_sub_broadcast_scalar() {
        let t1 = create_test_tensor(vec![11_i32, 12, 13, 14], vec![2, 2]);
        let s = create_test_tensor(vec![10_i32], vec![]); 
        let expected_data = vec![1, 2, 3, 4];
        let expected_shape = vec![2, 2];
        let result = &t1 - &s;
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);

        let s2 = create_test_tensor(vec![100_i32], vec![]); 
        let t2 = create_test_tensor(vec![1_i32, 10, 20, 30], vec![2, 2]);
        let expected_data_rev = vec![99, 90, 80, 70];
        let result_rev = &s2 - &t2;
        assert_eq!(result_rev.data(), expected_data_rev);
        assert_eq!(result_rev.shape(), expected_shape);
    }

    #[test]
    fn test_sub_broadcast_vector() {
        let t1 = create_test_tensor(vec![11_i32, 22, 33, 14, 25, 36], vec![2, 3]);
        let v = create_test_tensor(vec![10_i32, 20, 30], vec![3]);
        let expected_data = vec![1, 2, 3, 4, 5, 6];
        let expected_shape = vec![2, 3];
        let result = &t1 - &v;
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    #[test]
    fn test_sub_broadcast_column_vector() {
        let t1 = create_test_tensor(vec![11_i32, 12, 13, 24, 25, 26], vec![2, 3]);
        let v_col = create_test_tensor(vec![10_i32, 20], vec![2, 1]);
        let expected_data = vec![1, 2, 3, 4, 5, 6];
        let expected_shape = vec![2, 3];
        let result = &t1 - &v_col;
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    #[test]
    fn test_sub_broadcast_backward_scalar() {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = create_test_tensor_with_grad::<f32>(vec![10.0], vec![]);
        
        let c = &a - &s;
        assert!(c.requires_grad());
        let grad_fn_option = c.0.borrow().grad_fn.clone();
        assert!(grad_fn_option.is_some());
        let grad_fn = grad_fn_option.unwrap();

        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        grad_fn.backward(&upstream_grad);

        let grad_a_ref = a.borrow_grad(); 
        let grad_a = grad_a_ref.as_ref().expect("Grad A should exist");
        let grad_s_ref = s.borrow_grad();
        let grad_s = grad_s_ref.as_ref().expect("Grad S should exist");

        assert_eq!(grad_a.data(), vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(grad_a.shape(), vec![2, 2]);

        let expected_grad_s_val = -(0.1 + 0.2 + 0.3 + 0.4);
        assert_eq!(grad_s.data().len(), 1);
        assert!((grad_s.data()[0] - expected_grad_s_val).abs() < 1e-6);
        assert_eq!(grad_s.shape(), vec![1, 1]); 
    }
} 