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
            result.borrow_tensor_data_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
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
                assert_eq!(existing_grad.shape(), local_gradient.shape(), "Gradient shape mismatch");
                *existing_grad += &local_gradient;
            })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Neg<Output = T> + AddAssign + Copy + Clone + Default + Debug + 'static + Add<Output = T> + Zero + One + Sum<T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        // Check if inputs require gradient calculation
        let needs_grad_a = self.input_a.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        let needs_grad_b = self.input_b.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        
        if needs_grad_a || needs_grad_b {
            // Clone the upstream gradient as it might be needed for both inputs.
            let grad_clone = upstream_grad.clone();
            
            if needs_grad_a {
                // Gradient w.r.t. input A is simply the upstream gradient.
                // However, we need to potentially reduce its shape if broadcasting occurred.
                // `reduce_gradient` sums the gradient over the broadcasted dimensions
                // to match the original shape of input A.
                let grad_a = reduce_gradient(&grad_clone, &self.input_a_shape);
                accumulate_gradient(gradients, &self.input_a, grad_a);
            }
    
            if needs_grad_b {
                // Gradient w.r.t. input B is the negative of the upstream gradient.
                // We also need to reduce its shape if input B was broadcasted.
                let grad_b_unreduced = reduce_gradient(&grad_clone, &self.input_b_shape);
                // Negate the reduced gradient.
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
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Sub, AddAssign, Neg};
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::collections::HashMap;
    use std::rc::Rc;

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

        assert_eq!(result.data().to_vec(), expected_data, "Data mismatch");
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
        let grad_fn_option = c.borrow_tensor_data().grad_fn.clone();
        assert!(grad_fn_option.is_some());
        let grad_fn = grad_fn_option.unwrap();

        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![1.0, -1.0], vec![2]);
        grad_fn.backward(&upstream_grad, &mut gradients);

        {
            let grad_a_res = gradients.get(&Rc::as_ptr(&a.data)).expect("Grad A missing");
            let grad_b_res = gradients.get(&Rc::as_ptr(&b.data)).expect("Grad B missing");
            
            let expected_grad_a_data = vec![1.0, -1.0];
            let expected_grad_b_data = vec![-1.0, 1.0]; 
            let expected_shape = vec![2];
            assert_eq!(grad_a_res.data().to_vec(), expected_grad_a_data, "Grad A data mismatch");
            assert_eq!(grad_a_res.shape(), expected_shape, "Grad A shape mismatch");
            assert_eq!(grad_b_res.data().to_vec(), expected_grad_b_data, "Grad B data mismatch");
            assert_eq!(grad_b_res.shape(), expected_shape, "Grad B shape mismatch");
        }
        
        let upstream_grad_2 = Tensor::new(vec![0.5, 0.5], vec![2]);
        grad_fn.backward(&upstream_grad_2, &mut gradients);

        let grad_a_accum = gradients.get(&Rc::as_ptr(&a.data)).expect("Accum Grad A missing");
        let grad_b_accum = gradients.get(&Rc::as_ptr(&b.data)).expect("Accum Grad B missing");
        let expected_accum_grad_a_data = vec![1.5, -0.5]; 
        let expected_accum_grad_b_data = vec![-1.5, 0.5];
        let expected_accum_shape = vec![2];

        assert_eq!(grad_a_accum.data().to_vec(), expected_accum_grad_a_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.shape(), expected_accum_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.data().to_vec(), expected_accum_grad_b_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.shape(), expected_accum_shape, "Accum Grad B shape mismatch");
    }

    #[test]
    fn test_sub_broadcast_scalar() {
        let t1 = create_test_tensor(vec![11_i32, 12, 13, 14], vec![2, 2]);
        let s = create_test_tensor(vec![10_i32], vec![]); 
        let expected_data = vec![1, 2, 3, 4];
        let expected_shape = vec![2, 2];
        let result = &t1 - &s;
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), expected_shape);

        let s2 = create_test_tensor(vec![100_i32], vec![]); 
        let t2 = create_test_tensor(vec![1_i32, 10, 20, 30], vec![2, 2]);
        let expected_data_rev = vec![99, 90, 80, 70];
        let result_rev = &s2 - &t2;
        assert_eq!(result_rev.data().to_vec(), expected_data_rev);
        assert_eq!(result_rev.shape(), expected_shape);
    }

    #[test]
    fn test_sub_broadcast_vector() {
        let t1 = create_test_tensor(vec![11_i32, 22, 33, 14, 25, 36], vec![2, 3]);
        let v = create_test_tensor(vec![10_i32, 20, 30], vec![3]);
        let expected_data = vec![1, 2, 3, 4, 5, 6];
        let expected_shape = vec![2, 3];
        let result = &t1 - &v;
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    #[test]
    fn test_sub_broadcast_column_vector() {
        let t1 = create_test_tensor(vec![11_i32, 12, 13, 24, 25, 26], vec![2, 3]);
        let v_col = create_test_tensor(vec![10_i32, 20], vec![2, 1]);
        let expected_data = vec![1, 2, 3, 4, 5, 6];
        let expected_shape = vec![2, 3];
        let result = &t1 - &v_col;
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    #[test]
    fn test_sub_broadcast_backward_scalar() {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = create_test_tensor_with_grad::<f32>(vec![10.0], vec![]);
        
        let c = &a - &s;
        assert!(c.requires_grad());
        let grad_fn_option = c.borrow_tensor_data().grad_fn.clone();
        assert!(grad_fn_option.is_some());
        let grad_fn = grad_fn_option.unwrap();

        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        grad_fn.backward(&upstream_grad, &mut gradients);

        let grad_a = gradients.get(&Rc::as_ptr(&a.data)).expect("Grad A should exist");
        let grad_s = gradients.get(&Rc::as_ptr(&s.data)).expect("Grad S should exist");

        assert_eq!(grad_a.data().to_vec(), vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(grad_a.shape(), vec![2, 2]);

        let expected_grad_s_val = -(0.1 + 0.2 + 0.3 + 0.4);
        assert_eq!(grad_s.data().len(), 1);
        assert!((grad_s.data().to_vec()[0] - expected_grad_s_val).abs() < 1e-6);
        assert_eq!(grad_s.shape(), Vec::<usize>::new());
    }
} 