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
            result.borrow_tensor_data_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

#[derive(Debug)]
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
            .and_modify(|existing_grad| { *existing_grad += &local_gradient; })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for DivBackward<T>
where
    T: Div<Output = T> + Mul<Output = T> + Neg<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        // Check if inputs require gradient calculation
        let needs_grad_a = self.input_a_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        let needs_grad_b = self.input_b_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);

        if needs_grad_a || needs_grad_b {
            // Clone the upstream gradient as it might be needed for both inputs.
            let grad_clone = upstream_grad.clone();
            
            if needs_grad_a {
                // Gradient w.r.t. input A is upstream_grad / input_B.
                // Perform element-wise division, handling potential broadcasting.
                let grad_a_unreduced = &grad_clone / &self.input_b_val; 
                // Reduce the gradient shape if broadcasting occurred during the forward Div.
                let grad_a = reduce_gradient(&grad_a_unreduced, &self.input_a_shape); 
                accumulate_gradient(gradients, &self.input_a_ref, grad_a);
            }
            
            if needs_grad_b {
                // Gradient w.r.t. input B is upstream_grad * (-input_A / input_B^2).
                // Calculate intermediate terms, handling potential broadcasting.
                let b_squared = &self.input_b_val * &self.input_b_val; // B^2
                let a_div_b_squared = &self.input_a_val / &b_squared; // A / B^2
                let neg_a_div_b_squared = -&a_div_b_squared;         // -A / B^2
                let grad_b_unreduced = &grad_clone * &neg_a_div_b_squared; // upstream * (-A / B^2)
                // Reduce the gradient shape if broadcasting occurred during the forward Div.
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
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Div, Mul, Neg, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::collections::HashMap;
    use std::rc::Rc;

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

        assert_eq!(result.data().to_vec(), expected_data, "Data mismatch");
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
        let grad_fn_opt = c.borrow_tensor_data().grad_fn.clone();
        assert!(grad_fn_opt.is_some());
        let grad_fn = grad_fn_opt.unwrap();

        let mut gradients = HashMap::new();
        // Simuler l'appel depuis Tensor::backward
        let initial_upstream_grad = Tensor::new(vec![1.0_f32], vec![1]);
        gradients.insert(Rc::as_ptr(&c.data), initial_upstream_grad);
        
        // Cloner le gradient avant de l'utiliser et de passer `gradients` en mutable
        let upstream_grad_clone = gradients.get(&Rc::as_ptr(&c.data)).unwrap().clone();
        grad_fn.backward(&upstream_grad_clone, &mut gradients);

        // Vérifier les gradients dans le HashMap
        let grad_a_res = gradients.get(&Rc::as_ptr(&a.data)).expect("Grad A missing");
        let grad_b_res = gradients.get(&Rc::as_ptr(&b.data)).expect("Grad B missing");

        let expected_grad_a = vec![0.5_f32];
        let expected_grad_b = vec![-1.5_f32];

        assert_eq!(grad_a_res.data().to_vec(), expected_grad_a, "Gradient for A mismatch");
        assert_eq!(grad_a_res.shape(), vec![1]);
        assert_eq!(grad_b_res.data().to_vec(), expected_grad_b, "Gradient for B mismatch");
        assert_eq!(grad_b_res.shape(), vec![1]);
    }

    // NEW BROADCASTING TESTS (Forward only for now)
    #[test]
    fn test_div_broadcast_scalar() {
        let t1 = create_test_tensor(vec![10.0_f32, 20.0, 30.0, 40.0], vec![2, 2]);
        let s = create_test_tensor(vec![10.0_f32], vec![]); 
        let expected_data = vec![1.0, 2.0, 3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = &t1 / &s;
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), expected_shape);

        // s / t1
        let s2 = create_test_tensor(vec![100.0_f32], vec![]); 
        let t2 = create_test_tensor(vec![10.0_f32, 20.0, 50.0, 100.0], vec![2, 2]);
        let expected_data_rev = vec![10.0, 5.0, 2.0, 1.0];
        let result_rev = &s2 / &t2;
        assert_eq!(result_rev.data().to_vec(), expected_data_rev);
        assert_eq!(result_rev.shape(), expected_shape);
    }

    #[test]
    fn test_div_broadcast_vector() {
        let t1 = create_test_tensor(vec![10_i32, 20, 30, 40, 50, 60], vec![2, 3]);
        let v = create_test_tensor(vec![10_i32, 10, 10], vec![3]); // Row vector
        let expected_data = vec![1, 2, 3, 4, 5, 6];
        let expected_shape = vec![2, 3];
        let result = &t1 / &v;
        assert_eq!(result.data().to_vec(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    // TODO: Add backward broadcasting tests for Div (once Mul also supports broadcasting)
}