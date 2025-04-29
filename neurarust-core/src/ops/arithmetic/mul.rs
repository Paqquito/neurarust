use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord, coord_to_index_broadcasted, reduce_gradient};
use std::ops::{Mul, AddAssign, Neg, MulAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::collections::HashMap;

// --- Forward Operation --- 

/// Implements element-wise multiplication (Hadamard product) for two Tensors with broadcasting.
impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + 'static + Default + Debug + Zero + One + Sum + Neg<Output=T>,
{
    type Output = Tensor<T>;

    fn mul(self, other: &'b Tensor<T>) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();

        let result_shape = broadcast_shapes(&self_shape, &other_shape)
            .expect(&format!("Shapes {:?} and {:?} cannot be broadcasted for multiplication.", self_shape, other_shape));

        let self_td = self.borrow_tensor_data();
        let other_td = other.borrow_tensor_data();

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

            result_data.push(self_td.data[offset_a] * other_td.data[offset_b]);
        }

        drop(self_td);
        drop(other_td);

        let requires_grad = self.requires_grad() || other.requires_grad();
        let result = Tensor::new(result_data, result_shape.clone());
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = MulBackward {
                input_a_shape: self_shape.clone(),
                input_b_shape: other_shape.clone(),
                input_a_ref: self.get_weak_ref(),
                input_b_ref: other.get_weak_ref(),
                input_a_val: self.clone(),
                input_b_val: other.clone(),
                _phantom: PhantomData,
            };
            result.borrow_tensor_data_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
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
            .and_modify(|existing_grad| { *existing_grad += &local_gradient; })
            .or_insert(local_gradient);
    }
}

impl<T> BackwardOp<T> for MulBackward<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + Neg<Output=T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        // Check if inputs require gradient calculation
        let needs_grad_a = self.input_a_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        let needs_grad_b = self.input_b_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);

        if needs_grad_a || needs_grad_b {
            // Clone the upstream gradient as it might be needed for both inputs.
            let grad_clone = upstream_grad.clone();
            
            if needs_grad_a {
                // Gradient w.r.t. input A is upstream_grad * input_B.
                // Perform the element-wise multiplication. Note that this multiplication itself
                // might involve broadcasting if upstream_grad and input_b_val have broadcastable shapes 
                // (which happens if the original A and B were broadcasted for the forward Mul).
                let grad_a_unreduced = &grad_clone * &self.input_b_val;
                // Reduce the gradient shape if broadcasting occurred during the forward Mul.
                // This sums the gradient over the dimensions that were originally broadcasted for input A.
                let grad_a = reduce_gradient(&grad_a_unreduced, &self.input_a_shape);
                accumulate_gradient(gradients, &self.input_a_ref, grad_a);
            }
            
            if needs_grad_b {
                // Gradient w.r.t. input B is upstream_grad * input_A.
                // Perform the element-wise multiplication, handling potential broadcasting.
                let grad_b_unreduced = &grad_clone * &self.input_a_val;
                // Reduce the gradient shape if broadcasting occurred during the forward Mul.
                // This sums the gradient over the dimensions that were originally broadcasted for input B.
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
    use std::ops::{Mul, AddAssign, Neg};
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Mul<Output = T> + Default + Sum + Neg<Output=T>>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Mul<Output = T> + Default + Sum + Neg<Output=T>>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_mul_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![5_i32, 12, 21, 32];
        let expected_shape = vec![2, 2];
        let result = &t1 * &t2;
        
        assert_eq!(result.data().to_vec(), expected_data, "Data mismatch");
        assert_eq!(result.shape(), expected_shape, "Shape mismatch");
        assert!(!result.requires_grad());
    }

    #[test]
    #[should_panic(expected = "cannot be broadcasted")]
    fn test_mul_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        let _result = &t1 * &t_non_broadcast;
    }

    #[test]
    fn test_mul_propagate_requires_grad() {
        let t1 = create_test_tensor::<f32>(vec![1.0], vec![1]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let res = &t1 * &t2;
        assert!(res.requires_grad());

        let t3 = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        let res2 = &t3 * &t1;
        assert!(res2.requires_grad());

        let res3 = &t2 * &t3;
        assert!(res3.requires_grad());
    }

    #[test]
    fn test_mul_backward() {
        let a = create_test_tensor_with_grad::<f32>(vec![2.0, 3.0], vec![2]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0], vec![2]);
        let c = &a * &b;

        assert!(c.requires_grad());
        let grad_fn = c.borrow_tensor_data().grad_fn.clone().unwrap();

        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![1.0, -1.0], vec![2]);
        gradients.insert(Rc::as_ptr(&c.data), upstream_grad.clone());
        grad_fn.backward(&upstream_grad, &mut gradients);

        {
            let grad_a = gradients.get(&Rc::as_ptr(&a.data)).expect("Grad A missing");
            let grad_b = gradients.get(&Rc::as_ptr(&b.data)).expect("Grad B missing");
            
            let expected_grad_a_data = vec![4.0, -5.0];
            let expected_grad_b_data = vec![2.0, -3.0]; 
            let expected_shape = vec![2];
            assert_eq!(grad_a.data().to_vec(), expected_grad_a_data, "Grad A data mismatch");
            assert_eq!(grad_a.shape(), expected_shape, "Grad A shape mismatch");
            assert_eq!(grad_b.data().to_vec(), expected_grad_b_data, "Grad B data mismatch");
            assert_eq!(grad_b.shape(), expected_shape, "Grad B shape mismatch");
        }

        let upstream_grad_2 = Tensor::new(vec![0.5, 0.5], vec![2]);
        grad_fn.backward(&upstream_grad_2, &mut gradients); 
        
        let grad_a_accum = gradients.get(&Rc::as_ptr(&a.data)).expect("Accum Grad A missing");
        let grad_b_accum = gradients.get(&Rc::as_ptr(&b.data)).expect("Accum Grad B missing");
        
        let expected_accum_grad_a_data = vec![6.0, -2.5]; 
        let expected_accum_grad_b_data = vec![3.0, -1.5];
        let expected_accum_shape = vec![2];

        assert_eq!(grad_a_accum.data().to_vec(), expected_accum_grad_a_data, "Accum Grad A data mismatch");
        assert_eq!(grad_a_accum.shape(), expected_accum_shape, "Accum Grad A shape mismatch");
        assert_eq!(grad_b_accum.data().to_vec(), expected_accum_grad_b_data, "Accum Grad B data mismatch");
        assert_eq!(grad_b_accum.shape(), expected_accum_shape, "Accum Grad B shape mismatch");
    }

    #[test]
    fn test_mul_broadcast_scalar() {
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0], vec![3]);
        let scalar = create_test_tensor(vec![2.0_f32], vec![1]);
        let result = &t1 * &scalar;
        let expected_shape = vec![3];
        assert_eq!(result.shape(), expected_shape);
        assert_eq!(result.data().to_vec(), vec![2.0, 4.0, 6.0]);

        let result2 = &scalar * &t1;
        assert_eq!(result2.shape(), expected_shape);
        assert_eq!(result2.data().to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_broadcast_vector() {
        let t_matrix = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t_vector = create_test_tensor(vec![10.0_f32, 20.0, 30.0], vec![3]);
        let result = &t_matrix * &t_vector;
        let expected_shape = vec![2, 3];
        assert_eq!(result.shape(), expected_shape);
        assert_eq!(result.data().to_vec(), vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    }

    #[test]
    fn test_mul_broadcast_backward_scalar() {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
        let s = create_test_tensor_with_grad::<f32>(vec![10.0], vec![]);
        
        let c = &a * &s;
        assert!(c.requires_grad());
        let grad_fn = c.borrow_tensor_data().grad_fn.clone().unwrap();

        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);
        gradients.insert(Rc::as_ptr(&c.data), upstream_grad.clone());
        grad_fn.backward(&upstream_grad, &mut gradients);

        let grad_a = gradients.get(&Rc::as_ptr(&a.data)).expect("Grad A missing");
        let grad_s = gradients.get(&Rc::as_ptr(&s.data)).expect("Grad S missing");

        assert_eq!(grad_a.data().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(grad_a.shape(), vec![3]);

        let expected_grad_s_val = (0.1 * 1.0) + (0.2 * 2.0) + (0.3 * 3.0);
        assert_eq!(grad_s.data().len(), 1);
        assert!((grad_s.data().to_vec()[0] - expected_grad_s_val).abs() < 1e-6);
        assert_eq!(grad_s.shape(), Vec::<usize>::new());
    }
} 