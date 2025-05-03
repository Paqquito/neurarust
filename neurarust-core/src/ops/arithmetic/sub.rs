use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Neg, Sub};
use std::sync::{Arc, RwLockReadGuard};

// --- Backward Operation Structure ---

/// Backward operation context for subtraction (a - b).
/// Stores cloned Tensors of the inputs and their original shapes.
/// Cloning the Tensor (which clones the internal Arc<RwLock<TensorData>>) is Send + Sync.
#[derive(Debug)]
struct SubBackward<T: 'static + Debug + Copy + Send + Sync> {
    a: Tensor<T>,
    b: Tensor<T>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + Zero
        + AddAssign
        + 'static
        + Default
        + PartialEq
        + Sum
        + One
        + PartialOrd
        + Neg<Output = T>,
{
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.a.get_node_id(), self.b.get_node_id()]
    }

    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = grad_output * 1
        let grad_a = reduce_gradient_to_shape(grad_output, &self.a_shape)?;

        // grad_b = grad_output * (-1)
        let neg_grad_output = crate::ops::arithmetic::neg_op(grad_output)?;
        let grad_b = reduce_gradient_to_shape(&neg_grad_output, &self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}

// --- Kernel de Calcul ---

/// Noyau de calcul privé pour la soustraction élément par élément avec broadcasting.
fn sub_kernel<T>(
    a_guard: &RwLockReadGuard<'_, TensorData<T>>,
    b_guard: &RwLockReadGuard<'_, TensorData<T>>,
    a_data_slice: &[T],
    b_data_slice: &[T],
    output_shape: &[usize],
) -> Result<Vec<T>, NeuraRustError>
where
    T: Sub<Output = T> + Copy + Debug,
{
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    let result_strides = calculate_strides(output_shape);

    let rank_diff_a = output_shape.len().saturating_sub(a_guard.shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_guard.shape.len());

    let mut input_a_coords = vec![0; a_guard.shape.len()];
    let mut input_b_coords = vec![0; b_guard.shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, output_shape);

        // Indice pour A
        for dim_idx in 0..a_guard.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_a = a_guard.get_offset(&input_a_coords);
        let val_a = a_data_slice[offset_a];

        // Indice pour B
        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_b = b_guard.get_offset(&input_b_coords);
        let val_b = b_data_slice[offset_b];

        result_data_vec.push(val_a - val_b);
    }

    Ok(result_data_vec)
}

// --- Forward Operation ---

pub fn sub_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Sub<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + 'static
        + PartialEq
        + PartialOrd
        + Send
        + Sync,
{
    // --- Autograd Setup ---
    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut a_maybe_clone: Option<Tensor<T>> = None;
    let mut b_maybe_clone: Option<Tensor<T>> = None;
    let mut a_shape_maybe: Option<Vec<usize>> = None;
    let mut b_shape_maybe: Option<Vec<usize>> = None;
    if requires_grad {
        a_maybe_clone = Some(a.clone());
        b_maybe_clone = Some(b.clone());
        a_shape_maybe = Some(a.shape());
        b_shape_maybe = Some(b.shape());
    }

    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Cannot subtract tensors on different devices: {:?} and {:?}",
            a_guard.device, b_guard.device
        )));
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Subtraction is currently only supported on CPU, not {:?}",
            device
        )));
    }

    let a_data_arc = a_guard.data.cpu_data()?.clone();
    let b_data_arc = b_guard.data.cpu_data()?.clone();
    let a_data_slice = a_data_arc.as_slice();
    let b_data_slice = b_data_arc.as_slice();

    // --- Shape and Broadcasting ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    let output_shape = broadcast_shapes(a_shape, b_shape).map_err(|_e| {
        NeuraRustError::BroadcastError {
            shape1: a_shape.clone(),
            shape2: b_shape.clone(),
        }
    })?;

    // --- Calculation (Appel au Kernel) ---
    let result_data_vec = sub_kernel(
        &a_guard,
        &b_guard,
        a_data_slice,
        b_data_slice,
        &output_shape,
    )?;

    // Drop read locks
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data_vec, output_shape.clone())?;

    // --- Autograd Linkage (The General Pattern) ---
    if requires_grad {
        let backward_context = SubBackward {
            a: a_maybe_clone.unwrap(),
            b: b_maybe_clone.unwrap(),
            a_shape: a_shape_maybe.unwrap(),
            b_shape: b_shape_maybe.unwrap(),
        };
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }

    Ok(result_tensor)
}

/// REMOVED: In-place SubAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;
    use crate::utils::testing::{create_test_tensor, create_test_tensor_with_grad};

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let expected_data = vec![9.0f32, 18.0, 27.0, 36.0];
        let expected_shape = vec![2, 2];

        let result = sub_op(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();

        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(
            res_cpu_data.as_slice(),
            expected_data.as_slice(),
            "Data mismatch"
        );
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
    }

    #[test]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

        let result = sub_op(&t1, &t_non_broadcast);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            NeuraRustError::BroadcastError { .. }
        ));
    }

    #[test]
    fn test_sub_broadcasting() {
        let t1 = create_test_tensor(vec![10.0f32, 20.0], vec![1, 2]);
        let t2 = create_test_tensor(vec![1.0f32, 2.0], vec![2, 1]);
        let expected_data = vec![9.0f32, 19.0, 8.0, 18.0];
        let expected_shape = vec![2, 2];

        let result = sub_op(&t1, &t2).expect("Broadcasting sub failed");
        assert_eq!(result.shape(), expected_shape);
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        let t_mat = create_test_tensor(vec![10.0f32, 20.0], vec![2, 1]);
        let t_scalar = Tensor::scalar(3.0_f32);
        let expected_scalar_sub = vec![7.0f32, 17.0];
        let result_scalar = sub_op(&t_mat, &t_scalar).expect("Scalar sub failed");
        assert_eq!(result_scalar.shape(), vec![2, 1]);
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc
            .cpu_data()
            .expect("Scalar sub result not on CPU");
        assert_eq!(
            scalar_res_cpu_data.as_slice(),
            expected_scalar_sub.as_slice()
        );

        let expected_scalar_sub_rev = vec![-7.0_f32, -17.0];
        let result_scalar_rev = sub_op(&t_scalar, &t_mat).expect("Scalar sub reverse failed");
        assert_eq!(result_scalar_rev.shape(), vec![2, 1]);
        let scalar_rev_res_buffer_arc = result_scalar_rev.borrow_data_buffer();
        let scalar_rev_res_cpu_data = scalar_rev_res_buffer_arc
            .cpu_data()
            .expect("Scalar sub reverse result not on CPU");
        assert_eq!(
            scalar_rev_res_cpu_data.as_slice(),
            expected_scalar_sub_rev.as_slice()
        );
    }

    #[test]
    fn test_sub_backward_simple() {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
        let b = create_test_tensor_with_grad::<f32>(vec![4.0, 5.0, 6.0], vec![3]);
        let output_grad = Tensor::<f32>::ones(vec![3]).unwrap();

        // Check analytical gradients
        let c = sub_op(&a, &b).unwrap();
        c.backward(Some(output_grad.clone())).unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        let expected_grad_a = vec![1.0f32, 1.0, 1.0];
        let expected_grad_b = vec![-1.0f32, -1.0, -1.0];

        // Bind buffer data to local variables to extend lifetime
        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();

        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        for (i, &val) in grad_a_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_a[i], epsilon = 1e-6);
        }
         for (i, &val) in grad_b_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_b[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sub_backward_broadcast() {
        let a = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2, 1]);
        let b = create_test_tensor_with_grad::<f32>(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let output_grad = Tensor::<f32>::ones(vec![2, 3]).unwrap();

        // Check analytical gradients
        let c = sub_op(&a, &b).unwrap();
        c.backward(Some(output_grad.clone())).unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        // Expected grad for 'a': sum(output_grad) over axis 1 -> shape [2, 1]
        let expected_grad_a = vec![3.0f32, 3.0]; // [1+1+1, 1+1+1]
        // Expected grad for 'b': -sum(output_grad) over axis 0 -> shape [1, 3]
        let expected_grad_b = vec![-2.0f32, -2.0, -2.0]; // [- (1+1), - (1+1), - (1+1)]

        // Bind buffer data to local variables
        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();

        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        assert_eq!(grad_a.shape(), vec![2, 1]);
        assert_eq!(grad_b.shape(), vec![1, 3]);

        for (i, &val) in grad_a_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_a[i], epsilon = 1e-6);
        }
         for (i, &val) in grad_b_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_b[i], epsilon = 1e-6);
        }
    }
}

// Déplacer reduce_gradient_to_shape ici, avant son utilisation dans BackwardOp
/// Helper function to reduce a gradient tensor to a target shape.
fn reduce_gradient_to_shape<T>(
    gradient: &Tensor<T>,
    target_shape: &[usize],
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + Zero
        + AddAssign
        + 'static
        + Default
        + PartialEq
        + std::iter::Sum
        + num_traits::One
        + PartialOrd,
{
    if gradient.shape() == target_shape {
        Ok(gradient.clone())
    } else {
        if target_shape.is_empty() {
            crate::ops::reduction::sum_axes(gradient, &[], false)
        } else {
            let current_shape = gradient.shape();
            let rank_diff = current_shape.len().saturating_sub(target_shape.len());
            let mut axes_to_reduce: Vec<usize> = (0..rank_diff).collect();

            for i in 0..target_shape.len() {
                if target_shape[i] == 1 && current_shape[rank_diff + i] > 1 {
                    axes_to_reduce.push(rank_diff + i);
                } else if target_shape[i] != current_shape[rank_diff + i] && target_shape[i] != 1 {
                    return Err(NeuraRustError::InternalError(format!(
                        "Cannot reduce gradient shape {:?} to {:?}: Incompatible dimensions found.",
                        current_shape, target_shape
                    )));
                }
            }

            if axes_to_reduce.is_empty() {
                if current_shape == target_shape {
                    Ok(gradient.clone())
                } else {
                    return Err(NeuraRustError::InternalError(format!(
                         "Cannot reduce gradient shape {:?} to {:?}: No reduction axes found but shapes differ.",
                         current_shape, target_shape
                      )));
                }
            } else {
                let reduced_grad =
                    crate::ops::reduction::sum_axes(gradient, &axes_to_reduce, false)?;
                let final_shape: Vec<usize> = target_shape.to_vec();
                let reduced_numel: usize = reduced_grad.shape().iter().product();
                let target_numel: usize = target_shape.iter().product();
                if reduced_numel != target_numel {
                    return Err(NeuraRustError::InternalError(format!(
                         "Gradient reduction produced incompatible shape {:?} (numel {}) for target {:?} (numel {}). Reduction axes: {:?}.",
                         reduced_grad.shape(), reduced_numel, target_shape, target_numel, axes_to_reduce
                     )));
                }
                crate::ops::view::reshape_op(&reduced_grad, final_shape)
            }
        }
    }
}
