use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};
use std::sync::{Arc, RwLock};

// --- Backward Operation Structure ---

/// Backward operation context for subtraction (a - b).
/// Stores cloned Tensors of the inputs and their original shapes.
/// Cloning the Tensor (which clones the internal Arc<RwLock<TensorData>>) is Send + Sync.
#[derive(Debug)]
struct SubBackward<T: 'static + Debug + Copy + Send + Sync> {
    // Store cloned Tensors instead of NodeIds
    a: Tensor<T>,
    b: Tensor<T>,
    // Keep shapes for broadcasting reduction
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + AddAssign
        + Add<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + Copy
        + Send
        + Sync
        + 'static
        + PartialEq
        + PartialOrd, // Added PartialOrd for reduce_sum_for_broadcast
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = grad_output (reduced if broadcasted)
        let grad_a = grad_output.reduce_sum_for_broadcast(&self.a_shape)?;

        // grad_b = -grad_output (reduced if broadcasted)
        let neg_grad_output = crate::ops::arithmetic::neg::neg_op(grad_output)?;
        let grad_b = neg_grad_output.reduce_sum_for_broadcast(&self.b_shape)?;

        // Ensure gradients are on the correct device
        let expected_device = grad_output.device(); // expected_device is StorageDevice
        if grad_a.device() != expected_device || grad_b.device() != expected_device {
            // Compare devices correctly
            return Err(NeuraRustError::BackwardError(format!(
                "SubBackward gradient device mismatch. Expected {:?}, got grad_a: {:?}, grad_b: {:?}",
                expected_device, grad_a.device(), grad_b.device()
            )));
        }

        Ok(vec![grad_a, grad_b]) // Return gradients in order [grad_a, grad_b]
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData<T>>> {
        // Return NodeIds obtained from the stored tensors
        vec![self.a.get_node_id(), self.b.get_node_id()]
    }
}

// --- Forward Operation ---

pub fn sub_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Sub<Output = T>
        + Add<Output = T>
        + Neg<Output = T>
        + Mul<Output = T>
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + AddAssign
        + 'static
        + PartialEq
        + PartialOrd
        + Send
        + Sync,
{
    // --- Autograd Setup ---
    let requires_grad = a.requires_grad() || b.requires_grad();
    // Store cloned tensors and shapes if grad is required
    let mut a_maybe_clone: Option<Tensor<T>> = None;
    let mut b_maybe_clone: Option<Tensor<T>> = None;
    let mut a_shape_maybe: Option<Vec<usize>> = None;
    let mut b_shape_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        a_maybe_clone = Some(a.clone()); // Clone the tensor (Arc clone)
        b_maybe_clone = Some(b.clone()); // Clone the tensor (Arc clone)
        a_shape_maybe = Some(a.shape()); // Get shape from original tensor
        b_shape_maybe = Some(b.shape()); // Get shape from original tensor
    }
    // --- End Autograd Setup ---

    // Acquire read locks for inputs
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            expected: a_guard.device,
            actual: b_guard.device,
            operation: "sub_op".to_string(),
        });
    }
    let device = a_guard.device;
    // For now, only CPU is supported for the operation itself
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Subtraction op is currently only supported on CPU, not {:?}",
            device
        )));
    }

    // --- Get CPU Data Buffers ---
    let a_data_arc = a_guard.data.cpu_data()?.clone();
    let b_data_arc = b_guard.data.cpu_data()?.clone();

    // --- Shape and Broadcasting ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    let output_shape =
        broadcast_shapes(a_shape, b_shape).map_err(|_e| NeuraRustError::BroadcastError {
            shape1: a_shape.clone(),
            shape2: b_shape.clone(),
        })?;

    // --- Calculation ---
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);

    let result_strides = calculate_strides(&output_shape);
    let rank_diff_a = output_shape.len().saturating_sub(a_shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_shape.len());

    let mut input_a_coords = vec![0; a_shape.len()];
    let mut input_b_coords = vec![0; b_shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, &output_shape);

        // Calculate index for a
        for dim_idx in 0..a_shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_a = a_guard.get_offset(&input_a_coords);
        let val_a = a_data_arc[offset_a];

        // Calculate index for b
        for dim_idx in 0..b_shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_b = b_guard.get_offset(&input_b_coords);
        let val_b = b_data_arc[offset_b];

        result_data_vec.push(val_a - val_b);
    }

    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data_vec, output_shape)?;

    // --- Autograd Linkage ---
    if requires_grad {
        let backward_context = SubBackward {
            // Pass the cloned tensors
            a: a_maybe_clone.unwrap(), // Safe due to requires_grad check
            b: b_maybe_clone.unwrap(), // Safe
            // Pass the captured shapes
            a_shape: a_shape_maybe.unwrap(),
            b_shape: b_shape_maybe.unwrap(),
        };
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(result_tensor)
}

/// REMOVED: In-place SubAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;
    use crate::Tensor;
    use num_traits::{One, Zero};
    use std::cmp::PartialEq;
    use std::default::Default;
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::ops::{AddAssign, Sub};
    use approx::assert_abs_diff_eq;

    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
        T: Clone
            + Debug
            + PartialEq
            + Zero
            + One
            + AddAssign
            + Add<Output = T>
            + Mul<Output = T>
            + Neg<Output = T>
            + Sub<Output = T>
            + Default
            + Sum
            + PartialOrd
            + num_traits::Float
            + approx::AbsDiffEq<Epsilon = T>
            + approx::RelativeEq<Epsilon = T>
            + approx::UlpsEq<Epsilon = T>
            + Send
            + Sync
            + 'static,
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
        T: Clone
            + Debug
            + PartialEq
            + Zero
            + One
            + AddAssign
            + Copy
            + Add<Output = T>
            + Default
            + Sum
            + PartialOrd
            + Send
            + Sync
            + 'static,
    {
        let tensor = Tensor::new(data, shape).unwrap();
        tensor.set_requires_grad(true).unwrap();
        tensor
    }

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
            assert_abs_diff_eq!(val, expected_grad_a[i], epsilon = 1e-6);
        }
         for (i, &val) in grad_b_data.iter().enumerate() {
            assert_abs_diff_eq!(val, expected_grad_b[i], epsilon = 1e-6);
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
            assert_abs_diff_eq!(val, expected_grad_a[i], epsilon = 1e-6);
        }
         for (i, &val) in grad_b_data.iter().enumerate() {
            assert_abs_diff_eq!(val, expected_grad_b[i], epsilon = 1e-6);
        }
    }
}
