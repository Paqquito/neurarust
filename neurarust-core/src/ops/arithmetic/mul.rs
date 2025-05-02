use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
// Need direct import for TensorData usage in BackwardOp
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
// Add Add trait needed by check_grad/acc_grad, PartialOrd for reduce_sum
use std::ops::{Add, AddAssign, Mul};
use std::sync::{Arc, RwLock};
use std::cmp::PartialOrd;

// --- Backward Operation Structure ---

/// Backward operation context for element-wise multiplication (a * b).
/// Stores cloned Tensors of the inputs and their original shapes.
#[derive(Debug)]
struct MulBackward<T: 'static + Debug + Copy + Send + Sync> {
    a: Tensor<T>, // Cloned input tensor a
    b: Tensor<T>, // Cloned input tensor b
    a_shape: Vec<usize>, // Original shape of a
    b_shape: Vec<usize>, // Original shape of b
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for MulBackward<T>
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
        + Copy
        + Send
        + Sync
        + 'static
        + PartialEq
        + PartialOrd,
{
    /// Computes gradients for the multiplication operation z = a * b.
    /// grad(a) = grad_output * b
    /// grad(b) = grad_output * a
    /// Handles broadcasting by summing gradients across broadcasted dimensions.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = grad_output * b (element-wise)
        // The result needs to be reduced to the original shape of a if broadcasting occurred.
        let grad_a_unreduced = mul_op(grad_output, &self.b)?; // Use the cloned b
        let grad_a = grad_a_unreduced.reduce_sum_for_broadcast(&self.a_shape)?;

        // grad_b = grad_output * a (element-wise)
        // The result needs to be reduced to the original shape of b if broadcasting occurred.
        let grad_b_unreduced = mul_op(grad_output, &self.a)?; // Use the cloned a
        let grad_b = grad_b_unreduced.reduce_sum_for_broadcast(&self.b_shape)?;

        // Ensure gradients are on the correct device
        let expected_device = grad_output.device();
        if grad_a.device() != expected_device || grad_b.device() != expected_device {
            return Err(NeuraRustError::BackwardError(format!(
                "MulBackward gradient device mismatch. Expected {:?}, got grad_a: {:?}, grad_b: {:?}",
                expected_device,
                grad_a.device(),
                grad_b.device()
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

/// Performs element-wise multiplication (Hadamard product) for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
/// This operation creates a new Tensor with copied data on the same device.
pub fn mul_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Mul<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    // --- Autograd Setup ---
    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut backward_context_maybe: Option<MulBackward<T>> = None;

    if requires_grad {
        backward_context_maybe = Some(MulBackward {
            a: a.clone(),
            b: b.clone(),
            a_shape: a.shape(),
            b_shape: b.shape(),
        });
    }
    // --- End Autograd Setup ---

    // Acquire read locks for inputs
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
         // Use DeviceMismatch error
        return Err(NeuraRustError::DeviceMismatch {
            expected: a_guard.device,
            actual: b_guard.device,
            operation: "mul_op".to_string(),
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Multiplication op is currently only supported on CPU, not {:?}",
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

        result_data_vec.push(val_a * val_b);
    }

    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data_vec, output_shape)?;

    // --- Autograd Linkage ---
    if let Some(backward_context) = backward_context_maybe {
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(result_tensor)
}

// --- std::ops::Mul implementation (calls the op function) ---
impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Mul<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    type Output = Result<Tensor<T>, NeuraRustError>; // Output Result

    fn mul(self, other: &'b Tensor<T>) -> Self::Output {
        mul_op(self, other) // Call the fallible op function
    }
}

/// REMOVED: In-place MulAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.
// impl<'a, T> MulAssign<&'a Tensor<T>> for Tensor<T>
// where
//     T: MulAssign + Copy + Clone,
// {
//     fn mul_assign(&mut self, other: &'a Tensor<T>) { ... }
// }

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*; // Import mul_op
    // Import grad_check and helpers
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::Tensor;
    use approx::assert_abs_diff_eq;
    use num_traits::{Float, One, Zero};
    use std::cmp::PartialEq;
    use std::default::Default;
    use std::fmt::Debug;
    use std::iter::Sum;
    // Remove unused Neg, Sub imports from tests
    use std::ops::{Add, AddAssign, Mul};

    // Update helpers bounds to include Float and approx traits for grad_check
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
            + Copy
            + Default
            + Sum
            + PartialOrd
            + Float // Requires f32 or f64
            + approx::AbsDiffEq<Epsilon = T>
            + approx::RelativeEq<Epsilon = T>
            + approx::UlpsEq<Epsilon = T>
            + Send
            + Sync
            + 'static,
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    // --- Forward Pass Tests --- 
    #[test]
    fn test_mul_tensors_ok() {
        let t1 = create_test_tensor(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![5.0f64, 6.0, 7.0, 8.0], vec![2, 2]);
        let expected_data = vec![5.0f64, 12.0, 21.0, 32.0];
        let expected_shape = vec![2, 2];

        let result = mul_op(&t1, &t2);
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

        let res_trait = (&t1 * &t2).expect("Mul trait failed");
        let trait_res_buffer_arc = res_trait.borrow_data_buffer();
        let trait_res_cpu_data = trait_res_buffer_arc
            .cpu_data()
            .expect("Trait result tensor not on CPU");
        assert_eq!(
            trait_res_cpu_data.as_slice(),
            expected_data.as_slice(),
            "Data mismatch (trait)"
        );
        assert_eq!(res_trait.shape(), expected_shape, "Shape mismatch (trait)");
    }

    #[test]
    fn test_mul_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5.0f64, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

        let result = mul_op(&t1, &t_non_broadcast);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            NeuraRustError::BroadcastError { .. }
        ));
    }

    #[test]
    fn test_mul_broadcasting() {
        let t1 = create_test_tensor(vec![2.0f64, 3.0], vec![1, 2]);
        let t2 = create_test_tensor(vec![10.0f64, 20.0], vec![2, 1]);
        let expected_data = vec![20.0f64, 30.0, 40.0, 60.0];
        let expected_shape = vec![2, 2];

        let result = mul_op(&t1, &t2).expect("Broadcasting mul failed");
        assert_eq!(result.shape(), expected_shape);
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        let t_mat = create_test_tensor(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_scalar = Tensor::scalar(10.0f64);
        let expected_scalar_mul = vec![10.0f64, 20.0, 30.0, 40.0];
        let result_scalar = mul_op(&t_mat, &t_scalar).expect("Scalar mul failed");
        assert_eq!(result_scalar.shape(), vec![2, 2]);
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc
            .cpu_data()
            .expect("Scalar mul result not on CPU");
        assert_eq!(
            scalar_res_cpu_data.as_slice(),
            expected_scalar_mul.as_slice()
        );

        let result_scalar_rev = mul_op(&t_scalar, &t_mat).expect("Scalar mul reverse failed");
        assert_eq!(result_scalar_rev.shape(), vec![2, 2]);
        let scalar_rev_res_buffer_arc = result_scalar_rev.borrow_data_buffer();
        let scalar_rev_res_cpu_data = scalar_rev_res_buffer_arc
            .cpu_data()
            .expect("Scalar mul reverse result not on CPU");
        assert_eq!(
            scalar_rev_res_cpu_data.as_slice(),
            expected_scalar_mul.as_slice()
        );
    }

    // --- Autograd Tests --- 

    #[test]
    fn test_mul_backward_simple() {
        let a_init = create_test_tensor(vec![1.0f64, 2.0, 3.0], vec![3]);
        a_init.set_requires_grad(true).unwrap();
        let a = a_init;

        let b_init = create_test_tensor(vec![4.0f64, 5.0, 6.0], vec![3]);
        b_init.set_requires_grad(true).unwrap();
        let b = b_init;

        let forward_fn = |inputs: &[Tensor<f64>]| mul_op(&inputs[0], &inputs[1]);

        check_grad(
            forward_fn,
            &[a.clone(), b.clone()],
            &Tensor::ones(vec![3]).unwrap(),
            1e-6,
            1e-6,
        )
        .expect("Gradient check failed for simple multiplication");

        let z = mul_op(&a, &b).unwrap();
        z.backward(Some(Tensor::ones(z.shape()).unwrap())).expect("Backward pass failed");

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        let expected_grad_a = vec![4.0f64, 5.0, 6.0];
        let expected_grad_b = vec![1.0f64, 2.0, 3.0];

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
    fn test_mul_backward_broadcast() {
        let a_init = create_test_tensor(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        a_init.set_requires_grad(true).unwrap();
        let a = a_init;

        let b_init = create_test_tensor(vec![10.0f64, 20.0, 30.0], vec![3]);
        b_init.set_requires_grad(true).unwrap();
        let b = b_init;

        let forward_fn = |inputs: &[Tensor<f64>]| mul_op(&inputs[0], &inputs[1]);

        check_grad(
            forward_fn,
            &[a.clone(), b.clone()],
            &Tensor::ones(vec![2, 3]).unwrap(),
            1e-6,
            1e-6,
        )
        .expect("Gradient check failed for broadcast multiplication");

        let z = mul_op(&a, &b).unwrap();
        z.backward(Some(Tensor::ones(z.shape()).unwrap())).expect("Backward pass failed");

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        let expected_grad_a = vec![10.0f64, 20.0, 30.0, 10.0, 20.0, 30.0];
        let expected_grad_b = vec![5.0f64, 7.0, 9.0];

        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();
        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        assert_eq!(grad_a.shape(), vec![2, 3]);
        assert_eq!(grad_b.shape(), vec![3]);

        for (i, &val) in grad_a_data.iter().enumerate() {
            assert_abs_diff_eq!(val, expected_grad_a[i], epsilon = 1e-6);
        }
        for (i, &val) in grad_b_data.iter().enumerate() {
            assert_abs_diff_eq!(val, expected_grad_b[i], epsilon = 1e-6);
        }
    }
}
