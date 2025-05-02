use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::ops::arithmetic::mul; // Need mul_op
use crate::ops::arithmetic::neg; // Need neg_op
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
// Add Neg, Sub traits if needed by internal ops
use std::ops::{Add, AddAssign, Div, Mul, Neg};
use std::sync::{Arc, RwLock};

// --- Backward Operation Structure ---

/// Backward operation context for element-wise division (a / b).
/// Stores cloned Tensors of the inputs and their original shapes.
#[derive(Debug)]
struct DivBackward<T: 'static + Debug + Copy + Send + Sync> {
    a: Tensor<T>, // Cloned input tensor a
    b: Tensor<T>, // Cloned input tensor b
    a_shape: Vec<usize>, // Original shape of a
    b_shape: Vec<usize>, // Original shape of b
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for DivBackward<T>
where
    // Add all required traits for div_op, mul_op, neg_op, reduce_sum
    T: Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + AddAssign
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Copy
        + Send
        + Sync
        + 'static
        + PartialEq
        + PartialOrd,
{
    /// Computes gradients for the division operation z = a / b.
    /// grad(a) = grad_output / b
    /// grad(b) = - grad_output * a / (b * b)
    /// Handles broadcasting by summing gradients across broadcasted dimensions.
    /// Returns `NeuraRustError::DivisionByZero` if division by zero occurs during gradient calculation.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = grad_output / b
        // Need to use div_op which handles division by zero.
        let grad_a_unreduced = div_op(grad_output, &self.b)?; // Use cloned b
        let grad_a = grad_a_unreduced.reduce_sum_for_broadcast(&self.a_shape)?;

        // grad_b = - (grad_output * a) / (b * b)
        // Calculate b * b
        let b_squared = mul::mul_op(&self.b, &self.b)?;
        // Calculate grad_output * a
        let grad_times_a = mul::mul_op(grad_output, &self.a)?;
        // Calculate (grad_output * a) / b_squared
        let grad_b_term = div_op(&grad_times_a, &b_squared)?; // div_op handles zero b_squared
        // Calculate -grad_b_term
        let grad_b_unreduced = neg::neg_op(&grad_b_term)?;
        // Reduce gradient for b
        let grad_b = grad_b_unreduced.reduce_sum_for_broadcast(&self.b_shape)?;

        // Ensure gradients are on the correct device
        let expected_device = grad_output.device();
        if grad_a.device() != expected_device || grad_b.device() != expected_device {
            return Err(NeuraRustError::BackwardError(format!(
                "DivBackward gradient device mismatch. Expected {:?}, got grad_a: {:?}, grad_b: {:?}",
                expected_device,
                grad_a.device(),
                grad_b.device()
            )));
        }

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData<T>>> {
        vec![self.a.get_node_id(), self.b.get_node_id()]
    }
}

// --- Forward Operation ---

/// Performs element-wise division for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn div_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    // Add all required bounds for autograd linkage and internal ops
    T: Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
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
    let mut backward_context_maybe: Option<DivBackward<T>> = None;

    if requires_grad {
        // Create context with clones
        backward_context_maybe = Some(DivBackward {
            a: a.clone(),
            b: b.clone(),
            a_shape: a.shape(),
            b_shape: b.shape(),
        });
    }
    // --- End Autograd Setup ---

    // Acquire read locks
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            expected: a_guard.device,
            actual: b_guard.device,
            operation: "div_op".to_string(),
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Division is currently only supported on CPU, not {:?}",
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

    // --- Calculation (with division by zero check) ---
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    let result_strides = calculate_strides(&output_shape);
    let rank_diff_a = output_shape.len().saturating_sub(a_shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_shape.len());
    let mut input_a_coords = vec![0; a_shape.len()];
    let mut input_b_coords = vec![0; b_shape.len()];

    // Optimization: Check for scalar divisor first
    if b_shape.iter().product::<usize>() == 1 && numel_result > 0 {
        let offset_b = b_guard.get_offset(&input_b_coords); // Coords are [0, 0, ...]
        let val_b = b_data_arc[offset_b];
        if val_b == T::zero() {
            return Err(NeuraRustError::DivisionByZero);
        }
        // Iterate only over a
        for i in 0..numel_result {
            let output_coords = index_to_coord(i, &result_strides, &output_shape);
            for dim_idx in 0..a_shape.len() {
                let output_coord_idx = rank_diff_a + dim_idx;
                input_a_coords[dim_idx] = if a_shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
            }
            let offset_a = a_guard.get_offset(&input_a_coords);
            let val_a = a_data_arc[offset_a];
            result_data_vec.push(val_a / val_b);
        }
    } else {
        // General case: iterate over output shape
        for i in 0..numel_result {
            let output_coords = index_to_coord(i, &result_strides, &output_shape);
            // Coords for a
            for dim_idx in 0..a_shape.len() {
                let output_coord_idx = rank_diff_a + dim_idx;
                input_a_coords[dim_idx] = if a_shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
            }
            let offset_a = a_guard.get_offset(&input_a_coords);
            let val_a = a_data_arc[offset_a];
            // Coords for b
            for dim_idx in 0..b_shape.len() {
                let output_coord_idx = rank_diff_b + dim_idx;
                input_b_coords[dim_idx] = if b_shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
            }
            let offset_b = b_guard.get_offset(&input_b_coords);
            let val_b = b_data_arc[offset_b];
            // Check for division by zero
            if val_b == T::zero() {
                return Err(NeuraRustError::DivisionByZero);
            }
            result_data_vec.push(val_a / val_b);
        }
    }

    drop(a_guard);
    drop(b_guard);

    // --- Create Result ---
    let result_tensor = Tensor::new(result_data_vec, output_shape)?;

    // --- Autograd Linkage ---
    if let Some(backward_context) = backward_context_maybe {
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }

    Ok(result_tensor)
}

// --- std::ops::Div implementation ---
impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    // Add all necessary bounds matching div_op
    T: Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
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
    type Output = Result<Tensor<T>, NeuraRustError>;

    fn div(self, other: &'b Tensor<T>) -> Self::Output {
        div_op(self, other)
    }
}

/// REMOVED: In-place DivAssign is generally not provided/meaningful with shared data.
// impl<'a, T> DivAssign<&'a Tensor<T>> for Tensor<T> { ... }

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::Tensor;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use num_traits::{Float, One, Zero, Signed};
    use std::cmp::PartialEq;
    use std::default::Default;
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::ops::{AddAssign, Div, Mul};

    fn create_test_tensor<
        T: Clone
            + Debug
            + PartialEq
            + Zero
            + One
            + AddAssign
            + Add<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + Signed
            + Copy
            + Default
            + Sum
            + PartialOrd
            + Float
            + approx::AbsDiffEq<Epsilon = T>
            + approx::RelativeEq<Epsilon = T>
            + approx::UlpsEq<Epsilon = T>
            + Send
            + Sync
            + 'static,
    >(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0f64, 21.0, 32.0, 45.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![2.0f64, 3.0, 4.0, 5.0], vec![2, 2]);
        let expected_data = vec![5.0f64, 7.0, 8.0, 9.0];
        let expected_shape = vec![2, 2];
        let result = div_op(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");

        let res_trait = (&t1 / &t2).expect("Div trait failed");
        let trait_buffer_arc = res_trait.borrow_data_buffer();
        let trait_cpu_data = trait_buffer_arc.cpu_data().expect("Trait result not on CPU");
         assert_eq!(trait_cpu_data.as_slice(), expected_data.as_slice());
         assert_eq!(res_trait.shape(), expected_shape, "Shape mismatch (trait)");
    }

    #[test]
    fn test_div_broadcasting() {
        let t1 = create_test_tensor(vec![10.0f64, 20.0], vec![1, 2]);
        let t2 = create_test_tensor(vec![2.0f64, 4.0], vec![2, 1]);
        let expected_data = vec![5.0f64, 10.0, 2.5, 5.0];
        let expected_shape = vec![2, 2];
        let result = div_op(&t1, &t2).expect("Broadcasting div failed");
        assert_eq!(result.shape(), expected_shape);
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        let t_mat = create_test_tensor(vec![10.0f64, 20.0, 30.0, 40.0], vec![2, 2]);
        let t_scalar = Tensor::scalar(10.0f64);
        let expected_scalar_div = vec![1.0f64, 2.0, 3.0, 4.0];
        let result_scalar = div_op(&t_mat, &t_scalar).expect("Scalar div failed");
        assert_eq!(result_scalar.shape(), vec![2, 2]);
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc
            .cpu_data()
            .expect("Scalar div result not on CPU");
        assert_eq!(
            scalar_res_cpu_data.as_slice(),
            expected_scalar_div.as_slice()
        );

        let result_scalar_rev = div_op(&t_scalar, &t_mat).expect("Scalar div reverse failed");
        let expected_scalar_div_rev = vec![1.0f64, 0.5, 1.0 / 3.0, 0.25];
        assert_eq!(result_scalar_rev.shape(), vec![2, 2]);
        let scalar_rev_buffer_arc = result_scalar_rev.borrow_data_buffer();
        let scalar_rev_cpu_data = scalar_rev_buffer_arc
            .cpu_data()
            .expect("Scalar div reverse result not on CPU");
        assert_relative_eq!(
            scalar_rev_cpu_data.as_slice(),
            expected_scalar_div_rev.as_slice()
        );
    }

    #[test]
    fn test_div_by_zero() {
        let t1 = create_test_tensor(vec![1.0f64, 2.0], vec![2]);
        let t2 = create_test_tensor(vec![1.0f64, 0.0], vec![2]);
        let result = div_op(&t1, &t2);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::DivisionByZero => {}
            _ => panic!("Expected DivisionByZero error"),
        }

        let t_scalar_zero = Tensor::scalar(0.0f64);
        let result_scalar = div_op(&t1, &t_scalar_zero);
        assert!(result_scalar.is_err());
        match result_scalar.err().unwrap() {
            NeuraRustError::DivisionByZero => {}
            _ => panic!("Expected DivisionByZero error for scalar divisor"),
        }
    }

    #[test]
    fn test_div_backward_simple() {
        let a_init = create_test_tensor(vec![6.0f64, 10.0], vec![2]);
        a_init.set_requires_grad(true).unwrap();
        let a = a_init;

        let b_init = create_test_tensor(vec![2.0f64, 5.0], vec![2]);
        b_init.set_requires_grad(true).unwrap();
        let b = b_init;

        let forward_fn = |inputs: &[Tensor<f64>]| div_op(&inputs[0], &inputs[1]);

        check_grad(
            forward_fn,
            &[a.clone(), b.clone()],
            &Tensor::ones(vec![2]).unwrap(),
            1e-6,
            1e-6,
        )
        .expect("Gradient check failed for simple division");

        let z = div_op(&a, &b).unwrap();
        z.backward(Some(Tensor::ones(z.shape()).unwrap())).expect("Backward pass failed");

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        let expected_grad_a = vec![0.5f64, 0.2];
        let expected_grad_b = vec![-1.5f64, -0.4];

        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();
        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        assert_abs_diff_eq!(grad_a_data.as_slice(), expected_grad_a.as_slice(), epsilon = 1e-6);
        assert_abs_diff_eq!(grad_b_data.as_slice(), expected_grad_b.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_div_backward_broadcast() {
        let a_init = create_test_tensor(vec![10.0f64, 20.0, 30.0, 40.0], vec![2, 2]);
        a_init.set_requires_grad(true).unwrap();
        let a = a_init;

        let b_init = create_test_tensor(vec![2.0f64, 4.0], vec![2, 1]);
        b_init.set_requires_grad(true).unwrap();
        let b = b_init;

        let forward_fn = |inputs: &[Tensor<f64>]| div_op(&inputs[0], &inputs[1]);

        check_grad(
            forward_fn,
            &[a.clone(), b.clone()],
            &Tensor::ones(vec![2, 2]).unwrap(),
            1e-6,
            1e-6,
        )
        .expect("Gradient check failed for broadcast division");

        let z = div_op(&a, &b).unwrap();
        z.backward(Some(Tensor::ones(z.shape()).unwrap())).expect("Backward pass failed");

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        let expected_grad_a = vec![0.5f64, 0.5, 0.25, 0.25];
        let expected_grad_b = vec![-7.5f64, -4.375];

        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();
        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        assert_eq!(grad_a.shape(), vec![2, 2]);
        assert_eq!(grad_b.shape(), vec![2, 1]);

        assert_abs_diff_eq!(grad_a_data.as_slice(), expected_grad_a.as_slice(), epsilon = 1e-6);
        assert_abs_diff_eq!(grad_b_data.as_slice(), expected_grad_b.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_div_backward_with_zero_divisor() {
        let a_init = create_test_tensor(vec![6.0f64, 10.0], vec![2]);
        a_init.set_requires_grad(true).unwrap();
        let a = a_init;

        let b_init = create_test_tensor(vec![2.0f64, 0.0], vec![2]);
        b_init.set_requires_grad(true).unwrap();
        let b = b_init;

        let z_result = div_op(&a, &b);
        assert!(z_result.is_err());
        assert!(matches!(z_result.err().unwrap(), NeuraRustError::DivisionByZero));

        let grad_out_a = Tensor::ones(vec![2]).unwrap();
        let backward_a = DivBackward { a: a.clone(), b: b.clone(), a_shape: a.shape(), b_shape: b.shape() };
        let grad_a_result = backward_a.backward(&grad_out_a);
        assert!(grad_a_result.is_err(), "Expected error for grad(a) due to zero divisor");

        let grad_out_b = Tensor::ones(vec![2]).unwrap();
        let backward_b = DivBackward { a: a.clone(), b: b.clone(), a_shape: a.shape(), b_shape: b.shape() };
        let grad_b_result = backward_b.backward(&grad_out_b);
        assert!(grad_b_result.is_err(), "Expected error for grad(b) due to zero divisor squared");
    }
}
