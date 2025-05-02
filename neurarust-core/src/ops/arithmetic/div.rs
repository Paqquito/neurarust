use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Div, Mul};

// --- Forward Operation ---

/// Performs element-wise division for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
/// This operation creates a new Tensor with copied data on the same device.
pub fn div_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Div<Output = T>
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
        + PartialEq,
{
    // Acquire read locks
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Cannot divide tensors on different devices: {:?} and {:?}",
            a_guard.device, b_guard.device
        )));
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

    // --- Calculation ---
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    let result_strides = calculate_strides(&output_shape);
    let rank_diff_a = output_shape.len().saturating_sub(a_guard.shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_guard.shape.len());
    let mut input_a_coords = vec![0; a_guard.shape.len()];
    let mut input_b_coords = vec![0; b_guard.shape.len()];
    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, &output_shape);
        // Coords for a
        for dim_idx in 0..a_guard.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            // Use guard for shape check
            input_a_coords[dim_idx] = if a_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_a = a_guard.get_offset(&input_a_coords);
        let val_a = a_data_arc[offset_a]; // Use CPU data arc
                                          // Coords for b
        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            // Use guard for shape check
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_b = b_guard.get_offset(&input_b_coords);
        let val_b = b_data_arc[offset_b]; // Use CPU data arc
                                          // Check for division by zero
        if val_b == T::zero() {
            return Err(NeuraRustError::DivisionByZero);
        }
        result_data_vec.push(val_a / val_b); // Division
    }

    // Drop locks
    drop(a_guard);
    drop(b_guard);

    // --- Create Result ---
    Tensor::new(result_data_vec, output_shape.clone())
    // Autograd setup removed
}

// --- std::ops::Div implementation (calls the op function) ---
impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: Div<Output = T>
        + Mul<Output = T>
        + AddAssign
        + Copy
        + Clone
        + 'static
        + Default
        + Debug
        + Zero
        + One
        + Sum
        + PartialEq,
{
    type Output = Result<Tensor<T>, NeuraRustError>; // Output Result

    fn div(self, other: &'b Tensor<T>) -> Self::Output {
        div_op(self, other) // Call the fallible op function
    }
}

/// REMOVED: In-place DivAssign is generally not provided/meaningful with shared data.
// impl<'a, T> DivAssign<&'a Tensor<T>> for Tensor<T> { ... }

// --- Backward Operation (REMOVED for Phase 0) ---
// #[derive(Debug)]
// struct DivBackward<T: 'static> { ... }
// impl<T> BackwardOp<T> for DivBackward<T> { ... }

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;
    use crate::Tensor;
    use approx::assert_relative_eq;
    use num_traits::{One, Zero};
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
            + Copy
            + Div<Output = T>
            + Mul<Output = T>
            + Default
            + Sum,
    >(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    // REMOVED: fn create_test_tensor_with_grad(...)

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0_f32, 21.0, 32.0, 45.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![2.0_f32, 3.0, 4.0, 5.0], vec![2, 2]);
        let expected_data = vec![5.0_f32, 7.0, 8.0, 9.0];
        let expected_shape = vec![2, 2];
        let result = div_op(&t1, &t2);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        // Updated data access
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
    }

    #[test]
    fn test_div_broadcasting() {
        let t1 = create_test_tensor(vec![10.0_f32, 20.0], vec![1, 2]);
        let t2 = create_test_tensor(vec![2.0_f32, 4.0], vec![2, 1]);
        let expected_data = vec![5.0_f32, 10.0, 2.5, 5.0];
        let expected_shape = vec![2, 2];
        let result = div_op(&t1, &t2).expect("Broadcasting div failed");
        assert_eq!(result.shape(), expected_shape);
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        let t_mat = create_test_tensor(vec![10_f32, 20.0, 30.0, 40.0], vec![2, 2]);
        let t_scalar = Tensor::scalar(10.0_f32);
        let expected_scalar_div = vec![1.0_f32, 2.0, 3.0, 4.0];
        let result_scalar = div_op(&t_mat, &t_scalar).expect("Scalar div failed");
        assert_eq!(result_scalar.shape(), vec![2, 2]);
        // Updated data access
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc
            .cpu_data()
            .expect("Scalar div result not on CPU");
        assert_eq!(
            scalar_res_cpu_data.as_slice(),
            expected_scalar_div.as_slice()
        );

        let result_scalar_rev = div_op(&t_scalar, &t_mat).expect("Scalar div reverse failed");
        let expected_scalar_div_rev = vec![1.0_f32, 0.5, 1.0 / 3.0, 0.25];
        assert_eq!(result_scalar_rev.shape(), vec![2, 2]);
        // Updated data access
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
        let t1 = create_test_tensor(vec![1.0_f32, 2.0], vec![2]);
        let t2 = create_test_tensor(vec![1.0_f32, 0.0], vec![2]);
        let result = div_op(&t1, &t2);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::DivisionByZero => {}
            _ => panic!("Expected DivisionByZero error"),
        }

        let t_scalar_zero = Tensor::scalar(0.0_f32);
        let result_scalar = div_op(&t1, &t_scalar_zero);
        assert!(result_scalar.is_err());
        match result_scalar.err().unwrap() {
            NeuraRustError::DivisionByZero => {}
            _ => panic!("Expected DivisionByZero error"),
        }
    }
    // REMOVED: Backward tests
}
