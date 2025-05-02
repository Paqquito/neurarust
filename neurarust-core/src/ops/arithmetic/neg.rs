use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Neg};

// --- Forward Operation ---

/// Performs unary negation for a Tensor.
/// Requires the tensor to be on CPU.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn neg_op<T>(a: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Neg<Output = T>
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
    // Acquire read lock
    let a_guard = a.read_data();

    // --- Device Check ---
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Negation is currently only supported on CPU, not {:?}",
            device
        )));
    }
    // --- Get CPU Data Buffer ---
    let a_data_arc = a_guard.data.cpu_data()?.clone();

    // --- Calculation (Handles Strides) ---
    let output_shape = a_guard.shape.clone();
    let numel = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel);

    // Need to iterate respecting strides
    let strides = &a_guard.strides;
    let offset = a_guard.offset;

    if numel > 0 {
        let mut current_coords = vec![0; output_shape.len()];
        for linear_idx in 0..numel {
            // Calculate current logical offset in the original buffer
            let mut relative_offset = 0;
            for i in 0..output_shape.len() {
                relative_offset += current_coords[i] * strides[i];
            }
            let logical_offset = offset + relative_offset;

            // Access data using the cloned Arc<Vec<T>>
            let val_a = a_data_arc[logical_offset];
            result_data_vec.push(-val_a);

            // Increment coordinates (standard n-dimensional iteration logic)
            if linear_idx < numel - 1 {
                // Avoid incrementing after the last element
                let mut dim_to_inc = output_shape.len() - 1;
                loop {
                    current_coords[dim_to_inc] += 1;
                    if current_coords[dim_to_inc] < output_shape[dim_to_inc] {
                        break; // Finished incrementing this dimension
                    }
                    current_coords[dim_to_inc] = 0; // Reset current dim, carry over
                    if dim_to_inc == 0 {
                        break; // Finished all increments (should not happen if linear_idx < numel - 1)
                    }
                    dim_to_inc -= 1;
                }
            }
        }
    }

    // Drop lock
    drop(a_guard);

    // --- Create Result ---
    Tensor::new(result_data_vec, output_shape)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use num_traits::{One, Zero};
    use std::cmp::PartialEq;
    use std::default::Default;
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::ops::{AddAssign, Neg};

    fn create_test_tensor<
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Neg<Output = T> + Default + Sum,
    >(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    #[test]
    fn test_neg_ok() {
        let t1 = create_test_tensor(vec![1.0_f32, -2.0, 3.0, -4.0], vec![2, 2]);
        let expected_data = vec![-1.0_f32, 2.0, -3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = neg_op(&t1);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        // Updated data access
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
    }

    // Add test for non-contiguous tensor
    /* // TODO: Re-enable this test when Tensor::transpose and Tensor::is_contiguous are implemented (Phase 1.1)
    #[test]
    fn test_neg_non_contiguous() {
         // Create a tensor, transpose it to make it non-contiguous
         let t_orig = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
         let t_transposed = t_orig.transpose(0, 1).expect("Transpose failed"); // [[1, 3], [2, 4]]

         let expected_data = vec![-1.0_f32, -3.0, -2.0, -4.0]; // Negated transposed data
         let expected_shape = vec![2, 2];

         let result = neg_op(&t_transposed);
         assert!(result.is_ok(), "neg_op failed: {:?}", result.err());
         let res_tensor = result.unwrap();

         assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");

         // Check result data (must be contiguous now)
         let res_buffer_arc = res_tensor.borrow_data_buffer();
         let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
         assert!(res_tensor.is_contiguous(), "Result tensor should be contiguous");
         assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice(), "Data mismatch");
     }
     */
}
