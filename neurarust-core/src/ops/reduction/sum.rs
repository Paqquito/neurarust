use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use std::ops::{AddAssign};
use num_traits::Zero;
use std::fmt::Debug;
use std::iter::Sum;
use std::cmp::PartialEq;
use std::default::Default;
use num_traits::One;
use std::cmp::PartialOrd; // Added for create_test_tensor
use crate::device::StorageDevice;

/// Calculates the sum of elements along specified axes.
/// Requires the tensor to be on CPU.
pub fn sum_axes<T>(
    input: &Tensor<T>,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Clone + Zero + AddAssign + Debug + Copy + Send + Sync + 'static + Default + PartialEq + PartialOrd + One + Sum,
{
    // Acquire read lock
    let input_guard = input.read_data();

    // --- Device Check --- 
    let device = input_guard.device; 
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            format!("Summation is currently only supported on CPU, not {:?}", device)
        ));
    }
    // --- Get CPU Data Buffer --- 
    let input_data_arc = input_guard.data.cpu_data()?.clone(); 

    // --- Shape and Axis Validation --- 
    let input_shape = input_guard.shape.clone();
    let input_rank = input_shape.len();

    // --- Handle Sum All Case --- 
    if axes.is_empty() {
        // Sum all elements using the cpu data arc
        let sum_val = input_data_arc.iter().map(|x| *x).sum::<T>();
        let output_shape = if keep_dims { vec![1; input_rank] } else { vec![] };
        // Result tensor on CPU
        return Tensor::new(vec![sum_val], output_shape);
    }

    // --- Validate Axes --- 
    let mut processed_axes = Vec::with_capacity(axes.len());
    for &axis in axes {
        if axis >= input_rank {
            // Drop guard before returning error (Now safe because input_shape is cloned)
            drop(input_guard);
            return Err(NeuraRustError::IndexOutOfBounds {
                index: vec![axis],
                shape: input_shape, // Use the cloned shape
            });
        }
        processed_axes.push(axis);
    }
    processed_axes.sort_unstable();
    processed_axes.dedup();

    // --- Calculate Output Shape --- 
    let mut output_shape = Vec::new();
    // Use the cloned input_shape here too
    for (dim, &size) in input_shape.iter().enumerate() {
        if !processed_axes.contains(&dim) {
            output_shape.push(size);
        } else if keep_dims {
            output_shape.push(1);
        }
    }
    // Handle edge cases for output shape (sum all with keep_dims, sum to scalar)
    if output_shape.is_empty() && !processed_axes.is_empty() { // Avoid overriding sum-all case handled above
         if keep_dims { output_shape = vec![1; input_rank]; }
         else { output_shape = vec![]; } // Sum to scalar
    }

    // --- Perform Summation --- 
    let output_numel: usize = if output_shape.is_empty() { 1 } else { output_shape.iter().product() };
    let mut result_data = vec![T::zero(); output_numel];

    let mut current_input_indices = vec![0; input_rank];
    let input_strides = &input_guard.strides;
    let input_offset = input_guard.offset;

    // Iterate through all elements of the input tensor
    for _i in 0..input_guard.numel() { // Use guard.numel()
        // Calculate input offset using strides
        let mut current_relative_offset = 0;
        for dim_idx in 0..input_rank {
            current_relative_offset += current_input_indices[dim_idx] * input_strides[dim_idx];
        }
        let logical_offset = input_offset + current_relative_offset;
        // Access value from the cloned CPU data Arc
        let val = input_data_arc[logical_offset];

        // Calculate the corresponding index in the output tensor
        let mut output_indices = Vec::with_capacity(output_shape.len());
        let mut output_idx_pos = 0;
        // Use the cloned input_shape here
        for (dim_idx, &coord) in current_input_indices.iter().enumerate() {
            if !processed_axes.contains(&dim_idx) {
                if output_idx_pos < output_shape.len() {
                    output_indices.push(coord);
                    output_idx_pos += 1;
                }
            } else if keep_dims {
                if output_idx_pos < output_shape.len() {
                    output_indices.push(0); // Index is 0 for kept reduced dimensions
                    output_idx_pos += 1;
                }
            }
        }
         
        // Calculate flat index for result_data
        let mut output_flat_idx = 0;
        if !output_shape.is_empty() { // Avoid index calculation for scalar output
            let mut stride_product = 1;
            for j in (0..output_shape.len()).rev() {
                output_flat_idx += output_indices[j] * stride_product;
                 // Calculate output strides on the fly if needed, or assume contiguity for output
                 if j > 0 { stride_product *= output_shape[j]; }
            }
        } // else output_flat_idx remains 0 for scalar output

        if output_flat_idx < result_data.len() { // Bounds check
             result_data[output_flat_idx] += val;
        }

        // Increment input indices (N-dimensional counter logic)
        if input_guard.numel() > 0 && _i < input_guard.numel() - 1 { // Avoid increment on last element
            let mut dim_to_increment = input_rank;
            while dim_to_increment > 0 {
                dim_to_increment -= 1;
                current_input_indices[dim_to_increment] += 1;
                // Use the cloned input_shape here
                if current_input_indices[dim_to_increment] < input_shape[dim_to_increment] {
                    break; // Successfully incremented
                }
                current_input_indices[dim_to_increment] = 0; // Reset and carry over
            }
        }
    }

    // Drop lock
    drop(input_guard);
    // Create result tensor on CPU
    Tensor::new(result_data, output_shape)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*; 
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::AddAssign;
    use std::fmt::Debug;
    use std::iter::Sum;
    use crate::error::NeuraRustError;
    use approx::assert_relative_eq;
    use std::default::Default;
    use std::cmp::PartialEq;
    use std::cmp::PartialOrd;

    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Clone + Zero + AddAssign + Debug + Copy + Send + Sync + 'static + Default + PartialEq + PartialOrd + One + Sum,
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    #[test]
    fn test_sum_all() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[], false).unwrap();
        assert_eq!(result.shape(), vec![]); // Scalar shape
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_relative_eq!(res_cpu_data[0], 21.0);
    }

    #[test]
    fn test_sum_axis_0() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[0], false).unwrap();
        assert_eq!(result.shape(), vec![3]);
        let expected_data = vec![5.0, 7.0, 9.0];
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_axis_1() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[1], false).unwrap();
        assert_eq!(result.shape(), vec![2]);
        let expected_data = vec![6.0, 15.0];
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
    }
    
    #[test]
    fn test_sum_axes_multiple() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        // Sum over axes 0 and 2
        let result = sum_axes(&t, &[0, 2], false).unwrap();
        assert_eq!(result.shape(), vec![2]);
        let expected_data = vec![14.0, 22.0];
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_keep_dims() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = sum_axes(&t, &[0], true).unwrap();
        assert_eq!(result.shape(), vec![1, 2]);
        let expected_data = vec![4.0, 6.0];
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        let result_all = sum_axes(&t, &[], true).unwrap();
        assert_eq!(result_all.shape(), vec![1, 1]);
        // Updated data access
        let res_all_buffer_arc = result_all.borrow_data_buffer();
        let res_all_cpu_data = res_all_buffer_arc.cpu_data().expect("Result all not on CPU");
        assert_relative_eq!(res_all_cpu_data[0], 10.0);
    }

    #[test]
    fn test_sum_invalid_axis() {
        let t = create_test_tensor(vec![1.0_f32, 2.0], vec![2]);
        let result = sum_axes(&t, &[1], false);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::IndexOutOfBounds { index, shape } => {
                assert_eq!(index, vec![1]);
                assert_eq!(shape, vec![2]);
            }
            _ => panic!("Expected IndexOutOfBounds error"),
        }
    }
} 