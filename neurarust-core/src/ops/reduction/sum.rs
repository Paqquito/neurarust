use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
// use crate::buffer::Buffer; // Removed unused import
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::One;
use num_traits::Zero;
use std::cmp::PartialEq;
use std::cmp::PartialOrd; // Added for create_test_tensor
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use std::sync::RwLock;

/// Backward operation context for `sum_axes`.
/// Stores information needed to compute the gradient for the input tensor.
#[derive(Debug)]
struct SumAxesBackward<T: Debug + Copy + Send + Sync + 'static> {
    input_node: Arc<RwLock<TensorData<T>>>,
    input_shape: Vec<usize>, // Original shape of the input tensor
    axes: Vec<usize>,     // Axes along which summation was performed
    keep_dims: bool,      // Whether dims were kept in the output
}

// --- BackwardOp Implementation for SumAxesBackward ---

impl<T> BackwardOp<T> for SumAxesBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Zero
        + One
        + AddAssign
        + Default
        + PartialEq
        + Sum
        + Add<Output = T>
        + PartialOrd,
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let target_shape = self.input_shape.clone();
        let target_rank = target_shape.len();

        // 1. Prepare the grad_output: Reshape if keep_dims was false to match rank.
        let grad_output_reshaped = if !self.keep_dims && !self.axes.is_empty() {
            let mut shape_with_kept_dims = Vec::with_capacity(target_rank);
            let mut current_grad_dim = 0;
            for i in 0..target_rank {
                if self.axes.contains(&i) {
                    shape_with_kept_dims.push(1);
                } else {
                    // Ensure grad_output shape matches non-reduced dims
                    if current_grad_dim >= grad_output.shape().len()
                        || grad_output.shape()[current_grad_dim] != target_shape[i]
                    {
                        return Err(NeuraRustError::ShapeMismatch {
                            expected: target_shape.clone(), // Or a more specific expected shape
                            actual: grad_output.shape(),
                            operation: "sum_axes_backward (reshape)".to_string(),
                        });
                    }
                    shape_with_kept_dims.push(target_shape[i]);
                    current_grad_dim += 1;
                }
            }
            crate::ops::view::reshape_op(grad_output, shape_with_kept_dims)?
        } else if self.axes.is_empty() && !self.keep_dims {
            // Special case: Summing all elements without keep_dims -> scalar grad_output.
            // Reshape the scalar grad_output to have the rank of the input, with all dimensions as 1.
            let shape_all_ones = vec![1; target_rank];
            crate::ops::view::reshape_op(grad_output, shape_all_ones)?
        } else {
            // If keep_dims was true or axes was empty (sum all with keep_dims), grad_output already has the correct rank.
            grad_output.clone()
        };

        // 2. Ensure Contiguous Gradient for Expansion
        // Manually create a contiguous copy if the reshaped tensor might not be.
        let grad_output_for_expand = {
            let reshaped_guard = grad_output_reshaped.read_data();
            if reshaped_guard.is_contiguous() {
                grad_output_reshaped.clone()
            } else {
                // Not contiguous, create a new contiguous tensor by copying data
                let shape = reshaped_guard.shape.clone();
                let numel = shape.iter().product::<usize>();
                let mut new_data = Vec::with_capacity(numel);
                // Assuming CPU
                let buffer_arc = reshaped_guard.data.cpu_data()?.clone();
                let data_slice = buffer_arc.as_slice();
                let mut current_indices = vec![0; shape.len()];
                for _ in 0..numel {
                    let offset = reshaped_guard.get_offset(&current_indices); // No ? needed
                    new_data.push(data_slice[offset]);
                    // Increment indices
                    if numel > 0 {
                         let mut dim_to_increment = shape.len();
                         while dim_to_increment > 0 {
                             dim_to_increment -= 1;
                             current_indices[dim_to_increment] += 1;
                             if current_indices[dim_to_increment] < shape[dim_to_increment] {
                                 break;
                             }
                             current_indices[dim_to_increment] = 0;
                         }
                     }
                }
                Tensor::new(new_data, shape)?
            }
        };

        // 3. Expand the contiguous grad_output_for_expand to the target shape.
        if grad_output_for_expand.shape() == target_shape {
            return Ok(vec![grad_output_for_expand]);
        }

        // --- Manual Expand Implementation (Corrected Logic) ---
        let grad_output_guard = grad_output_for_expand.read_data();
        let grad_output_buffer = grad_output_guard.data.cpu_data()?.clone();
        let grad_output_shape = grad_output_guard.shape.clone();
        let grad_output_strides = grad_output_guard.strides.clone();
        let grad_output_offset = grad_output_guard.offset;
        let grad_rank = grad_output_shape.len(); // Rank of the source gradient

        // Check ranks match (should after reshape)
        if target_rank != grad_rank {
             return Err(NeuraRustError::InternalError(format!(
                 "Internal Error in SumBackward: Rank mismatch before expansion. Target: {}, Grad: {}",
                 target_rank, grad_rank
             )));
        }

        let input_grad_numel = target_shape.iter().product::<usize>();
        let mut input_grad_data = vec![T::zero(); input_grad_numel];
        let target_strides = crate::tensor_data::TensorData::<T>::calculate_contiguous_strides(&target_shape);
        let mut current_target_indices = vec![0; target_rank];

        for _target_linear_idx in 0..input_grad_numel {
            // Calculate source offset based on target indices and broadcasting rules
            let mut grad_output_relative_offset = 0;
            for dim_idx in 0..target_rank { // Iterate dimensions
                let source_index_for_dim = if grad_output_shape[dim_idx] == 1 && target_shape[dim_idx] > 1 {
                    0 // Dimension was broadcasted, use index 0
                } else {
                    current_target_indices[dim_idx] // Dimension matched, use target index
                };
                grad_output_relative_offset += source_index_for_dim * grad_output_strides[dim_idx];
            }
            let grad_output_logical_offset = grad_output_offset + grad_output_relative_offset;
            let val = grad_output_buffer[grad_output_logical_offset];

            // Calculate target offset
            let mut input_grad_flat_idx = 0;
            for dim_idx in 0..target_rank {
                input_grad_flat_idx += current_target_indices[dim_idx] * target_strides[dim_idx];
            }
            input_grad_data[input_grad_flat_idx] = val;

            // Increment target indices
            if input_grad_numel > 0 && _target_linear_idx < input_grad_numel - 1 {
                let mut dim_to_increment = target_rank;
                while dim_to_increment > 0 {
                    dim_to_increment -= 1;
                    current_target_indices[dim_to_increment] += 1;
                    if current_target_indices[dim_to_increment] < target_shape[dim_to_increment] {
                        break;
                    }
                    current_target_indices[dim_to_increment] = 0;
                }
            }
        }

        let input_gradient = Tensor::new(input_grad_data, target_shape)?;
        Ok(vec![input_gradient])
    }

    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Calculates the sum of elements along specified axes.
/// Requires the tensor to be on CPU.
/// Supports autograd.
pub fn sum_axes<T>(
    input: &Tensor<T>,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Clone
        + Zero
        + AddAssign
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Default
        + PartialEq
        + PartialOrd
        + One
        + Sum
        + Add<Output = T>, // Added Add bound for autograd compatibility
{
    // --- Autograd Setup ---
    let requires_grad = input.requires_grad();
    let input_node_arc = input.data.clone(); // Clone the Arc<RwLock<TensorData>>
    let input_shape_clone = input.shape(); // Clone shape needed for backward op

    // --- Acquire read lock ---
    let input_guard = input.read_data();

    // --- Device Check ---
    let device = input_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Summation is currently only supported on CPU, not {:?}",
            device
        )));
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
        let output_shape = if keep_dims {
            vec![1; input_rank]
        } else {
            vec![]
        };
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
    let output_shape = { // Calculate output shape within its own scope
        let mut shape = Vec::new();
        let input_shape = &input_guard.shape; // Use guarded shape
        let input_rank = input_shape.len();
        // Handle Sum All Case (output shape calculation)
        if axes.is_empty() {
            if keep_dims {
                vec![1; input_rank]
            } else {
                vec![]
            }
        } else {
            // Calculate output shape based on processed_axes
            for (dim, &size) in input_shape.iter().enumerate() {
                if !processed_axes.contains(&dim) {
                    shape.push(size);
                } else if keep_dims {
                    shape.push(1);
                }
            }
            // Handle edge cases (sum to scalar with keep_dims=false)
             if shape.is_empty() && !processed_axes.is_empty() && !keep_dims {
                 shape = vec![]; // Ensure truly scalar shape
             } else if shape.is_empty() && keep_dims && input_rank > 0 {
                 // If all axes reduced and keep_dims=true, shape should be all 1s
                 shape = vec![1; input_rank];
             }
            shape // Return calculated shape
        }
    };

    // --- Perform Summation ---
    let output_numel: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };
    let mut result_data = vec![T::zero(); output_numel];

    let mut current_input_indices = vec![0; input_rank];
    let input_strides = &input_guard.strides;
    let input_offset = input_guard.offset;

    // Iterate through all elements of the input tensor
    for _i in 0..input_guard.numel() {
        // Use guard.numel()
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
        if !output_shape.is_empty() {
            // Avoid index calculation for scalar output
            let mut stride_product = 1;
            for j in (0..output_shape.len()).rev() {
                output_flat_idx += output_indices[j] * stride_product;
                // Calculate output strides on the fly if needed, or assume contiguity for output
                if j > 0 {
                    stride_product *= output_shape[j];
                }
            }
        } // else output_flat_idx remains 0 for scalar output

        if output_flat_idx < result_data.len() {
            // Bounds check
            result_data[output_flat_idx] += val;
        }

        // Increment input indices (N-dimensional counter logic)
        if input_guard.numel() > 0 && _i < input_guard.numel() - 1 {
            // Avoid increment on last element
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
    let result_tensor = Tensor::new(result_data, output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        // Create the backward operation context
        let backward_op = SumAxesBackward {
            input_node: input_node_arc,
            input_shape: input_shape_clone,
            axes: axes.to_vec(),
            keep_dims,
        };
        // Set grad_fn for the result tensor
        result_tensor.set_grad_fn(Some(Arc::new(backward_op)))?;
        // Ensure result tensor requires grad if input did
         result_tensor.set_requires_grad(true)?;
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;
    use crate::Tensor;
    use approx::assert_relative_eq;
    use num_traits::{One, Zero};
    use std::cmp::PartialEq;
    use std::cmp::PartialOrd;
    use std::default::Default;
    use std::iter::Sum;
    use std::ops::AddAssign;

    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
        T: Clone
            + Zero
            + AddAssign
            + Debug
            + Copy
            + Send
            + Sync
            + 'static
            + Default
            + PartialEq
            + PartialOrd
            + One
            + Sum,
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
        let t = create_test_tensor(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        );
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
        let res_all_cpu_data = res_all_buffer_arc
            .cpu_data()
            .expect("Result all not on CPU");
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

// --- Autograd Tests ---

#[cfg(test)]
mod autograd_tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::Tensor;

    // Helper to create tensors for tests
    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }
    // Added helper for f64
    fn create_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn test_sum_axes_backward_simple_keep_dims() {
        // Use f64
        let input = create_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        input.set_requires_grad(true).unwrap();

        // Adjust function signature for f64
        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[0], true)
        };

        let output_shape = vec![1, 3];
        // Use f64 for output_grad
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5; // Epsilon can remain the same or be smaller for f64
        let tolerance = 1e-7; // Stricter tolerance for f64

        // Pass f64 tensors to check_grad
        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

     #[test]
    fn test_sum_axes_backward_simple_no_keep_dims() {
        // Use f64
        let input = create_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[1], false) // Sum along axis 1, no keep dims -> shape [2]
        };

        let output_shape = vec![2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap(); // Use f64
        let epsilon = 1e-5;
        let tolerance = 1e-7; // Stricter tolerance for f64

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }


    #[test]
    fn test_sum_all_backward_keep_dims() {
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f32>]| -> Result<Tensor<f32>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[], true) // Sum all, keep dims -> shape [1, 1]
        };

        let output_shape = vec![1, 1];
        let output_grad = Tensor::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-4; // Increased tolerance for f32

        let grad_check_result = check_grad(func, &[input.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_sum_all_backward_no_keep_dims() {
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f32>]| -> Result<Tensor<f32>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[], false) // Sum all, no keep dims -> shape [] (scalar)
        };

        let output_shape = vec![]; // Scalar shape
        let output_grad = Tensor::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-4; // Increased tolerance for f32

        let grad_check_result = check_grad(func, &[input.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_sum_multiple_axes_backward() {
        // Use f64
        let input = create_tensor_f64((1..=24).map(|x| x as f64).collect::<Vec<_>>(), vec![2, 3, 4]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[0, 2], false) // Sum along axes 0 and 2 -> shape [3]
        };

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap(); // Use f64
        let epsilon = 1e-5;
        let tolerance = 1e-7; // Stricter tolerance for f64

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

      #[test]
     fn test_sum_no_reduction_backward() {
         let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
         input.set_requires_grad(true).unwrap();

         let epsilon = 1e-5;
         let tolerance = 1e-4; // Increased tolerance for f32

         // Case 1: Sum all (axes=[]) keep_dims=false -> Scalar output
         let func1 = |inputs: &[Tensor<f32>]| {
             assert_eq!(inputs.len(), 1);
             sum_axes(&inputs[0], &[], false)
        };
         let output1_shape = vec![];
         let output1_grad = Tensor::ones(output1_shape).unwrap();
         let check1 = check_grad(func1, &[input.clone()], &output1_grad, epsilon, tolerance);
         assert!(check1.is_ok(), "Sum all (scalar) grad check failed: {:?}", check1.err());
         // --- Clear grad manually before next check ---
         input.clear_grad();

         // Case 2: Sum all (axes=[]) keep_dims=true -> Output shape [1]
         let func2 = |inputs: &[Tensor<f32>]| {
             assert_eq!(inputs.len(), 1);
             sum_axes(&inputs[0], &[], true)
         };
         let output2_shape = vec![1]; // Original shape was [3], sum all keep_dims -> [1]
         let output2_grad = Tensor::ones(output2_shape).unwrap();
         let check2 = check_grad(func2, &[input.clone()], &output2_grad, epsilon, tolerance);
          assert!(check2.is_ok(), "Sum all (keep_dims) grad check failed: {:?}", check2.err());
     }

     // TODO: Add tests for higher dimensions if needed.
     // TODO: Add tests involving views before/after sum if needed.
}

// --- End Autograd Tests ---
