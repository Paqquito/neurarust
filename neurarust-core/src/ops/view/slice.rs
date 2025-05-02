// neurarust-core/src/ops/view/slice.rs

use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{calculate_strides, index_to_coord};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};
use super::SliceArg; // Importer depuis le mod.rs parent

/// Performs the slicing operation, creating a view.
///
/// If the input tensor requires gradients, the output tensor will also require gradients
/// and have its `grad_fn` set to a `SliceBackward` operation node.
///
/// # Arguments
/// * `tensor`: The input tensor to slice.
/// * `ranges`: A slice of `SliceArg` defining the slice for each dimension.
///             The length must match the tensor's rank. `end` is exclusive.
///
/// # Returns
/// A new Tensor representing the view, or an error.
pub(crate) fn slice_op<T>(
    tensor: &Tensor<T>,
    ranges: &[SliceArg],
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Default
        + Send
        + Sync
        + 'static
        + Debug
        + Copy
        + Zero
        + AddAssign
        + PartialEq
        + PartialOrd
        + Sum
        + One,
{
    // --- Autograd Setup ---
    let requires_grad = tensor.requires_grad();
    let mut input_id_maybe: Option<NodeId<T>> = None;
    let mut input_shape_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
        input_shape_maybe = Some(tensor.shape());
    }

    // Access the pub(crate) field directly
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for slicing".to_string(),
        )
    })?;

    if ranges.len() != guard.shape.len() {
        return Err(NeuraRustError::DimensionMismatch {
            expected: guard.shape.len(),
            actual: ranges.len(),
        });
    }

    let mut new_shape = Vec::with_capacity(guard.shape.len());
    let mut new_offset = guard.offset;
    let mut current_ranges = Vec::with_capacity(ranges.len());

    for (i, range) in ranges.iter().enumerate() {
        let start = range.start;
        let end = range.end;
        let dim_size = guard.shape[i];

        if start > end {
            return Err(NeuraRustError::SliceError {
                message: format!(
                    "Invalid slice range start > end ({}:{}) for dimension {} with size {}",
                    start, end, i, dim_size
                ),
            });
        }
        if end > dim_size {
            return Err(NeuraRustError::SliceError {
                message: format!(
                    "Invalid slice range end > size ({}:{}) for dimension {} with size {}",
                    start, end, i, dim_size
                ),
            });
        }

        new_shape.push(end - start);
        if dim_size > 0 {
            new_offset += start * guard.strides[i];
        }
        current_ranges.push(range.clone());
    }

    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let strides = guard.strides.clone();

    // --- Create View TensorData ---
    // Now it's safe to drop the guard
    drop(guard);

    let new_td = TensorData::new_view(buffer_arc, device, new_offset, new_shape.clone(), strides);

    // Construct the Tensor directly using its field
    let result_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        // Ensure captured values exist (unwrap is safe due to requires_grad check and assignment above)
        let input_shape = input_shape_maybe.unwrap();

        // 1. Create backward context
        let backward_context = SliceBackward {
            input_id: input_id_maybe.unwrap(),
            input_shape,
            ranges: current_ranges,
        };
        // 2. Wrap context in Arc
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        // 3. Set autograd properties on result
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---
    Ok(result_tensor)
}

// --- Slice Backward Operation ---

/// Backward operation context for the slice operation.
/// Stores the NodeId of the input tensor, its original shape, strides, offset,
/// and the slice ranges used in the forward pass.
#[derive(Debug)]
struct SliceBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    input_shape: Vec<usize>,
    ranges: Vec<SliceArg>,
}

// Mark SliceBackward as Send + Sync (unsafe justification as per AddBackward).
// We guarantee pointer validity and synchronized access via TensorData's RwLock.
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for SliceBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for SliceBackward<T> {}

impl<T> BackwardOp<T> for SliceBackward<T>
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
        + PartialOrd
        + Sum
        + One,
{
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input_id]
    }

    /// Computes the gradient for the input tensor of the slice operation.
    /// It creates a zero tensor with the shape of the original input and then
    /// "scatters" (adds) the output gradient (`grad_output`) into the locations
    /// corresponding to the slice.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // Note: T requires AddAssign, Zero, Copy, Add, One, Sum etc based on usage below

        let device = grad_output.device();
        let input_numel = self.input_shape.iter().product();
        let mut input_grad_full_data = vec![T::zero(); input_numel];

        let grad_output_guard = grad_output.read_data();

        // Handle empty grad_output: return zero grad for the whole input shape.
        if grad_output_guard.numel() == 0 {
            // Release guard early
            drop(grad_output_guard);
            let input_grad_full = Tensor::new(input_grad_full_data, self.input_shape.clone())?;
            if input_grad_full.device() != device {
                return Err(NeuraRustError::InternalError(format!(
                    "Slice backward (empty grad) created gradient on {:?} but expected {:?}",
                    input_grad_full.device(), device
                )));
            }
            return Ok(vec![input_grad_full]);
        }

        if device != StorageDevice::CPU {
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "Slice backward pass currently only supported on CPU, got grad_output on {:?}",
                device
            )));
        }
        // Get buffer Arc only once
        let grad_output_buffer_arc = grad_output_guard.data.cpu_data()?.clone();
        let grad_output_buffer_slice = grad_output_buffer_arc.as_slice();
        let grad_output_shape = &grad_output_guard.shape;
        let grad_output_strides = &grad_output_guard.strides;
        let grad_output_base_offset = grad_output_guard.offset;

        // Debug: Print shape of grad_output (keep for now?)
        dbg!(grad_output_shape);

        let grad_numel = grad_output_guard.numel();
        let input_rank = self.input_shape.len();
        let grad_rank = grad_output_shape.len();

        // Pre-calculate input strides once
        let contiguous_input_strides = calculate_strides(&self.input_shape);

        // Restore rank check - This check might be too strict if broadcasting/reduction
        // happened between the slice and the backward call. However, for a simple slice
        // backward, the ranks *should* typically match unless the slice itself reduces rank,
        // or the grad_output is unexpectedly scalar.
        if input_rank != grad_rank {
            // --- Handle Rank Mismatch --- Special case: grad_output is scalar
            if grad_rank == 0 && input_rank > 0 {
                // If grad_output is scalar, add its value to all elements
                // within the original slice range in the input_grad.
                if grad_numel != 1 {
                    return Err(NeuraRustError::InternalError(
                        "Scalar grad_output (rank 0) has numel != 1".to_string()
                    ));
                }
                // Get the single scalar value from grad_output
                // Offset is simply the base offset for a scalar
                let grad_scalar_val = grad_output_buffer_slice[grad_output_base_offset];

                // Iterate through all indices corresponding to the *slice ranges*
                // within the *input shape* and add the scalar gradient.
                // This requires a multi-dimensional iterator or nested loops.

                // Calculate the shape of the slice result *without* performing the slice
                let slice_result_shape: Vec<usize> = self.ranges.iter().map(|r| r.end - r.start).collect();
                let slice_result_numel: usize = slice_result_shape.iter().product();
                let slice_result_strides = calculate_strides(&slice_result_shape);

                for slice_linear_idx in 0..slice_result_numel {
                    let slice_coord = index_to_coord(slice_linear_idx, &slice_result_strides, &slice_result_shape);

                    // Convert slice coord back to input coord
                    let mut input_coord = vec![0; input_rank];
                    for dim in 0..input_rank {
                        input_coord[dim] = slice_coord[dim] + self.ranges[dim].start;
                    }

                    // Calculate linear offset into the input_grad_full_data buffer
                    let input_grad_linear_offset = input_coord
                        .iter()
                        .zip(&contiguous_input_strides)
                        .map(|(&coord, &stride)| coord * stride)
                        .sum::<usize>();

                    if input_grad_linear_offset < input_grad_full_data.len() {
                        input_grad_full_data[input_grad_linear_offset] += grad_scalar_val;
                    } else {
                         return Err(NeuraRustError::InternalError(format!(
                            "Slice backward (scalar grad) calculated offset {} out of bounds for input grad buffer size {} (input_coord: {:?}, input_shape: {:?})",
                            input_grad_linear_offset, input_grad_full_data.len(), input_coord, self.input_shape
                        )));
                    }
                }

            } else {
                // If ranks mismatch and grad is not scalar, it's an unsupported/error case.
                return Err(NeuraRustError::InternalError(format!(
                    "Slice backward rank mismatch: input {}, grad_output {}",
                    input_rank, grad_rank
                )));
            }
        }
        // --- Handle Matching Ranks --- (Original Logic)
        else {
            // Ranks match, proceed with original element-wise scatter
            // let _current_grad_coord = vec![0; grad_rank]; // Unused? Removed.
            for grad_linear_idx in 0..grad_numel {
                // Calculate coords in the grad_output tensor view
                let grad_output_coord = index_to_coord(grad_linear_idx, grad_output_strides, grad_output_shape);

                // Calculate the actual offset in the underlying buffer for grad_output
                let grad_val_offset = grad_output_guard.get_offset(&grad_output_coord);
                // Get the gradient value
                let grad_val = grad_output_buffer_slice[grad_val_offset];

                // Calculate the corresponding coordinates in the original input tensor
                let mut input_coord = vec![0; input_rank];
                for dim in 0..input_rank {
                    // Check index bounds implicitly handled by coord calculation?
                    // Need to be careful if grad_output_coord length differs from input_rank (handled above)
                    input_coord[dim] = grad_output_coord[dim] + self.ranges[dim].start;
                }

                // Calculate the linear offset in the (flat) input gradient buffer
                let input_grad_linear_offset = input_coord
                    .iter()
                    .zip(&contiguous_input_strides)
                    .map(|(&coord, &stride)| coord * stride)
                    .sum::<usize>();

                // Add the gradient value to the corresponding element in the input gradient buffer
                if input_grad_linear_offset < input_grad_full_data.len() {
                    input_grad_full_data[input_grad_linear_offset] += grad_val;
                } else {
                    return Err(NeuraRustError::InternalError(format!(
                        "Slice backward calculated offset {} out of bounds for input grad buffer size {} (input_coord: {:?}, input_shape: {:?})",
                        input_grad_linear_offset, input_grad_full_data.len(), input_coord, self.input_shape
                    )));
                }
            }
        }
        // --- End Rank Handling ---

        drop(grad_output_guard);

        let input_grad_full = Tensor::new(input_grad_full_data, self.input_shape.clone())?;
        if input_grad_full.device() != device {
            return Err(NeuraRustError::InternalError(format!(
                "Slice backward created gradient on {:?} but expected {:?}",
                input_grad_full.device(), device
            )));
        }

        Ok(vec![input_grad_full])
    }
}


// --- Tests for Slice Op ---
#[cfg(test)]
mod tests {
    use super::*; // Importe slice_op, SliceBackward, etc. depuis le module parent (slice.rs)
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::ops::view::SliceArg; // Importer explicitement
    use crate::tensor::Tensor;
    use crate::utils::testing::{create_test_tensor, create_test_tensor_with_grad};
     // Importer pour test
     // Importer pour test

    #[test]
    fn test_slice_basic() {
        let t = create_test_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        );
        let ranges = vec![SliceArg::new(0, 1), SliceArg::new(1, 3)];
        let sliced = t.slice(&ranges).unwrap();
        let expected_data = vec![2.0, 3.0];
        let expected_shape = vec![1, 2];
        assert_eq!(sliced.shape(), expected_shape, "Shape mismatch");
        assert_eq!(sliced.get(&[0, 0]).unwrap(), expected_data[0], "Data mismatch at [0,0]");
        assert_eq!(sliced.get(&[0, 1]).unwrap(), expected_data[1], "Data mismatch at [0,1]");
    }

    #[test]
    fn test_slice_full() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let ranges = vec![SliceArg::new(0, 2), SliceArg::new(0, 2)];
        let sliced = t.slice(&ranges).unwrap();
        assert_eq!(sliced, t, "Full slice should be equal to original");
        assert!(t.is_contiguous());
        assert!(sliced.is_contiguous());
    }

    #[test]
    fn test_slice_empty_dim() {
        let t = create_test_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        );
        let ranges = vec![SliceArg::new(0, 2), SliceArg::new(1, 1)];
        let sliced = t.slice(&ranges).unwrap();
        assert_eq!(sliced.shape(), vec![2, 0]);
        assert_eq!(sliced.numel(), 0);
    }

    #[test]
    fn test_slice_invalid_range_start_gt_end() {
        let t = create_test_tensor(vec![1.0, 2.0], vec![2]);
        let ranges = vec![SliceArg::new(1, 0)];
        let result = t.slice(&ranges);
        assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
    }

    #[test]
    fn test_slice_invalid_range_end_gt_size() {
        let t = create_test_tensor(vec![1.0, 2.0], vec![2]);
        let ranges = vec![SliceArg::new(0, 3)];
        let result = t.slice(&ranges);
        assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
    }

    #[test]
    fn test_slice_rank_mismatch() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let ranges = vec![SliceArg::new(0, 1)];
        let result = t.slice(&ranges);
        assert!(matches!(
            result,
            Err(NeuraRustError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_slice_view_data_sharing() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let t = Tensor::new(data.clone(), shape.clone()).unwrap();

        let ranges = vec![SliceArg::new(0, 1), SliceArg::new(0, 2)];
        let sliced_view = t.slice(&ranges).unwrap();

        let t_data_ptr = Arc::as_ptr(&t.read_data().data);
        let sliced_data_ptr = Arc::as_ptr(&sliced_view.read_data().data);
        assert_eq!(t_data_ptr, sliced_data_ptr, "Slice should share the data buffer");

        let t_guard = t.read_data();
        let sliced_guard = sliced_view.read_data();
        assert_eq!(sliced_guard.shape, vec![1, 2]);
        let ranges_offset = vec![SliceArg::new(1, 2), SliceArg::new(0, 1)];
        let sliced_offset_view = t.slice(&ranges_offset).unwrap();
        let sliced_offset_guard = sliced_offset_view.read_data();
        assert_eq!(sliced_offset_guard.offset, t_guard.offset + 1 * t_guard.strides[0] + 0 * t_guard.strides[1], "Slice offset calculation seems wrong");
        assert_eq!(sliced_offset_guard.offset, 2);

        assert_eq!(sliced_guard.offset, t_guard.offset, "Slice offset should be base offset for slice starting at 0");

        assert_eq!(sliced_guard.strides, t_guard.strides);
    }

    #[test]
    fn test_slice_backward() {
        let slice_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1, "Slice function expects one input tensor");
            let ranges = vec![SliceArg::new(0, 1), SliceArg::new(1, 3)];
            inputs[0].slice(&ranges)
        };
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![1, 2]).unwrap();
        let result = check_grad(slice_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed: {:?}", result.err());
    }

    #[test]
    fn test_slice_backward_scalar_result() {
        let slice_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            let ranges = vec![SliceArg::new(1, 2), SliceArg::new(2, 3)];
            inputs[0].slice(&ranges)
        };
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![1, 1]).unwrap();
        let result = check_grad(slice_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for scalar slice: {:?}", result.err());
    }

    #[test]
    fn test_slice_backward_empty_result() {
        let slice_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            let ranges = vec![SliceArg::new(0, 2), SliceArg::new(1, 1)];
            inputs[0].slice(&ranges)
        };
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        );
        let dummy_output_grad = Tensor::<f64>::ones(vec![2, 0]).unwrap();
        let result = check_grad(slice_fn, &[input_data.clone()], &dummy_output_grad, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for empty slice: {:?}", result.err());

        let output = slice_fn(&[input_data.clone()]).unwrap();
        assert_eq!(output.numel(), 0);
        output.backward(None).unwrap();
        let grad = input_data.grad().unwrap();
        let expected_grad = Tensor::<f64>::zeros_like(&input_data).unwrap();
        assert_eq!(grad, expected_grad, "Gradient for empty slice output should be zero");
    }
} 