// neurarust-core/src/ops/view_ops.rs

use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData; // Needed for TensorData::new_view
                                    // use crate::buffer::Buffer;          // Not directly used in this file
                                    // use crate::device::StorageDevice;  // Not directly used in this file
use crate::tensor::utils::{calculate_strides, index_to_coord};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

// Define a type alias or struct for slice arguments for clarity.
// Using a tuple (start, end) for each dimension for now.
// Excludes the 'step' for simplicity initially.
#[derive(Debug, Clone)]
pub struct SliceArg {
    pub start: usize,
    pub end: usize,
    // Optional: Add step later if needed
    // pub step: usize,
}

impl SliceArg {
    // Helper constructor if needed
    pub fn new(start: usize, end: usize) -> Self {
        SliceArg { start, end }
    }
}

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

/// Performs the transpose operation between two dimensions, creating a view.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `dim1`: The first dimension to transpose.
/// * `dim2`: The second dimension to transpose.
///
/// # Returns
/// A new Tensor representing the transposed view, or an error.
pub(crate) fn transpose_op<T>(
    tensor: &Tensor<T>,
    dim1: usize,
    dim2: usize,
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

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
    }

    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for transpose".to_string(),
        )
    })?;

    let rank = guard.shape.len();

    // 2. Validate dimensions
    if dim1 >= rank || dim2 >= rank {
        return Err(NeuraRustError::DimensionMismatch {
            expected: rank,
            actual: std::cmp::max(dim1, dim2),
        });
    }

    // 3. Calculate new shape and strides
    let mut new_shape = guard.shape.clone();
    let mut new_strides = guard.strides.clone();

    // Swap shape and strides at dim1 and dim2
    new_shape.swap(dim1, dim2);
    new_strides.swap(dim1, dim2);

    // 4. Get other necessary info
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    // Drop the read guard before creating the new RwLock
    drop(guard);

    // 5. Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape,
        new_strides,
    );

    // 6. Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        let backward_context = TransposeBackward {
            input_id: input_id_maybe.unwrap(),
            dim1,
            dim2,
            _phantom: std::marker::PhantomData,
        };

        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        new_tensor.set_requires_grad(true)?;
        new_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(new_tensor)
}

/// Performs the permute operation, creating a view with reordered dimensions.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `dims`: A slice representing the desired permutation of dimensions.
///           Must contain each dimension index from 0 to rank-1 exactly once.
///
/// # Returns
/// A new Tensor representing the permuted view, or an error.
pub(crate) fn permute_op<T>(
    tensor: &Tensor<T>,
    dims: &[usize],
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
    let mut inverse_dims_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
        let rank = tensor.shape().len();
        if rank > 0 {
            let mut inverse_dims = vec![0; rank];
            for (i, &dim) in dims.iter().enumerate() {
                if dim < rank {
                    inverse_dims[dim] = i;
                } else {
                    return Err(NeuraRustError::InvalidPermutation { dims: dims.to_vec(), rank });
                }
            }
            inverse_dims_maybe = Some(inverse_dims);
        } else if !dims.is_empty() {
            // If rank is 0 but dims is not empty, it's an error handled below,
            // so no need to calculate inverse_dims.
        } else {
            inverse_dims_maybe = Some(vec![]);
        }
    }
    // --- End Autograd Setup ---

    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for permute".to_string(),
        )
    })?;

    let rank = guard.shape.len();

    // Add check for scalar tensor (rank 0)
    if rank == 0 {
        if !dims.is_empty() {
            return Err(NeuraRustError::DimensionMismatch {
                expected: 0,
                actual: dims.len(),
            });
        } else {
            return Err(NeuraRustError::DimensionMismatch {
                expected: 0,
                actual: 0,
            });
        }
    }

    // 2. Validate permutation dimensions (original check, now only for rank > 0)
    if dims.len() != rank {
        return Err(NeuraRustError::DimensionMismatch {
            expected: rank,
            actual: dims.len(),
        });
    }
    let mut seen = vec![false; rank];
    for &dim in dims {
        if dim >= rank || seen[dim] {
            return Err(NeuraRustError::InvalidPermutation {
                dims: dims.to_vec(),
                rank,
            });
        }
        seen[dim] = true;
    }

    // 3. Calculate new shape and strides
    let mut new_shape = Vec::with_capacity(rank);
    let mut new_strides = Vec::with_capacity(rank);
    for &new_dim_index in dims {
        new_shape.push(guard.shape[new_dim_index]);
        new_strides.push(guard.strides[new_dim_index]);
    }

    // 4. Get other necessary info
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    drop(guard);

    // 5. Create new TensorData using new_view
    let new_td = TensorData::new_view(buffer_arc, device, offset, new_shape, new_strides);

    // 6. Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        let inverse_dims = inverse_dims_maybe.ok_or_else(|| NeuraRustError::InternalError("Missing inverse permutation for permute backward pass".to_string()))?;

        let backward_context = PermuteBackward {
            input_id: input_id_maybe.unwrap(),
            inverse_dims,
            _phantom: std::marker::PhantomData,
        };

        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        new_tensor.set_requires_grad(true)?;
        new_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(new_tensor)
}

/// Performs the reshape operation. Currently only supports creating a view
/// for contiguous tensors. For non-contiguous tensors, call `.contiguous()` first.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `new_shape_vec`: The desired new shape.
///
/// # Returns
/// A new Tensor representing the reshaped view, or an error.
pub(crate) fn reshape_op<T>(
    tensor: &Tensor<T>,
    new_shape_vec: Vec<usize>,
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
    let mut original_shape_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
        original_shape_maybe = Some(tensor.shape());
    }
    // --- End Autograd Setup ---

    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for reshape".to_string(),
        )
    })?;

    // 2. Validate number of elements
    let original_numel: usize = guard.shape.iter().product();
    let new_numel: usize = new_shape_vec.iter().product();

    if original_numel != new_numel {
        return Err(NeuraRustError::ShapeMismatch {
            expected: guard.shape.clone(),
            actual: new_shape_vec,
            operation: "reshape".to_string(),
        });
    }

    // 3. Check for contiguity (current limitation)
    if !guard.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(
            "Reshape currently only supports contiguous tensors. Call .contiguous() first."
                .to_string(),
        ));
    }

    // 4. If contiguous and numel matches, calculate new strides and create view
    let new_strides = TensorData::<T>::calculate_contiguous_strides(&new_shape_vec);

    // Get necessary info from locked data
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    drop(guard);

    // Create new TensorData using new_view
    let new_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape_vec.clone(),
        new_strides,
    );

    // Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        let original_shape = original_shape_maybe.ok_or_else(|| NeuraRustError::InternalError("Missing original shape for reshape backward pass".to_string()))?;

        let backward_context = ReshapeBackward {
            input_id: input_id_maybe.unwrap(),
            original_shape,
            _phantom: std::marker::PhantomData,
        };

        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        new_tensor.set_requires_grad(true)?;
        new_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(new_tensor)
}

// Placeholder for Debug trait implementation if needed

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

// --- Transpose Backward Operation ---

#[derive(Debug)]
struct TransposeBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    dim1: usize,
    dim2: usize,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for TransposeBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for TransposeBackward<T> {}

impl<T> BackwardOp<T> for TransposeBackward<T>
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
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input_id]
    }

    /// Computes the gradient for the input tensor of the transpose operation.
    /// Since transpose is just rearranging data (a view), the backward pass
    /// simply applies the *same* transpose operation to the incoming gradient.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let input_grad = transpose_op(grad_output, self.dim1, self.dim2)?;
        Ok(vec![input_grad])
    }
}

// --- Permute Backward Operation ---

#[derive(Debug)]
struct PermuteBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    inverse_dims: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for PermuteBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for PermuteBackward<T> {}

impl<T> BackwardOp<T> for PermuteBackward<T>
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
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input_id]
    }

    /// Computes the gradient for the input tensor of the permute operation.
    /// This involves applying the *inverse* permutation to the incoming gradient.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let input_grad = permute_op(grad_output, &self.inverse_dims)?;
        Ok(vec![input_grad])
    }
}

// --- Reshape Backward Operation ---

#[derive(Debug)]
struct ReshapeBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    original_shape: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for ReshapeBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for ReshapeBackward<T> {}

impl<T> BackwardOp<T> for ReshapeBackward<T>
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
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input_id]
    }

    /// Computes the gradient for the input tensor of the reshape operation.
    /// Since reshape (as implemented here as a view) only changes metadata,
    /// the backward pass simply reshapes the incoming gradient back to the
    /// original input tensor's shape.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let _input_id = self.input_id; // Keep field access if needed elsewhere potentially
        let input_grad = reshape_op(grad_output, self.original_shape.clone())?;
        Ok(vec![input_grad])
    }
}

// --- Tests for View Ops ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;

    fn create_test_tensor<T>(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T>
    where
        T: Default
            + Debug
            + Clone
            + Copy
            + PartialEq
            + PartialOrd
            + Zero
            + One
            + Sum
            + Send
            + Sync
            + 'static,
    {
        Tensor::new(data, shape).unwrap()
    }

    fn create_test_tensor_with_grad<T>(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T>
    where
        T: Default
            + Debug
            + Clone
            + Copy
            + PartialEq
            + PartialOrd
            + Zero
            + One
            + Sum
            + Send
            + Sync
            + 'static
            + AddAssign,
    {
        let tensor = Tensor::new(data, shape).unwrap();
        tensor.set_requires_grad(true).unwrap();
        tensor
    }

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

    #[test]
    fn test_transpose_backward() {
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![3, 2]).unwrap();

        // Calculate analytical gradient
        let output = input_data.transpose(0, 1).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is transpose(output_grad_val)
        let expected_grad = transpose_op(&output_grad_val, 0, 1).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

    #[test]
    fn test_transpose_backward_higher_dim() {
        let input_data = create_test_tensor_with_grad(
            (1..=24).map(|x| x as f64).collect(),
            vec![2, 3, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![2, 4, 3]).unwrap();

        // Calculate analytical gradient
        let output = input_data.transpose(1, 2).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is transpose(output_grad_val)
        let expected_grad = transpose_op(&output_grad_val, 1, 2).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

    #[test]
    fn test_permute_backward() {
        let input_data = create_test_tensor_with_grad(
            (1..=24).map(|x| x as f64).collect(),
            vec![2, 3, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![4, 2, 3]).unwrap();

        // Calculate analytical gradient
        let dims = vec![2, 0, 1];
        let output = input_data.permute(&dims).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is permute(output_grad_val) with inverse dims
        let mut inverse_dims = vec![0; dims.len()];
        for (i, &dim) in dims.iter().enumerate() {
            inverse_dims[dim] = i;
        }
        let expected_grad = permute_op(&output_grad_val, &inverse_dims).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

    #[test]
    fn test_permute_backward_identity() {
        let permute_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].permute(&[0, 1])
        };
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![2, 2]).unwrap();
        let result = check_grad(permute_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for identity permute: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].reshape(vec![3, 2])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=6).map(|x| x as f64).collect(),
            vec![2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![3, 2]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for reshape: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward_flatten() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            let numel = inputs[0].numel();
            inputs[0].reshape(vec![numel])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=12).map(|x| x as f64).collect(),
            vec![2, 2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![12]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for flatten reshape: {:?}", result.err());
    }

    #[test]
    fn test_reshape_backward_add_dim() {
        let reshape_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].reshape(vec![2, 2, 1, 3])
        };
        let input_data = create_test_tensor_with_grad(
            (1..=12).map(|x| x as f64).collect(),
            vec![2, 2, 3],
        );
        assert!(input_data.is_contiguous());
        let output_grad_val = Tensor::<f64>::ones(vec![2, 2, 1, 3]).unwrap();
        let result = check_grad(reshape_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for reshape adding dim: {:?}", result.err());
    }

    #[test]
    fn test_transpose_basic() {
        let t = create_test_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let transposed = t.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), vec![3, 2]);

        let t_guard = t.read_data();
        let transposed_guard = transposed.read_data();
        assert_eq!(Arc::as_ptr(&t_guard.data), Arc::as_ptr(&transposed_guard.data), "Transpose should share buffer");
        assert_eq!(transposed_guard.offset, t_guard.offset);
        assert_eq!(transposed_guard.strides, vec![1, 3]);
        assert!(!transposed.is_contiguous());

        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[0, 0])], 1.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[0, 1])], 4.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[1, 0])], 2.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[1, 1])], 5.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[2, 0])], 3.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[2, 1])], 6.0);
    }
}
