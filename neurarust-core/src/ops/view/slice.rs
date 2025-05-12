// neurarust-core/src/ops/view/slice.rs

use crate::autograd::BackwardOp;
// Assuming non-generic
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use crate::buffer::{Buffer, CpuBuffer};

use std::sync::{Arc, RwLock};
use std::fmt::Debug;

/// Represents the different ways to index or slice a tensor dimension.
///
/// This enum is used as input to the `slice` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceArg {
    /// Represents the `...` ellipsis, expanding to the necessary number of full slices (`:`).
    Ellipsis,
    /// Represents a standard slice `start:end:step`.
    /// Indices can be negative (counting from the end).
    Slice(isize, isize, isize),
    /// Represents indexing with a single integer.
    /// Removes the dimension being indexed.
    Index(isize),
    /// Represents inserting a new dimension of size 1 (like `np.newaxis`).
    NewAxis,
}

/// Internal representation of a processed slice for a single dimension.
#[derive(Debug, Clone, Copy)]
pub struct SliceRange {
    /// The starting index (inclusive) in the original dimension.
    pub start: usize,
    /// The step size for the slice.
    pub step: usize,
    /// The number of elements selected in this dimension (size of the dimension after slicing).
    pub size: usize,
}

/// Backward operation context for the `slice` operation.
///
/// Stores information needed to propagate gradients back through a slice:
/// - The original input tensor's data (`input_node`).
/// - The shape of the original input tensor (`original_shape`).
/// - The calculated `SliceRange` for each dimension (`ranges`), defining the exact slice taken.
#[derive(Debug)]
struct SliceBackward {
    input_node: Arc<RwLock<TensorData>>,
    original_shape: Vec<usize>,
    ranges: Vec<SliceRange>,
}

// --- Backward Operation Implementation ---
impl BackwardOp for SliceBackward {
    /// Computes the gradient for the slice operation.
    ///
    /// This involves creating a gradient tensor filled with zeros that has the same
    /// shape as the original input tensor. Then, the incoming gradient (`grad_output`)
    /// is added (scattered) into this zero tensor at the locations corresponding to
    /// the original slice.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output slice of the original operation.
    ///
    /// # Returns
    ///
    /// An `Ok(vec![])` as the gradient is accumulated directly into the input node's
    /// `.grad` field. Returns an error if memory allocation, buffer access, or
    /// arithmetic operations fail.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let input_node_guard = self.input_node.read().map_err(|_| NeuraRustError::LockError {
            lock_type: "read".to_string(),
            reason: "Failed to lock input node for read in SliceBackward".to_string(),
        })?;
        let input_dtype = input_node_guard.dtype;
        let _input_device = input_node_guard.device; // Prefix with _
        let input_strides = &input_node_guard.strides; // Strides of the original input
        let input_offset = input_node_guard.offset; // Offset of the original input

        // 1. Create a zero tensor with the shape of the original input
        let grad_input = match input_dtype { // Make grad_input non-mutable
            DType::F32 => crate::tensor::zeros(&self.original_shape)?,
            DType::F64 => crate::tensor::zeros_f64(&self.original_shape)?,
            DType::I32 | DType::I64 | DType::Bool => {
                return Err(NeuraRustError::UnsupportedOperation(
                    "slice_op n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                );
            },
        };

        // Ensure grad_output is contiguous for easier iteration
        let grad_output_contiguous = grad_output.contiguous()?;
        let grad_output_guard = grad_output_contiguous.read_data();
        let grad_output_shape = &grad_output_guard.shape;
        let grad_output_strides = &grad_output_guard.strides;
        let grad_output_offset = grad_output_guard.offset;
        let grad_output_buffer = grad_output_guard.buffer();

        // Get mutable access to grad_input buffer using clone-on-write
        let grad_input_buffer_arc = grad_input.write_data().buffer.clone(); // Clone the Arc
        let mut grad_input_buffer_owned = match Arc::try_unwrap(grad_input_buffer_arc) {
            Ok(buffer) => buffer, // We have unique ownership
            Err(arc) => (*arc).clone(), // Clone the inner Buffer as the Arc is shared
        };

        let rank = self.original_shape.len();
        if grad_output_shape.len() != rank {
            return Err(NeuraRustError::RankMismatch { expected: rank, actual: grad_output_shape.len() });
        }

        // Iterate through logical elements of grad_output
        for grad_output_logical_idx in 0..grad_output_guard.numel() {
            let grad_output_coords = crate::tensor::utils::index_to_coord(grad_output_logical_idx, grad_output_shape);
            
            // Calculate corresponding coords in the original input space
            let mut input_coords = vec![0; rank];
            let mut is_valid_coord = true;
            for dim in 0..rank {
                let range = self.ranges[dim]; // Get the SliceRange for this dimension
                if grad_output_coords[dim] >= range.size { // Check bounds based on slice size
                     is_valid_coord = false; // Should not happen if shapes match
                     break;
                }
                input_coords[dim] = range.start + grad_output_coords[dim] * range.step;
                if input_coords[dim] >= self.original_shape[dim] { // Check bounds against original shape
                    is_valid_coord = false; // Should not happen if ranges were validated correctly
                    break;
                }

            }

            if !is_valid_coord {
                // This indicates an internal logic error or issue with shape/range validation upstream
                eprintln!("Warning: Calculated input coordinate out of bounds during slice backward.");
                continue; // Or return an error?
            }

            // Calculate physical offsets
            let grad_output_physical_offset = grad_output_offset + grad_output_coords.iter().zip(grad_output_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
            let grad_input_physical_offset = input_offset + input_coords.iter().zip(input_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();

            // Perform scatter-add based on DType using the owned, mutable buffer
            match (&mut grad_input_buffer_owned, &**grad_output_buffer) {
                (Buffer::Cpu(CpuBuffer::F32(grad_in_arc_mut)), Buffer::Cpu(CpuBuffer::F32(grad_out_arc))) => {
                    // Clone-on-write for the inner Vec<f32>
                    let grad_in_vec = Arc::make_mut(grad_in_arc_mut);
                    let grad_out_vec: &Vec<f32> = grad_out_arc;
                    if grad_output_physical_offset < grad_out_vec.len() && grad_input_physical_offset < grad_in_vec.len() {
                        grad_in_vec[grad_input_physical_offset] += grad_out_vec[grad_output_physical_offset];
                    } else {
                        return Err(NeuraRustError::InternalError("Offset out of bounds during F32 slice backward add".to_string()));
                    }
                }
                (Buffer::Cpu(CpuBuffer::F64(grad_in_arc_mut)), Buffer::Cpu(CpuBuffer::F64(grad_out_arc))) => {
                    // Clone-on-write for the inner Vec<f64>
                    let grad_in_vec = Arc::make_mut(grad_in_arc_mut);
                    let grad_out_vec: &Vec<f64> = grad_out_arc;
                     if grad_output_physical_offset < grad_out_vec.len() && grad_input_physical_offset < grad_in_vec.len() {
                        grad_in_vec[grad_input_physical_offset] += grad_out_vec[grad_output_physical_offset];
                    } else {
                         return Err(NeuraRustError::InternalError("Offset out of bounds during F64 slice backward add".to_string()));
                    }
                }
                // Handle other DType combinations or errors later
                _ => return Err(NeuraRustError::DataTypeMismatch { 
                    expected: input_dtype, 
                    actual: grad_output.dtype(), 
                    operation: "SliceBackward scatter-add".to_string()
                 }),
            }
        }
        
        // Update the buffer in the guard with the potentially modified owned buffer
        // We need grad_input as a Tensor, not just its buffer
        // Let's reconstruct grad_input Tensor using the potentially updated buffer
        let grad_input = Tensor { data: Arc::new(RwLock::new(TensorData::new_view(
            Arc::new(grad_input_buffer_owned),
            _input_device,
            0,
            self.original_shape.clone(),
            crate::tensor::utils::calculate_strides(&self.original_shape),
        ).map_err(|e| NeuraRustError::InternalError(format!("Failed to create grad_input tensor view: {}", e)))?)) };

        
        // Drop guards before accumulating gradient
        drop(grad_output_guard);
        drop(input_node_guard);
        
        // Return the calculated gradient for the input
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Creates a view of the tensor representing a slice along multiple dimensions.
///
/// This is a crate-internal function, typically called via the `Tensor::slice` method.
/// It takes an input tensor and a slice of `SliceArg` enums, returning a new tensor
/// view without copying data.
///
/// **Note:** The parsing logic for `SliceArg` (handling Ellipsis, negative indices,
/// NewAxis) is complex and currently marked as TODO/partially implemented in the source.
/// The backward pass assumes valid `SliceRange` are computed.
///
/// # Arguments
///
/// * `input` - The tensor to slice.
/// * `ranges` - A slice of `SliceArg` specifying the slice for each dimension.
///   The number and interpretation of args must be compatible with the tensor's rank.
///
/// # Returns
///
/// A `Result` containing the sliced `Tensor` view. Returns an error if:
/// *   Slice arguments are invalid or incompatible with the tensor shape.
/// *   Memory allocation or autograd operations fail.
///
/// # Example (Conceptual - Use `Tensor::slice` instead)
///
/// ```rust,ignore
/// // Assuming t is a Tensor of shape [5, 10]
/// // use crate::ops::view::slice::slice_op;
/// // use crate::ops::view::slice::SliceArg;
///
/// // Slice corresponding to t[1:4:2, 5:]
/// let slice_args = [
///     SliceArg::Slice(1, 4, 2), // Dimension 0: start=1, end=4 (exclusive), step=2 -> indices 1, 3
///     SliceArg::Slice(5, 10, 1), // Dimension 1: start=5, end=10 (exclusive), step=1 -> indices 5, 6, 7, 8, 9
/// ];
/// let sliced_view = slice_op(&t, &slice_args)?;
/// // sliced_view will have shape [2, 5]
///
/// // Slice corresponding to t[..., 0] (last element along first axis)
/// let slice_args_ellipsis = [
///     SliceArg::Ellipsis,
///     SliceArg::Index(0),
/// ];
/// let sliced_view_ellipsis = slice_op(&t, &slice_args_ellipsis)?;
/// // sliced_view_ellipsis will have shape [5]
/// ```
/// assert_eq!(slice.get_f32_data().unwrap(), vec![6.0, 7.0, 10.0, 11.0]);
/// # Ok::<(), NeuraRustError>(())
/// # }
/// // Example ignored as doc-test: illustrative purpose
/// ```rust, ignore
/// use neurarust_core::{tensor::Tensor, error::NeuraRustError, slice};
/// use neurarust_core::ops::view::slice_op;
///
pub fn slice_op(
    input: &Tensor,
    ranges: &[SliceArg],
) -> Result<Tensor, NeuraRustError> {
    let input_data_guard = input.data.read().unwrap();

    // --- Parse SliceArgs and Validate --- (Needs implementation)
    // This part needs to handle Ellipsis, negative indices, NewAxis etc.
    // and convert SliceArg into SliceRange for each dimension.
    // For now, assume a simplified scenario where SliceArg is already SliceRange
    // and matches the rank.

    // Placeholder: Simulate parsing/validation
    let rank = input_data_guard.shape.len();
    if ranges.len() != rank { // Basic rank check
        return Err(NeuraRustError::SliceError { message: format!("Number of slice args ({}) does not match tensor rank ({})", ranges.len(), rank) });
    }

    let mut parsed_ranges = Vec::with_capacity(rank);
    let mut new_shape = Vec::with_capacity(rank);
    let mut new_strides = Vec::with_capacity(rank);
    let mut new_offset = input_data_guard.offset;

    for (dim, slice_arg) in ranges.iter().enumerate() {
        let dim_size = input_data_guard.shape[dim];
        let stride = input_data_guard.strides[dim];

        // --- Use the new normalize_slice function --- 
        let range = super::utils::normalize_slice(*slice_arg, dim_size)?;
        // --- Remove old conversion and validation logic --- 
        /*
        let range = match slice_arg {
            SliceArg::Slice(start, end, step) => { // Assume non-negative for now
                // ... (old logic) ...
            }
             SliceArg::Index(_) => return Err(NeuraRustError::SliceError{message: "Indexing not yet supported by this simplified slice_op".to_string()}),
             SliceArg::Ellipsis => return Err(NeuraRustError::SliceError{message: "Ellipsis not yet supported by this simplified slice_op".to_string()}),
             SliceArg::NewAxis => return Err(NeuraRustError::SliceError{message: "NewAxis not yet supported by this simplified slice_op".to_string()}),
        };
        
        if range.end > dim_size || range.start >= range.end && range.size > 0 { 
             return Err(NeuraRustError::SliceError { message: format!("Slice range {:?} out of bounds for dimension {} with size {}", range, dim, dim_size) });
        }
        */

        parsed_ranges.push(range);
        new_shape.push(range.size);
        new_strides.push(stride * range.step);
        new_offset += range.start * stride;
    }
    
    let ranges_vec = parsed_ranges; // Use the parsed ranges for backward

    // --- Prepare Data for View --- 
    let buffer_arc = Arc::clone(&input_data_guard.buffer);
    let device = input_data_guard.device;
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };
    let original_shape_clone = input_data_guard.shape.clone(); // For backward

    // Drop the read lock
    drop(input_data_guard);

    // Create the view TensorData (now returns Result)
    let view_td = TensorData::new_view(buffer_arc, device, new_offset, new_shape, new_strides)?;

    // Create the output tensor
    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    // --- Autograd Setup ---
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                lock_type: "write".to_string(),
                reason: "Failed to lock output TensorData for write (autograd setup in slice_op)".to_string(),
            })?;
            output_data_write_guard.requires_grad = true;
            let backward_op = SliceBackward {
                 input_node: node_arc, // Use the correct variable name 'input_node_arc'
                 original_shape: original_shape_clone, // Pass original shape
                 ranges: ranges_vec, // Pass calculated SliceRange vector
             };
            output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in slice_op".to_string()));
        }
    }

    Ok(output_tensor)
}

// Link the external tests file
#[cfg(test)]
#[path = "slice_test.rs"] mod tests; 