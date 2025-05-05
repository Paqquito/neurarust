use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::buffer::{Buffer, CpuBuffer}; // Need Buffer types
use crate::types::DType; // Need DType
use std::fmt::Debug;
 // Keep Debug for recursive helper
 // Keep Arc

// Helper function for recursive multidimensional iteration used by contiguous()
// Made generic over numeric type T
fn copy_non_contiguous_recursive<T>(
    original_guard: &TensorData, // Keep non-generic TensorData ref
    original_data_slice: &[T],  // Generic slice
    new_buffer: &mut Vec<T>,    // Generic output buffer
    current_indices: &mut Vec<usize>,
    current_dim: usize,
) -> Result<(), NeuraRustError>
where
    T: Copy + Debug, // Add required traits
{
    if current_dim == original_guard.shape.len() {
        let original_offset = original_guard.get_offset(current_indices);
        if original_offset >= original_data_slice.len() {
            return Err(NeuraRustError::InternalError(format!(
                "Contiguous copy error: Offset {} out of bounds for buffer len {}",
                original_offset, original_data_slice.len()
            )));
        }
        new_buffer.push(original_data_slice[original_offset]); // Works for any T: Copy
    } else {
        for i in 0..original_guard.shape[current_dim] {
            current_indices[current_dim] = i;
            copy_non_contiguous_recursive(
                original_guard,
                original_data_slice,
                new_buffer,
                current_indices,
                current_dim + 1,
            )?;
        }
    }
    Ok(())
}

/// This `impl` block provides methods for creating views of a `Tensor` or manipulating its shape and layout.
/// Many of these methods return new `Tensor` instances that share the underlying data
/// but have different metadata (shape, strides, offset), making them efficient
/// but potentially resulting in non-contiguous tensors.
impl Tensor {
    /// Creates a view of the tensor by slicing along specified dimensions.
    ///
    /// Slicing allows selecting a portion of the tensor data without copying.
    /// The `ranges` argument specifies the start and end indices for each dimension.
    /// See [`ops::view::SliceArg`](../ops/view/enum.SliceArg.html) for how to define ranges.
    ///
    /// This method delegates the operation (including autograd handling) to
    /// [`ops::view::slice_op`](../ops/view/fn.slice_op.html).
    ///
    /// # Arguments
    /// * `ranges`: A slice of `SliceArg` defining the start and end points for each dimension.
    ///           The length of the slice must match the rank of the tensor.
    ///
    /// # Returns
    /// A `Result` containing the new `Tensor` view, or a `NeuraRustError` if slicing is invalid.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// use neurarust_core::ops::view::SliceArg;
    ///
    /// let t = Tensor::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), vec![2, 3, 4]).unwrap();
    /// // Shape [2, 3, 4]
    ///
    /// // Slice the first element along dim 0 (using Slice(0, 1, 1) to keep the dimension),
    /// // all elements along dim 1, and elements 1 to 2 (exclusive) along dim 2
    /// // Use Slice(0, 3, 1) for the full slice [0:3:1] along dimension 1 (size 3)
    /// // Use Slice(1, 3, 1) for the slice [1:3:1] along dimension 2 (size 4)
    /// let sliced = t.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 3, 1), SliceArg::Slice(1, 3, 1)]).unwrap();
    /// assert_eq!(sliced.shape(), vec![1, 3, 2]); // Shape reflects slice (Slice keeps dim but adjusts size)
    /// // Expected data: Elements at [0, 0, 1], [0, 0, 2], [0, 1, 1], [0, 1, 2], [0, 2, 1], [0, 2, 2]
    /// // Original indices: 1, 2, 5, 6, 9, 10
    /// assert_eq!(sliced.get_f32_data().unwrap(), vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
    /// assert!(!sliced.is_contiguous()); // Slicing often creates non-contiguous views
    /// ```
    pub fn slice(&self, ranges: &[crate::ops::view::SliceArg]) -> Result<Self, NeuraRustError> {
        // Reactivate the call to the underlying slice_op function
        crate::ops::view::slice_op(self, ranges)
    }

    /// Creates a view of the tensor with two specified dimensions transposed (swapped).
    ///
    /// This operation does not move data in memory; it only changes the tensor's strides.
    /// The resulting tensor shares the same underlying data but is usually non-contiguous.
    ///
    /// This method delegates the operation (including autograd handling) to
    /// [`ops::view::transpose_op`](../ops/view/fn.transpose_op.html).
    ///
    /// # Arguments
    /// * `dim1`: The first dimension to transpose.
    /// * `dim2`: The second dimension to transpose.
    ///
    /// # Returns
    /// A `Result` containing the new transposed `Tensor` view, or a `NeuraRustError` if dimensions are invalid.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// // [[1., 2., 3.], [4., 5., 6.]]
    ///
    /// let t_transposed = t.transpose(0, 1).unwrap();
    /// assert_eq!(t_transposed.shape(), vec![3, 2]);
    /// // Data remains [1, 2, 3, 4, 5, 6], but access is transposed:
    /// // [[1., 4.], [2., 5.], [3., 6.]] (logical view)
    /// assert!(!t_transposed.is_contiguous());
    /// // Verify logical data order using get_f32_data which handles non-contiguity
    /// assert_eq!(t_transposed.get_f32_data().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, NeuraRustError> {
        crate::ops::view::transpose_op(self, dim1, dim2)
    }

    /// Creates a view of the tensor with dimensions permuted according to the specified order.
    ///
    /// Similar to `transpose` but allows rearranging multiple dimensions at once.
    /// This operation only changes the tensor's strides and does not move data.
    /// The resulting tensor shares the same underlying data and is often non-contiguous.
    ///
    /// This method delegates the operation (including autograd handling) to
    /// [`ops::view::permute_op`](../ops/view/fn.permute_op.html).
    ///
    /// # Arguments
    /// * `dims`: A slice of `usize` specifying the new order of dimensions. It must be a permutation
    ///           of `0..rank` where `rank` is the number of dimensions of the tensor.
    ///
    /// # Returns
    /// A `Result` containing the new permuted `Tensor` view, or a `NeuraRustError` if the dimensions are invalid.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), vec![2, 3, 4]).unwrap();
    ///
    /// let t_permuted = t.permute(&[2, 0, 1]).unwrap(); // New order: old dim 2, old dim 0, old dim 1
    /// assert_eq!(t_permuted.shape(), vec![4, 2, 3]);
    /// assert!(!t_permuted.is_contiguous());
    /// // You can verify the logical order using get_f32_data if needed.
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Result<Self, NeuraRustError> {
        crate::ops::view::permute_op(self, dims)
    }

    /// Creates a view of the tensor with a different shape, possibly changing the number of dimensions.
    ///
    /// The total number of elements must remain the same.
    /// This operation attempts to reuse the existing data buffer without copying.
    /// It can only succeed without copying if the tensor is contiguous or if the new shape
    /// can be achieved by manipulating strides compatibly with the existing layout.
    /// If a copy is required (e.g., reshaping a non-contiguous tensor into an incompatible shape),
    /// the underlying `reshape_op` might handle it by implicitly calling `contiguous()` first (TBC).
    ///
    /// This method delegates the operation (including autograd handling) to
    /// [`ops::view::reshape_op`](../ops/view/fn.reshape_op.html).
    ///
    /// # Arguments
    /// * `new_shape`: The desired new shape as a `Vec<usize>`.
    ///
    /// # Returns
    /// A `Result` containing the reshaped `Tensor`, or a `NeuraRustError` if the reshape is invalid
    /// (e.g., number of elements differs, incompatible layout).
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// let t_reshaped = t.reshape(vec![3, 2]).unwrap();
    /// assert_eq!(t_reshaped.shape(), vec![3, 2]);
    /// // Reshaping a contiguous tensor usually results in a contiguous view
    /// assert!(t_reshaped.is_contiguous());
    /// assert_eq!(t_reshaped.get_f32_data().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///
    /// let t_flattened = t.reshape(vec![6]).unwrap();
    /// assert_eq!(t_flattened.shape(), vec![6]);
    ///
    /// // Reshaping a non-contiguous tensor might require a copy internally (handled by op)
    /// let transposed = t.transpose(0, 1).unwrap(); // shape [3, 2], non-contiguous
    /// // We need to make it contiguous before reshaping if the op doesn't handle it implicitly
    /// let reshaped_from_transposed = transposed.contiguous().unwrap().reshape(vec![6]); // Attempt to flatten
    /// assert!(reshaped_from_transposed.is_ok());
    /// let reshaped_tensor = reshaped_from_transposed.unwrap();
    /// // The result should be contiguous now because we called contiguous()
    /// assert!(reshaped_tensor.is_contiguous());
    /// // Check data order after transpose + contiguous + reshape
    /// assert_eq!(reshaped_tensor.get_f32_data().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        crate::ops::view::reshape_op(self, new_shape)
    }

    /// Returns a tensor with the same data but guaranteed to be contiguous in memory.
    ///
    /// If the original tensor (`self`) is already contiguous (i.e., `self.is_contiguous()` is true),
    /// this method returns a shallow clone of the tensor, which is very efficient.
    ///
    /// If the original tensor is not contiguous (e.g., it's a result of `transpose`, `permute`,
    /// or certain `slice` operations), this method performs the following:
    /// 1. Allocates a new buffer of the correct size.
    /// 2. Copies the elements from the original tensor's view into the new buffer in
    ///    standard row-major (contiguous) order.
    /// 3. Creates a new `Tensor` pointing to this new buffer, with contiguous strides.
    ///
    /// **Note:** If the original tensor required gradients, the returned contiguous tensor
    /// will also require gradients and will have a `grad_fn` linking it back to the
    /// original tensor (or the copy operation). Gradient computation will flow correctly.
    ///
    /// Currently supports CPU tensors with `DType::F32` and `DType::F64`.
    ///
    /// # Returns
    /// A `Result` containing the contiguous `Tensor`, or a `NeuraRustError` if:
    /// - Locking the internal data fails (`LockError`).
    /// - An internal error occurs during data copying (`InternalError`).
    /// - The operation is attempted on an unsupported device (e.g., GPU) or data type (`UnsupportedOperation`).
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert!(t.is_contiguous());
    /// let tc = t.contiguous().unwrap();
    /// assert!(tc.is_contiguous());
    /// // assert!(std::ptr::eq(t.data.as_ref(), tc.data.as_ref())); // Should point to same Arc<RwLock<TensorData>>
    ///
    /// let transposed = t.transpose(0, 1).unwrap();
    /// assert!(!transposed.is_contiguous());
    /// let transposed_c = transposed.contiguous().unwrap();
    /// assert!(transposed_c.is_contiguous());
    /// // assert!(!std::ptr::eq(transposed.data.as_ref(), transposed_c.data.as_ref())); // Should be different data
    /// assert_eq!(transposed_c.get_f32_data().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // Data is now in logical contiguous order
    /// ```
    pub fn contiguous(&self) -> Result<Self, NeuraRustError> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let guard = self.data.read().map_err(|_| NeuraRustError::LockError{
                lock_type: "read".to_string(),
                reason: "Failed to lock for contiguous()".to_string()
            })?;
            let td_ref = &*guard;

            let device = td_ref.device;
            let shape = td_ref.shape.clone();
            let numel = td_ref.numel();

            // Dispatch based on dtype and device
            match (td_ref.dtype, device) {
                (DType::F32, StorageDevice::CPU) => {
                    let mut new_buffer_vec = Vec::with_capacity(numel);
                    match &*td_ref.buffer {
                        Buffer::Cpu(CpuBuffer::F32(original_cpu_data_arc)) => {
                            let original_f32_data: &[f32] = original_cpu_data_arc;
                            let mut current_indices = vec![0; shape.len()];
                            // Call generic recursive function
                            copy_non_contiguous_recursive(
                                td_ref,
                                original_f32_data,
                                &mut new_buffer_vec,
                                &mut current_indices,
                                0,
                            )?;
                        }
                        _ => return Err(NeuraRustError::InternalError("Mismatched buffer type for F32 dtype in contiguous()".to_string()))
                    }
                    drop(guard);
                    Tensor::new(new_buffer_vec, shape)
                }
                (DType::F64, StorageDevice::CPU) => {
                    let mut new_buffer_vec: Vec<f64> = Vec::with_capacity(numel);
                    match &*td_ref.buffer {
                        Buffer::Cpu(CpuBuffer::F64(original_cpu_data_arc)) => {
                            let original_f64_data: &[f64] = original_cpu_data_arc;
                            let mut current_indices = vec![0; shape.len()];
                            // Call generic recursive function for F64
                            copy_non_contiguous_recursive(
                                td_ref,
                                original_f64_data,
                                &mut new_buffer_vec,
                                &mut current_indices,
                                0,
                            )?;
                        }
                        _ => return Err(NeuraRustError::InternalError("Mismatched buffer type for F64 dtype in contiguous()".to_string()))
                    }
                    drop(guard);
                    // Call F64 constructor
                    Tensor::new_f64(new_buffer_vec, shape)
                }
                 // TODO: Add cases for other DTypes (e.g., I64) later
                 // (DType::I64, StorageDevice::CPU) => { ... }
                (dtype, StorageDevice::GPU) => {
                    Err(NeuraRustError::UnsupportedOperation(
                        format!("GPU contiguous copy not yet implemented for dtype {:?}", dtype)
                    ))
                }
            }
        }
    }
}
