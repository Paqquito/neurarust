use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::types::DType; // Need DType
use crate::tensor::iter_utils::{NdArraySimpleIter, NdArraySimpleIterF64};
use crate::ops::view::contiguous::ContiguousBackward; // Importer l'op Backward
use crate::autograd::graph::NodeId; // Importer NodeId
use std::sync::Arc;

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
    /// This operation attempts to create a view without copying data, which is only possible
    /// if the tensor is already **contiguous**.
    ///
    /// If the tensor is not contiguous (e.g., after a `transpose` or `slice`), calling
    /// this method will likely result in an error from the underlying `reshape_op`.
    /// To reshape a non-contiguous tensor, you should first call `.contiguous()` to get
    /// a tensor with a contiguous memory layout, and then call `.reshape()` on the result:
    /// `tensor.contiguous()?.reshape(new_shape)`.
    ///
    /// This method delegates the operation (including autograd handling) to
    /// [`ops::view::reshape_op`](../ops/view/fn.reshape_op.html).
    ///
    /// # Arguments
    /// * `new_shape`: The desired new shape as a `Vec<usize>`.
    ///
    /// # Returns
    /// A `Result` containing the reshaped `Tensor` view, or a `NeuraRustError` if the reshape is invalid
    /// (e.g., number of elements differs, tensor is not contiguous).
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert!(t.is_contiguous());
    ///
    /// // Reshape a contiguous tensor (creates a view)
    /// let t_reshaped = t.reshape(vec![3, 2]).unwrap();
    /// assert_eq!(t_reshaped.shape(), vec![3, 2]);
    /// assert!(t_reshaped.is_contiguous()); // Reshape of contiguous is contiguous
    ///
    /// // Attempt to reshape a non-contiguous tensor directly (likely fails)
    /// let transposed = t.transpose(0, 1).unwrap(); // shape [3, 2], non-contiguous
    /// assert!(!transposed.is_contiguous());
    /// let direct_reshape_result = transposed.reshape(vec![6]);
    /// // This fails because reshape_op requires contiguous input
    /// assert!(direct_reshape_result.is_err());
    ///
    /// // Correct way to reshape a non-contiguous tensor:
    /// let reshaped_copied = transposed.contiguous().unwrap().reshape(vec![6]).unwrap();
    /// assert_eq!(reshaped_copied.shape(), vec![6]);
    /// assert!(reshaped_copied.is_contiguous()); // Result is contiguous after copy
    /// // Data order reflects the transpose before flattening
    /// assert_eq!(reshaped_copied.get_f32_data().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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
            let a_guard = self.read_data();
            let requires_grad = a_guard.requires_grad;
            let a_node_id: NodeId = Arc::as_ptr(&self.data);
            let output_shape = self.shape();
            let strides = self.strides();
            let offset = a_guard.offset;
            let numel = a_guard.numel();
            let dtype = a_guard.dtype;
            let device = a_guard.device;

            if device != StorageDevice::CPU {
                 return Err(NeuraRustError::UnsupportedOperation(
                    "contiguous() currently only supports CPU tensors.".to_string(),
                ));
            }

            let output_tensor = match dtype {
                DType::F32 => {
                    let buffer_arc_ref = a_guard.buffer();
                    let buffer_ref = (&*buffer_arc_ref).try_get_cpu_f32()?;
                    let data_slice = buffer_ref.as_slice();
                    let iter = NdArraySimpleIter::new(
                        data_slice,
                        &output_shape,
                        &strides,
                        offset,
                    )?;
                    let mut new_data: Vec<f32> = Vec::with_capacity(numel);
                    for value in iter {
                        new_data.push(value);
                    }
                    if new_data.len() != numel {
                         return Err(NeuraRustError::InternalError(format!(
                            "Contiguous copy loop resulted in wrong number of elements (F32): expected {}, got {}",
                            numel, new_data.len()
                        )));
                    }
                    drop(a_guard);
                    Tensor::new(new_data, output_shape)?
                }
                DType::F64 => {
                    let buffer_arc_ref = a_guard.buffer();
                    let buffer_ref = (&*buffer_arc_ref).try_get_cpu_f64()?;
                    let data_slice = buffer_ref.as_slice();
                    let iter = NdArraySimpleIterF64::new(
                        data_slice,
                        &output_shape,
                        &strides,
                        offset,
                    )?;
                    let mut new_data: Vec<f64> = Vec::with_capacity(numel);
                    for value in iter {
                        new_data.push(value);
                    }
                     if new_data.len() != numel {
                         return Err(NeuraRustError::InternalError(format!(
                            "Contiguous copy loop resulted in wrong number of elements (F64): expected {}, got {}",
                            numel, new_data.len()
                        )));
                    }
                    drop(a_guard);
                    Tensor::new_f64(new_data, output_shape)?
                }
            };

            if requires_grad {
                let grad_fn = ContiguousBackward {
                    a_node: a_node_id,
                };
                let mut output_guard = output_tensor.write_data();
                output_guard.grad_fn = Some(Arc::new(grad_fn));
                output_guard.requires_grad = true;
            }

            Ok(output_tensor)
        }
    }
}
