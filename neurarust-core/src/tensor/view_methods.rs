use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::types::DType; // Corrected DType import
use crate::ops::view::contiguous::ContiguousBackward;
use crate::autograd::graph::NodeId;
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
    /// [`ops::view::slice::slice_op`](../ops/view/fn.slice_op.html).
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
        crate::ops::view::slice::slice_op(self, ranges)
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
            let _requires_grad = a_guard.requires_grad;
            let _a_node_id: NodeId = Arc::as_ptr(&self.data);
            let _output_shape = self.shape();
            let _strides = self.strides();
            let _offset = a_guard.offset;
            let _numel = a_guard.numel();
            let _dtype = a_guard.dtype;
            let _device = a_guard.device;

            if _device != StorageDevice::CPU {
                 return Err(NeuraRustError::UnsupportedOperation(
                    "contiguous() currently only supports CPU tensors.".to_string(),
                ));
            }

            let output_tensor = match _dtype {
                DType::F32 => crate::tensor::zeros(&_output_shape)?,
                DType::F64 => crate::tensor::zeros_f64(&_output_shape)?,
                DType::I32 | DType::I64 | DType::Bool => todo!("view_methods: non supporté pour ce DType"),
            };

            if _requires_grad {
                let grad_fn = ContiguousBackward {
                    a_node: _a_node_id,
                };
                let mut output_guard = output_tensor.write_data();
                output_guard.grad_fn = Some(Arc::new(grad_fn));
                output_guard.requires_grad = true;
            }

            Ok(output_tensor)
        }
    }

    /// Flattens a contiguous range of dimensions into a single dimension.
    ///
    /// This method returns a view of the original tensor if the tensor is contiguous.
    ///
    /// # Arguments
    /// * `start_dim_isize`: The first dimension to flatten (inclusive). Supports negative indexing.
    /// * `end_dim_isize`: The last dimension to flatten (inclusive). Supports negative indexing.
    ///
    /// # Returns
    /// A `Result` containing the new flattened tensor (view) or a `NeuraRustError`.
    ///
    /// # Errors
    /// - `NeuraRustError::DimensionError` if `start_dim` > `end_dim` or if dimensions are out of bounds.
    /// - `NeuraRustError::ReshapeError` if the tensor is not contiguous (from the underlying `reshape` call).
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// use neurarust_core::error::NeuraRustError;
    ///
    /// # fn main() -> Result<(), NeuraRustError> {
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    /// let flattened_t = t.flatten(0, 1)?; // Flatten all dimensions
    /// assert_eq!(flattened_t.shape(), &[6]);
    ///
    /// let t2 = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
    /// let flattened_t2 = t2.flatten(1, 2)?; // Flatten dimensions 1 and 2
    /// assert_eq!(flattened_t2.shape(), &[2, 12]);
    ///
    /// let scalar = Tensor::new(vec![42.0], vec![])?;
    /// let flattened_scalar = scalar.flatten(0, 0)?;
    /// assert_eq!(flattened_scalar.shape(), &[1]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn flatten(&self, start_dim_isize: isize, end_dim_isize: isize) -> Result<Tensor, NeuraRustError> {
        let rank = self.rank();

        let start_dim = normalize_dim_for_flatten(start_dim_isize, rank, "start_dim")?;
        let end_dim = normalize_dim_for_flatten(end_dim_isize, rank, "end_dim")?;

        if start_dim > end_dim {
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "start_dim ({} normalized to {}) must be <= end_dim ({} normalized to {})",
                start_dim_isize, start_dim, end_dim_isize, end_dim
            )));
        }

        let mut new_shape_vec: Vec<usize> = Vec::new();

        if rank == 0 {
            new_shape_vec.push(1);
        } else {
            new_shape_vec.extend_from_slice(&self.shape()[0..start_dim]);

            let middle_dim_size: usize = self.shape()[start_dim..=end_dim].iter().product();
            new_shape_vec.push(middle_dim_size);

            if end_dim + 1 < rank {
                new_shape_vec.extend_from_slice(&self.shape()[end_dim + 1..]);
            }
        }
        self.reshape(new_shape_vec)
    }

    /// Adds a new dimension of size 1 at the specified position `dim`.
    ///
    /// This operation returns a new view of the tensor with the additional dimension.
    /// The underlying data is not copied.
    ///
    /// # Arguments
    /// * `dim`: The dimension at which to insert the new axis. Must be within `0 <= dim <= rank`
    ///          (where `rank` is the original rank of the tensor).
    ///
    /// # Returns
    /// A `Result` containing the new `Tensor` view with the unsqueezed dimension, 
    /// or a `NeuraRustError` if `dim` is out of bounds.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// // Shape: [2, 2]
    ///
    /// // Unsqueeze at dimension 0
    /// let unsqueezed_0 = t.unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed_0.shape(), vec![1, 2, 2]);
    ///
    /// // Unsqueeze at dimension 1
    /// let unsqueezed_1 = t.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed_1.shape(), vec![2, 1, 2]);
    ///
    /// // Unsqueeze at dimension 2 (at the end)
    /// let unsqueezed_2 = t.unsqueeze(2).unwrap();
    /// assert_eq!(unsqueezed_2.shape(), vec![2, 2, 1]);
    ///
    /// // Attempting to unsqueeze at an invalid dimension
    /// assert!(t.unsqueeze(3).is_err());
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<Self, NeuraRustError> {
        crate::ops::view::squeeze_unsqueeze::unsqueeze_op(self, dim)
    }

    /// Removes dimensions of size 1 from the shape of a tensor.
    ///
    /// If `dim` is `None`, all dimensions of size 1 are removed.
    /// If `dim` is `Some(d)`, only the dimension `d` is removed if its size is 1.
    /// If the specified dimension `d` is not of size 1, the tensor is returned unchanged (as a new view).
    ///
    /// This operation returns a new view of the tensor; the underlying data is not copied.
    ///
    /// # Arguments
    /// * `dim`: An optional dimension to squeeze. 
    ///          - If `None`, all dimensions of size 1 are removed.
    ///          - If `Some(d)`, only dimension `d` is considered for removal.
    ///
    /// # Returns
    /// A `Result` containing the new `Tensor` view with squeezed dimensions, 
    /// or a `NeuraRustError` if `dim` is out of bounds.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]).unwrap();
    /// // Shape: [1, 3, 1]
    ///
    /// // Squeeze all dimensions of size 1
    /// let squeezed_all = t.squeeze(None).unwrap();
    /// assert_eq!(squeezed_all.shape(), vec![3]);
    ///
    /// // Squeeze dimension 0
    /// let squeezed_0 = t.squeeze(Some(0)).unwrap();
    /// assert_eq!(squeezed_0.shape(), vec![3, 1]);
    ///
    /// // Squeeze dimension 2
    /// let squeezed_2 = t.squeeze(Some(2)).unwrap();
    /// assert_eq!(squeezed_2.shape(), vec![1, 3]);
    ///
    /// // Attempt to squeeze dimension 1 (size 3) - should be no-op for that dim
    /// let squeezed_1 = t.squeeze(Some(1)).unwrap();
    /// assert_eq!(squeezed_1.shape(), vec![1, 3, 1]);
    ///
    /// // Squeeze to scalar
    /// let scalar_t = Tensor::new(vec![5.0f32], vec![1, 1, 1]).unwrap();
    /// let scalar_squeezed = scalar_t.squeeze(None).unwrap();
    /// assert_eq!(scalar_squeezed.shape(), vec![]); // Empty shape for scalar
    /// ```
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Self, NeuraRustError> {
        crate::ops::view::squeeze_unsqueeze::squeeze_op(self, dim)
    }

    /// Creates a new view of this tensor with singleton dimensions (size 1)
    /// expanded to a larger size.
    ///
    /// Expanding a tensor does not allocate new memory, but only creates a new
    /// view on the existing tensor where dimensions with size 1 are expanded
    /// to match the target shape by setting the stride for that dimension to 0.
    /// Any dimension of size -1 in `new_shape` means that the size of that
    /// dimension is kept the same as the original tensor.
    /// The number of dimensions in `new_shape` must be greater than or equal
    /// to the number of dimensions in the original tensor.
    ///
    /// This method delegates the operation (including autograd handling) to
    /// [`ops::view::expand::expand_op`](../ops/view/fn.expand_op.html).
    ///
    /// # Arguments
    /// * `new_shape`: The desired expanded shape. Use `-1` for dimensions
    ///                that should not change size.
    ///
    /// # Returns
    /// A `Result` containing the new expanded `Tensor` view, or a `NeuraRustError`
    /// if the expansion is invalid (e.g., incompatible shapes).
    ///
    /// # Example
    /// ```
    /// # use neurarust_core::tensor::Tensor;
    /// # use neurarust_core::error::NeuraRustError;
    /// # fn main() -> Result<(), NeuraRustError> {
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1])?;
    /// // Shape [3, 1]
    ///
    /// let expanded = t.expand(&[3, 4])?;
    /// assert_eq!(expanded.shape(), vec![3, 4]);
    ///
    /// // Use -1 to keep the first dimension the same
    /// let expanded_neg1 = t.expand(&[-1, 4])?;
    /// assert_eq!(expanded_neg1.shape(), vec![3, 4]);
    ///
    /// // Add a new dimension
    /// let t2 = Tensor::new(vec![4.0], vec![1])?;
    /// let expanded_new_dim = t2.expand(&[2, 1, 5])?;
    /// assert_eq!(expanded_new_dim.shape(), vec![2, 1, 5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn expand(&self, new_shape: &[isize]) -> Result<Self, NeuraRustError> {
        crate::ops::view::expand::expand_op(self, new_shape)
    }
}

// Helper function to normalize dimension indices for flatten
// This function is private to the module
fn normalize_dim_for_flatten(dim: isize, rank: usize, dim_name: &str) -> Result<usize, NeuraRustError> {
    let current_dim: usize;
    if rank == 0 { // Scalar tensor
        if dim == 0 || dim == -1 {
            current_dim = 0;
        } else {
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "{} ({}) out of range for scalar tensor (rank 0). Expected 0 or -1.",
                dim_name, dim
            )));
        }
    } else { // Non-scalar tensor
        if dim < 0 {
            let d_abs = dim.abs() as usize;
            if d_abs == 0 || d_abs > rank { 
                 return Err(NeuraRustError::UnsupportedOperation(format!(
                    "Negative {} ({}) results in an out-of-bounds dimension for tensor of rank {}. Valid range for negative indexing is [-{}, -1].",
                    dim_name, dim, rank, rank
                )));
            }
            current_dim = rank - d_abs;
        } else {
            current_dim = dim as usize;
        }

        if current_dim >= rank {
            return Err(NeuraRustError::InvalidAxis {
                axis: current_dim,
                rank,
            });
        }
    }
    Ok(current_dim)
}

#[cfg(test)]
#[path = "view_methods_test.rs"]
mod tests;
