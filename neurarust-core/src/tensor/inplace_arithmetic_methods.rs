use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    // device::StorageDevice, // Likely unused
    // types::DType,          // Likely unused
    // tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64}, // Unused
    // tensor::utils::{broadcast_shapes, index_to_coord}, // Unused
};
// use std::sync::Arc; // Unused
// use std::ops::{Deref, DerefMut}; // Unused

// The module `inplace_ops` is declared in `tensor/mod.rs`
// No need for `pub mod inplace_ops;` here.

impl Tensor {
    /// Performs an in-place addition of another tensor to this tensor (`self += other`).
    ///
    /// The operation supports broadcasting of the `other` tensor to the shape of `self`.
    /// The `self` tensor is modified directly.
    ///
    /// # Arguments
    ///
    /// * `other`: The tensor to add to `self`.
    ///
    /// # Errors
    ///
    /// Returns `NeuraRustError::InplaceModificationError` if `self` requires gradients.
    /// Returns `NeuraRustError::DeviceMismatch` if `self` and `other` are on different devices.
    /// Returns `NeuraRustError::DataTypeMismatch` if `self` and `other` have different data types.
    /// Returns `NeuraRustError::ShapeMismatch` if `other` cannot be broadcast to the shape of `self`.
    /// Returns other errors related to buffer access or internal operations.
    pub fn add_(&mut self, other: &Tensor) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::add::perform_add_inplace(self, other)
    }

    /// Performs in-place subtraction of another tensor from this tensor.
    ///
    /// `self -= other`
    ///
    /// This operation modifies the tensor's data directly.
    /// It supports broadcasting the `other` tensor to the shape of `self`.
    ///
    /// # Arguments
    ///
    /// * `other`: The tensor to subtract from `self`.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensors have different data types.
    ///     * Broadcasting is not possible (i.e., `other` cannot be broadcast to `self`'s shape, or `self`'s shape would change).
    ///     * `self` requires gradient computation and is a leaf node or part of a graph.
    ///     * The underlying buffer of `self` cannot be uniquely (mutably) accessed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
    /// # let b = Tensor::new(vec![1.0f32, 1.0, 1.0, 1.0], vec![2, 2])?;
    /// # a.sub_(&b)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[0.0, 1.0, 2.0, 3.0]);
    /// #
    /// # let mut c = Tensor::new(vec![1.0f32, 2.0], vec![1, 2])?;
    /// # let d = Tensor::new(vec![1.0f32], vec![1])?; // Scalar broadcast
    /// # c.sub_(&d)?;
    /// # assert_eq!(c.get_f32_data().unwrap(), &[0.0, 1.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sub_(&mut self, other: &Tensor) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::sub::perform_sub_inplace(self, other)
    }

    /// Performs in-place multiplication of this tensor by another tensor.
    ///
    /// `self *= other`
    ///
    /// This operation modifies the tensor's data directly.
    /// It supports broadcasting the `other` tensor to the shape of `self`.
    ///
    /// # Arguments
    ///
    /// * `other`: The tensor to multiply `self` by.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensors have different data types.
    ///     * Broadcasting is not possible (i.e., `other` cannot be broadcast to `self`'s shape, or `self`'s shape would change).
    ///     * `self` requires gradient computation and is a leaf node or part of a graph.
    ///     * The underlying buffer of `self` cannot be uniquely (mutably) accessed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
    /// # let b = Tensor::new(vec![2.0f32, 2.0, 2.0, 2.0], vec![2, 2])?;
    /// # a.mul_(&b)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
    /// #
    /// # let mut c = Tensor::new(vec![1.0f32, 2.0], vec![1, 2])?;
    /// # let d = Tensor::new(vec![3.0f32], vec![1])?; // Scalar broadcast
    /// # c.mul_(&d)?;
    /// # assert_eq!(c.get_f32_data().unwrap(), &[3.0, 6.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn mul_(&mut self, other: &Tensor) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::mul::perform_mul_inplace(self, other)
    }
}

// The test module declaration previously here is now removed,
// as tests are handled by `#[cfg(test)] mod inplace_ops_tests;` in `tensor/mod.rs` 