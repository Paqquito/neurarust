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

    /// Performs in-place division of this tensor by another tensor.
    ///
    /// `self /= other`
    ///
    /// This operation modifies the tensor's data directly.
    /// It supports broadcasting the `other` tensor to the shape of `self`.
    /// It will return an `ArithmeticError` if division by zero occurs.
    ///
    /// # Arguments
    ///
    /// * `other`: The tensor to divide `self` by.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * Division by zero occurs (`ArithmeticError`).
    ///     * The tensors have different data types.
    ///     * Broadcasting is not possible.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2])?;
    /// # let b = Tensor::new(vec![2.0f32, 5.0, 2.0, 10.0], vec![2, 2])?;
    /// # a.div_(&b)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[5.0, 4.0, 15.0, 4.0]);
    /// #
    /// # let mut c = Tensor::new(vec![10.0f32, 20.0], vec![1, 2])?;
    /// # let d = Tensor::new(vec![2.0f32], vec![1])?; // Scalar broadcast
    /// # c.div_(&d)?;
    /// # assert_eq!(c.get_f32_data().unwrap(), &[5.0, 10.0]);
    /// #
    /// # let mut e = Tensor::new(vec![1.0f32], vec![1])?;
    /// # let f = Tensor::new(vec![0.0f32], vec![1])?;
    /// # assert!(matches!(e.div_(&f), Err(NeuraRustError::ArithmeticError(_))));
    /// # Ok(())
    /// # }
    /// ```
    pub fn div_(&mut self, other: &Tensor) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::div::perform_div_inplace(self, other)
    }

    /// Raises the elements of this tensor to the power of a scalar exponent, in-place.
    ///
    /// `self = self ^ exponent` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F32`.
    ///
    /// # Arguments
    ///
    /// * `exponent`: The `f32` scalar exponent.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F32`.
    ///     * `self` requires gradient computation.
    ///     * An arithmetic error occurs (e.g., negative base with a non-integer exponent).
    ///
    /// # Edge Cases
    /// * `0.0^0.0` is treated as `1.0`.
    /// * A negative base with a non-integer exponent will result in `ArithmeticError`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![1.0f32, 2.0, -3.0, 4.0], vec![2, 2])?;
    /// # a.pow_f32(2.0f32)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
    /// #
    /// # let mut b = Tensor::new(vec![0.0f32, 4.0], vec![2])?;
    /// # b.pow_f32(0.0f32)?;
    /// # assert_eq!(b.get_f32_data().unwrap(), &[1.0, 1.0]); // 0^0 = 1, 4^0 = 1
    /// #
    /// # let mut c = Tensor::new(vec![-2.0f32], vec![1])?;
    /// # assert!(matches!(c.pow_f32(0.5f32), Err(NeuraRustError::ArithmeticError(_))));
    /// # Ok(())
    /// # }
    /// ```
    pub fn pow_f32(&mut self, exponent: f32) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::pow::perform_pow_inplace_f32(self, exponent)
    }

    /// Raises the elements of this tensor to the power of a scalar exponent, in-place.
    ///
    /// `self = self ^ exponent` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F64`.
    ///
    /// # Arguments
    ///
    /// * `exponent`: The `f64` scalar exponent.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F64`.
    ///     * `self` requires gradient computation.
    ///     * An arithmetic error occurs (e.g., negative base with a non-integer exponent).
    ///
    /// # Edge Cases
    /// * `0.0^0.0` is treated as `1.0`.
    /// * A negative base with a non-integer exponent will result in `ArithmeticError`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new_f64(vec![1.0, 2.0, -3.0, 4.0], vec![2, 2])?;
    /// # a.pow_f64(2.0)?;
    /// # assert_eq!(a.get_f64_data().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn pow_f64(&mut self, exponent: f64) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::pow::perform_pow_inplace_f64(self, exponent)
    }

    /// Adds a scalar to each element of this tensor, in-place.
    ///
    /// `self += scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F32`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f32` scalar to add.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F32`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3])?;
    /// # a.add_scalar_f32(10.0f32)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[11.0, 12.0, 13.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_scalar_f32(&mut self, scalar: f32) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::add_scalar::perform_add_scalar_inplace_f32(self, scalar)
    }

    /// Adds a scalar to each element of this tensor, in-place.
    ///
    /// `self += scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F64`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f64` scalar to add.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F64`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new_f64(vec![1.0, 2.0, 3.0], vec![3])?;
    /// # a.add_scalar_f64(10.0)?;
    /// # assert_eq!(a.get_f64_data().unwrap(), &[11.0, 12.0, 13.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_scalar_f64(&mut self, scalar: f64) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::add_scalar::perform_add_scalar_inplace_f64(self, scalar)
    }

    /// Subtracts a scalar from each element of this tensor, in-place.
    ///
    /// `self -= scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F32`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f32` scalar to subtract.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F32`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![10.0f32, 20.0, 30.0], vec![3])?;
    /// # a.sub_scalar_f32(5.0f32)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[5.0, 15.0, 25.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sub_scalar_f32(&mut self, scalar: f32) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::sub_scalar::perform_sub_scalar_inplace_f32(self, scalar)
    }

    /// Subtracts a scalar from each element of this tensor, in-place.
    ///
    /// `self -= scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F64`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f64` scalar to subtract.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F64`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new_f64(vec![10.0, 20.0, 30.0], vec![3])?;
    /// # a.sub_scalar_f64(5.0)?;
    /// # assert_eq!(a.get_f64_data().unwrap(), &[5.0, 15.0, 25.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sub_scalar_f64(&mut self, scalar: f64) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::sub_scalar::perform_sub_scalar_inplace_f64(self, scalar)
    }

    /// Multiplies each element of this tensor by a scalar, in-place.
    ///
    /// `self *= scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F32`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f32` scalar to multiply by.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F32`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3])?;
    /// # a.mul_scalar_f32(10.0f32)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[10.0, 20.0, 30.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn mul_scalar_f32(&mut self, scalar: f32) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::mul_scalar::perform_mul_scalar_inplace_f32(self, scalar)
    }

    /// Multiplies each element of this tensor by a scalar, in-place.
    ///
    /// `self *= scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F64`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f64` scalar to multiply by.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * The tensor's DType is not `F64`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new_f64(vec![1.0, 2.0, 3.0], vec![3])?;
    /// # a.mul_scalar_f64(10.0)?;
    /// # assert_eq!(a.get_f64_data().unwrap(), &[10.0, 20.0, 30.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn mul_scalar_f64(&mut self, scalar: f64) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::mul_scalar::perform_mul_scalar_inplace_f64(self, scalar)
    }
}

// The test module declaration previously here is now removed,
// as tests are handled by `#[cfg(test)] mod inplace_ops_tests;` in `tensor/mod.rs` 