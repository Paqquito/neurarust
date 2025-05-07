use crate::{
    tensor::Tensor,
    error::NeuraRustError,
    types::DType,
    // device::StorageDevice, // Likely unused
    // tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64}, // Unused
    // tensor::utils::{broadcast_shapes, index_to_coord}, // Unused
    ops::traits::numeric::NeuraNumeric,
    // super::inplace_ops::{
    //     add::add_impl,
    //     add_scalar::add_scalar_impl,
    //     clamp::clamp_impl,
    //     div::div_impl,
    //     div_scalar::div_scalar_impl,
    //     mul::mul_impl,
    //     mul_scalar::mul_scalar_impl,
    //     pow::pow_f_impl,
    //     sub::sub_impl,
    //     sub_scalar::sub_scalar_impl,
    // },
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
        if self.dtype() != DType::F32 {
            return Err(NeuraRustError::DataTypeMismatch {
                operation: "pow_f32".to_string(),
                expected: DType::F32,
                actual: self.dtype(),
            });
        }
        crate::tensor::inplace_ops::pow::perform_pow_inplace(self, exponent)
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
        if self.dtype() != DType::F64 {
            return Err(NeuraRustError::DataTypeMismatch {
                operation: "pow_f64".to_string(),
                expected: DType::F64,
                actual: self.dtype(),
            });
        }
        crate::tensor::inplace_ops::pow::perform_pow_inplace(self, exponent)
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

    /// Divides each element of this tensor by a scalar, in-place.
    ///
    /// `self /= scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F32`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f32` scalar to divide by. Must not be zero.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * `scalar` is zero (`ArithmeticError`).
    ///     * The tensor's DType is not `F32`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new(vec![10.0f32, 20.0, 30.0], vec![3])?;
    /// # a.div_scalar_f32(10.0f32)?;
    /// # assert_eq!(a.get_f32_data().unwrap(), &[1.0, 2.0, 3.0]);
    /// #
    /// # let mut b = Tensor::new(vec![1.0f32], vec![1])?;
    /// # assert!(matches!(b.div_scalar_f32(0.0f32), Err(NeuraRustError::ArithmeticError(_))));
    /// # Ok(())
    /// # }
    /// ```
    pub fn div_scalar_f32(&mut self, scalar: f32) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::div_scalar::perform_div_scalar_inplace_f32(self, scalar)
    }

    /// Divides each element of this tensor by a scalar, in-place.
    ///
    /// `self /= scalar` (element-wise)
    ///
    /// This operation modifies the tensor's data directly.
    /// The tensor must be of DType `F64`.
    ///
    /// # Arguments
    ///
    /// * `scalar`: The `f64` scalar to divide by. Must not be zero.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(NeuraRustError)` if:
    ///     * `scalar` is zero (`ArithmeticError`).
    ///     * The tensor's DType is not `F64`.
    ///     * `self` requires gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neurarust_core::{Tensor, NeuraRustError, DType};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// # let mut a = Tensor::new_f64(vec![10.0, 20.0, 30.0], vec![3])?;
    /// # a.div_scalar_f64(10.0)?;
    /// # assert_eq!(a.get_f64_data().unwrap(), &[1.0, 2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn div_scalar_f64(&mut self, scalar: f64) -> Result<(), NeuraRustError> {
        crate::tensor::inplace_ops::div_scalar::perform_div_scalar_inplace_f64(self, scalar)
    }

    /// Performs in-place clamping of tensor elements.
    ///
    /// Each element `x` in the tensor will be clamped to be within the range `[min, max]`.
    /// If `min` is `None`, there is no lower bound.
    /// If `max` is `None`, there is no upper bound.
    ///
    /// The types of `min` and `max` (`S`) must be convertible to the tensor's element type.
    /// Currently, this operation is supported for `F32` and `F64` tensors.
    /// `S` is constrained by `NeuraNumeric`, which is currently implemented for `f32` and `f64`.
    ///
    /// # Arguments
    ///
    /// * `self`: A mutable reference to the tensor.
    /// * `min`: An optional minimum value. Values of type `S: NeuraNumeric`.
    /// * `max`: An optional maximum value. Values of type `S: NeuraNumeric`.
    ///
    /// # Errors
    ///
    /// * `NeuraRustError::InternalError`: If conversion from `S` to the tensor's `DType` fails unexpectedly
    ///   (should not happen if `S` is `f32` or `f64` as per `NeuraNumeric` current impls).
    /// * `NeuraRustError::InplaceModificationError`: If the tensor is not suitable for in-place modification.
    /// * `NeuraRustError::UnsupportedOperation`: If the tensor's `DType` is not `F32` or `F64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neurarust_core::{Tensor, DType, NeuraRustError, ops::traits::numeric::NeuraNumeric};
    /// # fn main() -> Result<(), NeuraRustError> {
    /// let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    /// tensor.clamp_(Some(2.5f32), Some(4.5f32))?;
    /// assert_eq!(tensor.get_f32_data()?, vec![2.5, 2.5, 3.0, 4.0, 4.5]);
    ///
    /// let mut tensor2 = Tensor::new_f64(vec![-1.0, 0.0, 1.0, 2.0], vec![4])?;
    /// tensor2.clamp_(Some(0.0f64), None)?;
    /// assert_eq!(tensor2.get_f64_data()?, vec![0.0, 0.0, 1.0, 2.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn clamp_<S: NeuraNumeric>(
        &mut self,
        min: Option<S>,
        max: Option<S>,
    ) -> Result<&mut Self, NeuraRustError> {
        if self.grad_fn().is_some() || (self.grad_fn().is_none() && self.requires_grad()) {
            return Err(NeuraRustError::InplaceModificationError {
                operation: "clamp_".to_string(),
                reason: "In-place operation is not allowed on a non-leaf tensor or a leaf tensor that requires grad.".to_string(),
            });
        }

        // Match DType. Since DType currently only has F32 and F64, this match is exhaustive.
        // If new DTypes are added, the compiler will error here, forcing an update.
        match self.dtype() {
            DType::F32 => {
                let min_val = match min {
                    Some(s_val) => Some(s_val.to_f32().ok_or_else(|| NeuraRustError::InternalError(
                        format!("clamp_: Failed to convert min value of type {} to f32.", std::any::type_name::<S>())
                    ))?),
                    None => None,
                };
                let max_val = match max {
                    Some(s_val) => Some(s_val.to_f32().ok_or_else(|| NeuraRustError::InternalError(
                        format!("clamp_: Failed to convert max value of type {} to f32.", std::any::type_name::<S>())
                    ))?),
                    None => None,
                };
                let mut tensor_data_guard = self.data.write().map_err(|e| NeuraRustError::LockError {
                    lock_type: "write".to_string(),
                    reason: format!("Failed to lock tensor data for clamp_ (F32): {}", e),
                })?;
                crate::tensor::inplace_ops::clamp::clamp_tensor_data::<f32>(&mut *tensor_data_guard, min_val, max_val)?;
            }
            DType::F64 => {
                let min_val = match min {
                    Some(s_val) => Some(s_val.to_f64().ok_or_else(|| NeuraRustError::InternalError(
                        format!("clamp_: Failed to convert min value of type {} to f64.", std::any::type_name::<S>())
                    ))?),
                    None => None,
                };
                let max_val = match max {
                    Some(s_val) => Some(s_val.to_f64().ok_or_else(|| NeuraRustError::InternalError(
                        format!("clamp_: Failed to convert max value of type {} to f64.", std::any::type_name::<S>())
                    ))?),
                    None => None,
                };
                let mut tensor_data_guard = self.data.write().map_err(|e| NeuraRustError::LockError {
                    lock_type: "write".to_string(),
                    reason: format!("Failed to lock tensor data for clamp_ (F64): {}", e),
                })?;
                crate::tensor::inplace_ops::clamp::clamp_tensor_data::<f64>(&mut *tensor_data_guard, min_val, max_val)?;
            }
            // No `_` arm is needed if DType only contains F32 and F64.
            // The compiler will enforce exhaustiveness if DType is extended.
        }
        Ok(self)
    }
}

// The test module declaration previously here is now removed,
// as tests are handled by `#[cfg(test)] mod inplace_ops_tests;` in `tensor/mod.rs` 