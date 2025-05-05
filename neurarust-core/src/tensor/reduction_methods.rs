use crate::{tensor::Tensor, error::NeuraRustError};

/// This `impl` block provides methods for performing reduction operations on a `Tensor`.
/// Reduction operations reduce the number of dimensions of a tensor by applying
/// an operation (like mean, max, sum) across specified axes.
impl Tensor {
    /// Computes the mean of the tensor elements over specified axes.
    ///
    /// If `axes` is `None`, the mean is computed over all elements, resulting in a scalar tensor.
    /// If `axes` is `Some`, the mean is computed along the specified dimensions.
    ///
    /// The `keep_dims` argument controls whether the reduced dimensions are kept with size 1
    /// or removed entirely.
    ///
    /// This method delegates the computation (including autograd handling) to
    /// [`ops::reduction::mean::mean_op`](../ops/reduction/mean/fn.mean_op.html).
    ///
    /// # Arguments
    /// * `axes`: An optional slice of `usize` specifying the axes along which to compute the mean.
    /// * `keep_dims`: If `true`, the reduced axes are retained in the output shape with size 1.
    ///                If `false`, the reduced axes are removed.
    ///
    /// # Returns
    /// A `Result` containing the resulting `Tensor` with the mean values, or a `NeuraRustError`.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// // [[1., 2., 3.], [4., 5., 6.]]
    ///
    /// // Mean over all elements
    /// let m_all = t.mean(None, false).unwrap();
    /// assert_eq!(m_all.shape(), vec![]); // Scalar shape
    /// assert_eq!(m_all.item_f32().unwrap(), 3.5); // (1+..+6)/6
    ///
    /// // Mean along axis 0 (collapse rows)
    /// let m_axis0 = t.mean(Some(&[0]), false).unwrap();
    /// assert_eq!(m_axis0.shape(), vec![3]);
    /// assert_eq!(m_axis0.get_f32_data().unwrap(), vec![2.5, 3.5, 4.5]); // [(1+4)/2, (2+5)/2, (3+6)/2]
    ///
    /// // Mean along axis 1 (collapse columns), keeping dimension
    /// let m_axis1_keep = t.mean(Some(&[1]), true).unwrap();
    /// assert_eq!(m_axis1_keep.shape(), vec![2, 1]);
    /// assert_eq!(m_axis1_keep.get_f32_data().unwrap(), vec![2.0, 5.0]); // [(1+2+3)/3, (4+5+6)/3]
    ///
    /// assert_eq!(t.mean(Some(&[0]), true).unwrap().shape(), &[1, 3]);
    /// ```
    pub fn mean(&self, axes: Option<&[usize]>, keep_dims: bool) -> Result<Self, NeuraRustError> {
        // La logique pour déterminer les axes réels (tous si None) est gérée DANS mean_op.
        // Il suffit donc de passer l'Option directement.
        // let all_axes: Vec<usize> = (0..self.rank()).collect();
        // let axes_to_reduce: &[usize] = match axes {
        //     Some(a) => a,
        //     None => &all_axes,
        // };
        // mean_op(self, Some(axes_to_reduce), keep_dims) // Incorrect
        crate::ops::reduction::mean::mean_op(self, axes, keep_dims) // Passer l'Option originale
    }

    /// Computes the maximum of the tensor elements over specified axes.
    ///
    /// If `axes` is `None`, the maximum is computed over all elements, resulting in a scalar tensor.
    /// If `axes` is `Some`, the maximum is computed along the specified dimensions.
    ///
    /// The `keep_dims` argument controls whether the reduced dimensions are kept with size 1
    /// or removed entirely.
    ///
    /// This method delegates the computation (including autograd handling) to
    /// [`ops::reduction::max::max_op`](../ops/reduction/max/fn.max_op.html).
    /// **Note:** The gradient definition for `max` is often such that the gradient flows only
    /// to the element(s) that held the maximum value.
    ///
    /// # Arguments
    /// * `axes`: An optional slice of `usize` specifying the axes along which to compute the maximum.
    /// * `keep_dims`: If `true`, the reduced axes are retained in the output shape with size 1.
    ///                If `false`, the reduced axes are removed.
    ///
    /// # Returns
    /// A `Result` containing the resulting `Tensor` with the maximum values, or a `NeuraRustError`.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0f32, 6.0, 3.0, 4.0, 5.0, 2.0], vec![2, 3]).unwrap();
    /// // [[1., 6., 3.], [4., 5., 2.]]
    ///
    /// // Max over all elements
    /// let max_all = t.max(None, false).unwrap();
    /// assert_eq!(max_all.shape(), vec![]); // Scalar shape
    /// assert_eq!(max_all.item_f32().unwrap(), 6.0);
    ///
    /// // Max along axis 0 (collapse rows)
    /// let max_axis0 = t.max(Some(&[0]), false).unwrap();
    /// assert_eq!(max_axis0.shape(), vec![3]);
    /// assert_eq!(max_axis0.get_f32_data().unwrap(), vec![4.0, 6.0, 3.0]); // [max(1,4), max(6,5), max(3,2)]
    ///
    /// // Max along axis 1 (collapse columns), keeping dimension
    /// let max_axis1_keep = t.max(Some(&[1]), true).unwrap();
    /// assert_eq!(max_axis1_keep.shape(), vec![2, 1]);
    /// assert_eq!(max_axis1_keep.get_f32_data().unwrap(), vec![6.0, 5.0]); // [max(1,6,3), max(4,5,2)]
    /// ```
    pub fn max(&self, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
        // Call the max_op function from the ops module
        crate::ops::reduction::max::max_op(self, axes, keep_dims)
    }

    /// Computes the sum of the tensor elements over specified axes.
    ///
    /// If `axes` is `None`, the sum is computed over all elements, resulting in a scalar tensor.
    /// If `axes` is `Some`, the sum is computed along the specified dimensions.
    ///
    /// The `keep_dims` argument controls whether the reduced dimensions are kept with size 1
    /// or removed entirely.
    ///
    /// This method delegates the computation (including autograd handling) to
    /// [`ops::reduction::sum::sum_op`](../ops/reduction/sum/fn.sum_op.html).
    ///
    /// # Arguments
    /// * `axes`: An optional slice of `usize` specifying the axes along which to compute the sum.
    /// * `keep_dims`: If `true`, the reduced axes are retained in the output shape with size 1.
    ///                If `false`, the reduced axes are removed.
    ///
    /// # Returns
    /// A `Result` containing the resulting `Tensor` with the sum values, or a `NeuraRustError`.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// // [[1., 2., 3.], [4., 5., 6.]]
    ///
    /// // Sum over all elements
    /// let s_all = t.sum(None, false).unwrap();
    /// assert_eq!(s_all.shape(), vec![]); // Scalar shape
    /// assert_eq!(s_all.item_f32().unwrap(), 21.0); // 1+..+6
    ///
    /// // Sum along axis 0 (collapse rows)
    /// let s_axis0 = t.sum(Some(&[0]), false).unwrap();
    /// assert_eq!(s_axis0.shape(), vec![3]);
    /// assert_eq!(s_axis0.get_f32_data().unwrap(), vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]
    ///
    /// // Sum along axis 1 (collapse columns), keeping dimension
    /// let s_axis1_keep = t.sum(Some(&[1]), true).unwrap();
    /// assert_eq!(s_axis1_keep.shape(), vec![2, 1]);
    /// assert_eq!(s_axis1_keep.get_f32_data().unwrap(), vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    /// ```
    pub fn sum(&self, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
        // Call the sum_op function from the ops module
        crate::ops::reduction::sum::sum_op(self, axes, keep_dims)
    }

    // TODO: Add min method here similarly?
}