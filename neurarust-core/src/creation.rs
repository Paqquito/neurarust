use crate::tensor::Tensor;
use num_traits::{One, Zero};

// Implementation block for Tensor creation methods (zeros, ones, etc.)
impl<T> Tensor<T>
where
    T: Clone, // Required for filling the Vec
{
    /// Creates a new `Tensor` filled with zeros with the specified shape.
    ///
    /// Requires the element type `T` to implement the [`Zero`] and [`Clone`] traits.
    ///
    /// # Examples
    /// ```
    /// // use neurarust_core::Tensor;
    /// // let zeros = Tensor::<f32>::zeros(vec![2, 2]);
    /// // assert_eq!(zeros.data(), &[0.0, 0.0, 0.0, 0.0]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel];
        Tensor::new(data, shape)
    }

    /// Creates a new `Tensor` filled with ones with the specified shape.
    ///
    /// Requires the element type `T` to implement the [`One`] and [`Clone`] traits.
    /// ///
    /// # Examples
    /// ```
    /// // use neurarust_core::Tensor;
    /// // let ones = Tensor::<i32>::ones(vec![1, 3]);
    /// // assert_eq!(ones.data(), &[1, 1, 1]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One,
    {
        let numel = shape.iter().product();
        let data = vec![T::one(); numel];
        Tensor::new(data, shape)
    }

    // TODO: Implement rand (will likely require rand crate and maybe distribution traits)
}

#[cfg(test)]
mod tests {
    use super::*; // Imports Tensor from crate::tensor via crate root
    use crate::Tensor;
    use num_traits::{One, Zero};

    #[test]
    fn test_zeros() {
        let shape = vec![2, 3];
        let t_zeros_f32 = Tensor::<f32>::zeros(shape.clone());
        let expected_data_f32 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(t_zeros_f32.shape(), shape);
        assert_eq!(t_zeros_f32.data(), expected_data_f32);
        assert!(!t_zeros_f32.requires_grad());

        let t_zeros_i32 = Tensor::<i32>::zeros(shape.clone());
        let expected_data_i32 = vec![0, 0, 0, 0, 0, 0];
        assert_eq!(t_zeros_i32.shape(), shape);
        assert_eq!(t_zeros_i32.data(), expected_data_i32);
        assert!(!t_zeros_i32.requires_grad());
    }

     #[test]
    fn test_ones() {
        let shape = vec![1, 4];
        let t_ones_f64 = Tensor::<f64>::ones(shape.clone());
        let expected_data_f64 = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(t_ones_f64.shape(), shape);
        assert_eq!(t_ones_f64.data(), expected_data_f64);
        assert!(!t_ones_f64.requires_grad());

        let t_ones_u8 = Tensor::<u8>::ones(shape.clone());
        let expected_data_u8 = vec![1, 1, 1, 1];
        assert_eq!(t_ones_u8.shape(), shape);
        assert_eq!(t_ones_u8.data(), expected_data_u8);
        assert!(!t_ones_u8.requires_grad());
    }
} 