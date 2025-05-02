// src/tensor/create.rs

use super::Tensor; // Access Tensor struct from parent
// Removed unused imports
// use crate::buffer::Buffer;
// use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use num_traits::{Float, NumCast, One, Zero};
use std::fmt::Debug;
use std::marker::Copy;
use std::sync::{Arc, RwLock};

// Imports for new creation functions
use rand::distributions::Standard; // Import Standard distribution
use rand::Rng;
use rand_distr::Distribution; // Keep Distribution
use rand_distr::StandardNormal; // StandardNormal distribution struct

impl<T> Tensor<T>
where
    T: 'static + Debug + Copy,
{
    /// Creates a new tensor from a flat vector and a shape.
    /// Assumes CPU device.
    pub fn new(data_vec: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where T: Default // Needed by TensorData::new
    {
        // Explicitly use the 2-argument call for TensorData::new
        let tensor_data = TensorData::<T>::new(data_vec, shape)?;
        Ok(Tensor { data: Arc::new(RwLock::new(tensor_data)) })
    }

    /// Creates a tensor of zeros with the specified shape on the CPU.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where
        T: Zero + Default,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![T::zero(); numel];
        Self::new(data, shape)
    }

    /// Creates a tensor of zeros with the same shape and device as another tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Result<Self, NeuraRustError>
    where
        T: Zero + Default,
    {
        let shape = other.shape().clone();
        Self::zeros(shape)
    }

    /// Creates a tensor of ones with the specified shape on the CPU.
    pub fn ones(shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where
        T: One + Default,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![T::one(); numel];
        Self::new(data, shape)
    }

    /// Creates a tensor filled with a specific value on the CPU.
    pub fn full(shape: Vec<usize>, fill_value: T) -> Result<Self, NeuraRustError>
    where
        T: Default,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![fill_value; numel];
        Self::new(data, shape)
    }

    /// Creates a scalar tensor (0-dimensional).
    pub fn scalar(value: T) -> Self
    where T: Default // Add Default bound here
    {
        // Call new which requires T: Default
        Self::new(vec![value], vec![]).expect("Scalar creation failed")
    }

    /// Creates a tensor of ones with the same shape as another tensor.
    pub fn ones_like(other: &Tensor<T>) -> Result<Self, NeuraRustError>
    where
        T: One + Default,
    {
        let shape = other.shape().clone();
        Self::ones(shape)
    }

    // --- NEW Creation Methods --- (Step 1.10)

    /// Creates a 1-D tensor of values from `start` to `end` (exclusive) with `step`.
    pub fn arange(start: T, end: T, step: T) -> Result<Self, NeuraRustError>
    where
        T: PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + Zero + Copy + NumCast + Default,
    {
        if step == T::zero() {
            return Err(NeuraRustError::UnsupportedOperation(
                "arange step cannot be zero".to_string(),
            ));
        }
        let mut data = Vec::new();
        let mut current = start;

        if step > T::zero() {
            while current < end {
                data.push(current);
                current = current + step;
            }
        } else { // step < 0
            while current > end {
                data.push(current);
                current = current + step; // Add negative step
            }
        }

        let shape = vec![data.len()];
        Self::new(data, shape)
    }

    /// Creates a 1-D tensor of `steps` values evenly spaced from `start` to `end` (inclusive).
    pub fn linspace(start: T, end: T, steps: usize) -> Result<Self, NeuraRustError>
    where
        T: Float + Default,
    {
        if steps == 0 {
            return Self::new(vec![], vec![0]);
        }
        if steps == 1 {
            return Self::new(vec![start], vec![1]);
        }

        let mut data = Vec::with_capacity(steps);
        let step_val = (end - start) / T::from(steps - 1).ok_or_else(|| NeuraRustError::InternalError("Failed to cast usize to T in linspace".to_string()))?;

        for i in 0..steps {
            let val = start + T::from(i).ok_or_else(|| NeuraRustError::InternalError("Failed to cast usize to T in linspace".to_string()))? * step_val;
            data.push(val);
        }
        // Ensure the last element is exactly `end` to avoid potential float inaccuracies
        if steps > 0 {
             *data.last_mut().unwrap() = end;
        }

        Self::new(data, vec![steps])
    }

    /// Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.
    pub fn eye(n: usize, m: Option<usize>) -> Result<Self, NeuraRustError>
    where
        T: Zero + One + Default,
    {
        let m_val = m.unwrap_or(n);
        let shape = vec![n, m_val];
        let numel = n * m_val;
        let mut data = vec![T::zero(); numel];

        let min_dim = n.min(m_val);
        for i in 0..min_dim {
            let index = i * m_val + i; // Row i, Col i
            if index < numel {
                data[index] = T::one();
            }
        }

        Self::new(data, shape)
    }

    /// Creates a tensor with the given shape, filled with random values from a
    /// uniform distribution between 0 (inclusive) and 1 (exclusive).
    pub fn rand(shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where
        T: rand::distributions::uniform::SampleUniform + Default,
        Standard: Distribution<T>,
    {
        let numel = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..numel).map(|_| rng.gen()).collect();
        Self::new(data, shape)
    }

    /// Creates a tensor with the given shape, filled with random values from a
    /// standard normal distribution (mean 0, variance 1).
    pub fn randn(shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where
        StandardNormal: Distribution<T>,
        T: Default + Copy,
    {
        let numel = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let dist = StandardNormal;
        let data: Vec<T> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
        Self::new(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // Correct import for the macro

    #[test]
    fn test_zeros_like() {
        let t = Tensor::<f64>::zeros_like(&Tensor::<f64>::zeros(vec![3, 4]).unwrap()).unwrap();
        assert_eq!(t.shape(), vec![3, 4]);
        let data = t.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(data.len(), 12);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_like() {
        let t = Tensor::<f64>::ones_like(&Tensor::<f64>::ones(vec![3, 4]).unwrap()).unwrap();
        assert_eq!(t.shape(), vec![3, 4]);
        let data = t.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(data.len(), 12);
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_arange() {
        let t = Tensor::<f64>::arange(0.0, 10.0, 1.0).unwrap();
        assert_eq!(t.shape(), vec![10]);
        let data = t.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(data.len(), 10);
        assert!(data.iter().zip(0..10).all(|(a, b)| *a == b as f64));
    }

    #[test]
    fn test_linspace() {
        let t1 = Tensor::<f64>::linspace(0.0, 10.0, 5).unwrap();
        assert_eq!(t1.shape(), vec![5]);
        let data1 = t1.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(data1[0], 0.0);
        assert_relative_eq!(data1[1], 2.5);
        assert_relative_eq!(data1[2], 5.0);
        assert_relative_eq!(data1[3], 7.5);
        assert_relative_eq!(data1[4], 10.0);

        let t_check = Tensor::<f64>::linspace(1.0, 3.0, 5).unwrap();
        let d_check = t_check.read_data().data.cpu_data().unwrap().clone();
        assert!(d_check.iter().zip([1.0, 1.5, 2.0, 2.5, 3.0].iter()).all(|(a, b)| *a == *b));
    }

    #[test]
    fn test_eye() {
        let t_sq = Tensor::<i32>::eye(3, None).unwrap();
        assert_eq!(t_sq.shape(), vec![3, 3]);
        assert!(t_sq.read_data().data.cpu_data().unwrap().iter().zip([1, 0, 0, 0, 1, 0, 0, 0, 1].iter()).all(|(a, b)| *a == *b));

        let t_rect_n_gt_m = Tensor::<f32>::eye(4, Some(2)).unwrap();
        assert_eq!(t_rect_n_gt_m.shape(), vec![4, 2]);
        assert!(t_rect_n_gt_m.read_data().data.cpu_data().unwrap().iter().zip([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0].iter()).all(|(a,b)| *a == *b));

        let t_rect_m_gt_n = Tensor::<f64>::eye(2, Some(4)).unwrap();
        assert_eq!(t_rect_m_gt_n.shape(), vec![2, 4]);
        assert!(t_rect_m_gt_n.read_data().data.cpu_data().unwrap().iter().zip([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].iter()).all(|(a,b)| *a == *b));

        let t_zero = Tensor::<i32>::eye(0, None).unwrap();
        assert_eq!(t_zero.shape(), vec![0, 0]);
        assert!(t_zero.read_data().data.cpu_data().unwrap().is_empty());
    }

    #[test]
    fn test_rand() {
        let t = Tensor::<f64>::rand(vec![3, 4]).unwrap();
        assert_eq!(t.shape(), vec![3, 4]);
        let data = t.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(data.len(), 12);
    }

    #[test]
    fn test_randn() {
        let shape = vec![1000]; // Use enough samples for rough check
        let numel = shape.iter().product::<usize>();
        let t = Tensor::<f64>::randn(shape.clone()).unwrap();
        assert_eq!(t.shape(), shape);
        let data = t.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(data.len(), numel);

        // Rough check for mean close to 0 and variance close to 1
        let mean = data.iter().sum::<f64>() / (numel as f64);
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (numel as f64);

        println!("Randn mean: {}, variance: {}", mean, variance);
        assert_relative_eq!(mean, 0.0, epsilon = 0.1); // Loose check for mean
        assert_relative_eq!(variance, 1.0, epsilon = 0.2); // Loose check for variance
    }

    #[test]
    fn test_ones_like_basic() {
        let t1 = Tensor::<f32>::zeros(vec![4, 1]).unwrap();
        let t2 = Tensor::ones_like(&t1).unwrap();
        assert_eq!(t2.shape(), vec![4, 1]);
        assert!(t2.read_data().data.cpu_data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_zeros_like_basic() {
        let t1 = Tensor::<f32>::ones(vec![2, 3]).unwrap();
        let t2 = Tensor::zeros_like(&t1).unwrap();
        assert_eq!(t2.shape(), vec![2, 3]);
        assert!(t2.read_data().data.cpu_data().unwrap().iter().all(|&x| x == 0.0));
    }
}