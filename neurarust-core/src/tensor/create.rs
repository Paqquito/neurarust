// src/tensor/create.rs

use super::Tensor; // Access Tensor struct from parent
use crate::tensor_data::TensorData;
use crate::error::NeuraRustError;
use crate::device::StorageDevice;
use num_traits::{Zero, One};
use std::sync::{Arc, RwLock};
use std::fmt::Debug;
use std::marker::Copy;

impl<T: 'static + Debug + Copy> Tensor<T> {
    /// Creates a new tensor from a vector of data and a shape.
    /// The tensor will be allocated on the CPU by default.
    pub fn new(data_vec: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Creates a tensor of zeros with the specified shape on the CPU.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: Zero {
        let numel = shape.iter().product::<usize>();
        let data_vec = vec![T::zero(); numel];
        Self::new(data_vec, shape)
    }

    /// Creates a tensor of zeros with the same shape and device as another tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Result<Self, NeuraRustError> where T: Zero {
        let other_guard = other.read_data();
        let shape = other_guard.shape.clone();
        let device = other_guard.device;
        drop(other_guard);

        match device {
            StorageDevice::CPU => {
                let numel = shape.iter().product::<usize>();
                let data_vec = vec![T::zero(); numel];
                Self::new(data_vec, shape)
            }
            StorageDevice::GPU => {
                Err(NeuraRustError::UnsupportedOperation("GPU zero tensor creation not yet implemented".to_string()))
            }
        }
    }

    /// Creates a tensor of ones with the specified shape on the CPU.
    pub fn ones(shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: One {
        let numel = shape.iter().product::<usize>();
        let data_vec = vec![T::one(); numel];
        Self::new(data_vec, shape)
    }

    /// Creates a tensor filled with a specific value on the CPU.
    pub fn full(shape: Vec<usize>, fill_value: T) -> Result<Self, NeuraRustError> {
        let numel = shape.iter().product::<usize>();
        let data_vec = vec![fill_value; numel];
        Self::new(data_vec, shape)
    }

    /// Creates a scalar tensor (0-dimensional) on the CPU.
    pub fn scalar(value: T) -> Self {
        Self::new(vec![value], vec![]).expect("Scalar creation failed")
    }
} 