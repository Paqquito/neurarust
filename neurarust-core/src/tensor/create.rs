// src/tensor/create.rs

use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;
 // Import necessary types

/// Creates a new tensor filled with zeros with the specified shape.
/// Currently creates an f32 tensor on the CPU.
pub fn zeros(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f32> = vec![0.0; numel]; // Create f32 data
    Tensor::new(data_vec, shape.to_vec())
}

/// Creates a new F64 tensor filled with zeros with the specified shape on the CPU.
pub fn zeros_f64(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f64> = vec![0.0; numel]; // Create f64 data
    Tensor::new_f64(data_vec, shape.to_vec())
}

/// Creates a new tensor filled with ones with the specified shape.
/// Currently creates an f32 tensor on the CPU.
pub fn ones(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f32> = vec![1.0; numel]; // Create f32 data
    Tensor::new(data_vec, shape.to_vec())
}

/// Creates a new F64 tensor filled with ones with the specified shape on the CPU.
pub fn ones_f64(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f64> = vec![1.0; numel]; // Create f64 data
    Tensor::new_f64(data_vec, shape.to_vec())
}

/// Creates a new tensor filled with a specific value with the specified shape.
/// Currently creates an f32 tensor on the CPU.
pub fn full(shape: &[usize], value: f32) -> Result<Tensor, NeuraRustError> { // value is now f32
    let numel = shape.iter().product();
    let data_vec: Vec<f32> = vec![value; numel]; // Create f32 data
    Tensor::new(data_vec, shape.to_vec())
}

/// Creates a new F64 tensor filled with a specific value with the specified shape on the CPU.
pub fn full_f64(shape: &[usize], value: f64) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f64> = vec![value; numel]; // Create f64 data
    Tensor::new_f64(data_vec, shape.to_vec())
}

/// Creates a new CPU F32 Tensor from a Vec<f32> and shape.
/// (Moved from tensor/mod.rs for consistency)
pub fn from_vec_f32(data_vec: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    Tensor::new(data_vec, shape)
}

/// Creates a new CPU F64 Tensor from a Vec<f64> and shape.
pub fn from_vec_f64(data_vec: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    Tensor::new_f64(data_vec, shape)
}

/// Creates a new tensor filled with zeros, having the same shape and device as the input tensor.
/// Creates a tensor of the same DType as the input.
pub fn zeros_like(tensor: &Tensor) -> Result<Tensor, NeuraRustError> {
    // TODO: Later, use tensor.device() to create on the same device.
    let shape = tensor.shape();
    match tensor.dtype() {
        DType::F32 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f32> = vec![0.0; numel];
            Tensor::new(data_vec, shape)
        }
        DType::F64 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f64> = vec![0.0; numel];
            Tensor::new_f64(data_vec, shape)
        }
        // Add other dtypes later
    }
}

/// Creates a new tensor filled with ones, having the same shape and device as the input tensor.
/// Creates a tensor of the same DType as the input.
pub fn ones_like(tensor: &Tensor) -> Result<Tensor, NeuraRustError> {
    // TODO: Later, use tensor.device() to create on the same device.
    let shape = tensor.shape();
    match tensor.dtype() {
        DType::F32 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f32> = vec![1.0; numel];
            Tensor::new(data_vec, shape)
        }
        DType::F64 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f64> = vec![1.0; numel];
            Tensor::new_f64(data_vec, shape)
        }
        // Add other dtypes later
    }
}

// --- Keep other creation functions like arange, linspace, eye, rand, randn --- 
// They might need adaptation later, especially regarding DType and Device.
// For now, let's assume they primarily work with f32 or can be adapted easily later.

pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor, NeuraRustError> {
    if (end > start && step <= 0.0) || (end < start && step >= 0.0) || step == 0.0 {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("Invalid step {} for arange({}, {})", step, start, end)
        ));
    }
    let numel = ((end - start) / step).ceil() as usize;
    let data_vec: Vec<f32> = (0..numel).map(|i| start + i as f32 * step).collect();
    Tensor::new(data_vec, vec![numel])
}

pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Tensor, NeuraRustError> {
    if steps < 2 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Linspace requires at least 2 steps".to_string()
        ));
    }
    let mut data_vec = Vec::with_capacity(steps);
    let step_size = (end - start) / (steps - 1) as f32;
    for i in 0..steps {
        data_vec.push(start + i as f32 * step_size);
    }
    Tensor::new(data_vec, vec![steps])
}

pub fn eye(n: usize) -> Result<Tensor, NeuraRustError> {
    let mut data_vec = vec![0.0f32; n * n];
    for i in 0..n {
        data_vec[i * n + i] = 1.0;
    }
    Tensor::new(data_vec, vec![n, n])
}

// Note: rand and randn should ideally take a device argument later
// and potentially use device-specific RNGs (e.g., cuRAND).

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn rand(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data_vec: Vec<f32> = (0..numel).map(|_| rng.gen::<f32>()).collect();
    Tensor::new(data_vec, shape.to_vec())
}

pub fn randn(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data_vec: Vec<f32> = (0..numel)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    Tensor::new(data_vec, shape.to_vec())
}

// Link the external tests file
#[cfg(test)]
#[path = "create_test.rs"] mod tests;