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


#[cfg(test)]
mod tests {
    use super::*; // Importe les fonctions du module parent (zeros, ones, full, etc.)
    use crate::tensor::Tensor;
    use crate::device::StorageDevice;
    
    // Ajouter l'import manquant pour DType
    use crate::types::DType; 

    #[test]
    fn test_zeros_like() {
        let tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let zeros_t = zeros_like(&tensor).unwrap();
        assert_eq!(zeros_t.shape(), tensor.shape());
        assert_eq!(zeros_t.numel(), tensor.numel());
        assert_eq!(zeros_t.device(), tensor.device());
        assert_eq!(zeros_t.dtype(), DType::F32);
        assert!(zeros_t.get_f32_data().unwrap().iter().all(|&x| x == 0.0));
    }

    // ... reste des tests (test_ones_like, test_arange, etc.) ...
    
    #[test]
    fn test_zeros() {
        let shape = vec![2, 3];
        let t = zeros(&shape).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.get_f32_data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let shape = vec![1, 4];
        let t = ones(&shape).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.get_f32_data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let shape = vec![3, 1, 2];
        let fill_val = 42.5_f32;
        let t = full(&shape, fill_val).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.get_f32_data().unwrap().iter().all(|&x| (x - fill_val).abs() < 1e-6));
    }

    #[test]
    fn test_eye() {
        let n = 3;
        let t = eye(n).unwrap();
        assert_eq!(t.shape(), vec![n, n]);
        assert_eq!(t.numel(), n * n);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F32);
        let expected_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(t.get_f32_data().unwrap(), expected_data);
    }

    #[test]
    fn test_rand() {
        let shape = vec![2, 2];
        let t = rand(&shape).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.get_f32_data().unwrap().iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_randn() {
        let shape = vec![3, 3];
        let t = randn(&shape).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 9);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F32);
        // Basic check: Data exists. More rigorous checks would involve statistical tests.
        assert!(!t.get_f32_data().unwrap().is_empty());
    }

    #[test]
    fn test_zeros_f64() {
        let shape = vec![2, 3];
        let t = zeros_f64(&shape).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F64);
        assert!(t.get_f64_data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_f64() {
        let shape = vec![1, 4];
        let t = ones_f64(&shape).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F64);
        assert!(t.get_f64_data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full_f64() {
        let shape = vec![3, 1, 2];
        let fill_val = -3.14159_f64;
        let t = full_f64(&shape, fill_val).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F64);
        assert!(t.get_f64_data().unwrap().iter().all(|&x| (x - fill_val).abs() < 1e-9));
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![1.1, 2.2, 3.3];
        let shape = vec![3];
        let t = from_vec_f64(data.clone(), shape.clone()).unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 3);
        assert_eq!(t.device(), StorageDevice::CPU);
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.get_f64_data().unwrap(), data);
    }

    #[test]
    fn test_zeros_like_f64() {
        let tensor_f64 = from_vec_f64(vec![10.0, 20.0], vec![2]).unwrap();
        let zeros_t = zeros_like(&tensor_f64).unwrap();
        assert_eq!(zeros_t.shape(), tensor_f64.shape());
        assert_eq!(zeros_t.numel(), tensor_f64.numel());
        assert_eq!(zeros_t.device(), tensor_f64.device()); // Assumes CPU
        assert_eq!(zeros_t.dtype(), DType::F64);
        assert!(zeros_t.get_f64_data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_like_f64() {
        let tensor_f64 = from_vec_f64(vec![-5.0], vec![1]).unwrap();
        let ones_t = ones_like(&tensor_f64).unwrap();
        assert_eq!(ones_t.shape(), tensor_f64.shape());
        assert_eq!(ones_t.numel(), tensor_f64.numel());
        assert_eq!(ones_t.device(), tensor_f64.device()); // Assumes CPU
        assert_eq!(ones_t.dtype(), DType::F64);
        assert!(ones_t.get_f64_data().unwrap().iter().all(|&x| x == 1.0));
    }
}