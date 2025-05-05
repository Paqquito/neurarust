#[cfg(test)]
// Pas besoin de `mod tests { ... }` ici, les tests sont directement dans le fichier.

use super::*; // Importe les fonctions du module parent (zeros, ones, full, etc.)
use crate::tensor::Tensor; // Importer Tensor directement si create::* ne suffit pas
use crate::device::StorageDevice;
use crate::types::DType; 
// NeuraRustError n'est pas nécessaire si on utilise unwrap() ou ? sur les Results des créations

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
// Missing tests for arange and linspace, should add them
#[test]
fn test_arange() {
    let start = 1.0;
    let end = 5.0;
    let step = 1.5;
    let t = arange(start, end, step).unwrap(); // 1.0, 2.5, 4.0
    assert_eq!(t.shape(), vec![3]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.get_f32_data().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 2.5).abs() < 1e-6);
    assert!((data[2] - 4.0).abs() < 1e-6);
}

#[test]
fn test_linspace() {
    let start = 0.0;
    let end = 10.0;
    let steps = 5;
    let t = linspace(start, end, steps).unwrap(); // 0.0, 2.5, 5.0, 7.5, 10.0
    assert_eq!(t.shape(), vec![steps]);
    assert_eq!(t.dtype(), DType::F32);
    let data = t.get_f32_data().unwrap();
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[1] - 2.5).abs() < 1e-6);
    assert!((data[2] - 5.0).abs() < 1e-6);
    assert!((data[3] - 7.5).abs() < 1e-6);
    assert!((data[4] - 10.0).abs() < 1e-6);
} 