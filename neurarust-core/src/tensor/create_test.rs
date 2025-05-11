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
    let shape = vec![2, 3, 4];
    let t = rand(shape.clone()).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.numel(), 24);
    // Check if values are within the expected range [0, 1)
    let data = t.get_f32_data().unwrap();
    assert!(data.iter().all(|&x| x >= 0.0 && x < 1.0));

    let scalar = rand(vec![]).unwrap();
    assert_eq!(scalar.shape(), vec![]);
    assert_eq!(scalar.numel(), 1);
    assert!(scalar.item_f32().unwrap() >= 0.0);
    assert!(scalar.item_f32().unwrap() < 1.0);
}

#[test]
fn test_randn() {
    let shape = vec![10, 5]; // Slightly larger shape
    let t = randn(shape.clone()).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.numel(), 50);
    // Basic check that values are generated (no strict distribution check)
    let data = t.get_f32_data().unwrap();
    assert_eq!(data.len(), 50);
    // Optional: Check mean/stddev for very large tensors, but can be flaky
    // let mean = data.iter().sum::<f32>() / data.len() as f32;
    // let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    // let stddev = variance.sqrt();
    // assert!((mean).abs() < 0.2); // Loose check for mean near 0
    // assert!((stddev - 1.0).abs() < 0.2); // Loose check for stddev near 1

    let scalar = randn(vec![]).unwrap();
    assert_eq!(scalar.shape(), vec![]);
    assert_eq!(scalar.numel(), 1);
    // No specific range check for randn scalar
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

#[test]
fn test_randint() {
    // Test F32
    let t_f32 = randint(0, 10, vec![2, 3], DType::F32, StorageDevice::CPU).unwrap();
    assert_eq!(t_f32.shape(), &[2, 3]);
    assert_eq!(t_f32.dtype(), DType::F32);
    let data_f32 = t_f32.get_f32_data().unwrap();
    assert_eq!(data_f32.len(), 6);
    for &val in data_f32.iter() {
        assert!(val >= 0.0 && val < 10.0, "Value out of range: {}", val);
        assert_eq!(val, val.trunc(), "Value not an integer: {}", val);
    }

    // Test F64
    let t_f64 = randint(-5, 5, vec![4], DType::F64, StorageDevice::CPU).unwrap();
    assert_eq!(t_f64.shape(), &[4]);
    assert_eq!(t_f64.dtype(), DType::F64);
    let data_f64 = t_f64.get_f64_data().unwrap();
    assert_eq!(data_f64.len(), 4);
    for &val in data_f64.iter() {
        assert!(val >= -5.0 && val < 5.0, "Value out of range: {}", val);
        assert_eq!(val, val.trunc(), "Value not an integer: {}", val);
    }

    // Test scalar shape
    let t_scalar = randint(100, 101, vec![], DType::F32, StorageDevice::CPU).unwrap();
    assert_eq!(t_scalar.shape(), &[] as &[usize]); // empty slice for scalar
    assert_eq!(t_scalar.item_f32().unwrap(), 100.0);


    // Test error for low >= high
    let err_range = randint(10, 0, vec![1], DType::F32, StorageDevice::CPU);
    assert!(matches!(err_range, Err(NeuraRustError::ArithmeticError(_))));

    let err_equal_range = randint(5, 5, vec![1], DType::F32, StorageDevice::CPU);
     assert!(matches!(err_equal_range, Err(NeuraRustError::ArithmeticError(_))));

    // Test error for unsupported DType (hypothetical, as we only have F32/F64 for now)
    // If DType::I32 were a variant but not supported by randint, this would be:
    // let err_dtype = randint(0, 1, vec![1], DType::I32, StorageDevice::CPU);
    // assert!(matches!(err_dtype, Err(NeuraRustError::UnsupportedOperation(_))));
    // For now, this test is more conceptual for when other DTypes are added to the enum
    // but not yet supported by this specific function.
}

#[test]
fn test_bernoulli_scalar() {
    // Test F32
    let p_f32 = 0.75;
    let t_f32 = bernoulli_scalar(p_f32, vec![1000], DType::F32, StorageDevice::CPU).unwrap();
    assert_eq!(t_f32.shape(), &[1000]);
    assert_eq!(t_f32.dtype(), DType::F32);
    let data_f32 = t_f32.get_f32_data().unwrap();
    let mut ones_f32 = 0;
    for &val in data_f32.iter() {
        assert!(val == 0.0 || val == 1.0, "Value not 0.0 or 1.0: {}", val);
        if val == 1.0 {
            ones_f32 += 1;
        }
    }
    // Check if the proportion of ones is roughly p_f32 (very loose check for randomness)
    let proportion_f32 = ones_f32 as f64 / 1000.0;
    assert!((proportion_f32 - p_f32).abs() < 0.1, "Proportion {} out of expected range for p={}", proportion_f32, p_f32);

    // Test F64
    let p_f64 = 0.25;
    let t_f64 = bernoulli_scalar(p_f64, vec![500], DType::F64, StorageDevice::CPU).unwrap();
    assert_eq!(t_f64.shape(), &[500]);
    assert_eq!(t_f64.dtype(), DType::F64);
    let data_f64 = t_f64.get_f64_data().unwrap();
    let mut ones_f64 = 0;
    for &val in data_f64.iter() {
        assert!(val == 0.0 || val == 1.0, "Value not 0.0 or 1.0: {}", val);
        if val == 1.0 {
            ones_f64 += 1;
        }
    }
    let proportion_f64 = ones_f64 as f64 / 500.0;
    assert!((proportion_f64 - p_f64).abs() < 0.1, "Proportion {} out of expected range for p={}", proportion_f64, p_f64);


    // Test scalar shape
    let t_scalar_1 = bernoulli_scalar(1.0, vec![], DType::F32, StorageDevice::CPU).unwrap();
    assert_eq!(t_scalar_1.item_f32().unwrap(), 1.0);
    let t_scalar_0 = bernoulli_scalar(0.0, vec![], DType::F64, StorageDevice::CPU).unwrap();
    assert_eq!(t_scalar_0.item_f64().unwrap(), 0.0);

    // Test error for p out of range
    let err_p_high = bernoulli_scalar(1.1, vec![1], DType::F32, StorageDevice::CPU);
    assert!(matches!(err_p_high, Err(NeuraRustError::ArithmeticError(_))));
    let err_p_low = bernoulli_scalar(-0.1, vec![1], DType::F32, StorageDevice::CPU);
    assert!(matches!(err_p_low, Err(NeuraRustError::ArithmeticError(_))));
} 