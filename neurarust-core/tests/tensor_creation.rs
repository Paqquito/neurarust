use neurarust_core::{
    device::StorageDevice,
    error::NeuraRustError,
    tensor::Tensor,
    // Utiliser les fonctions ré-exportées directement depuis tensor
    tensor::{zeros, ones, full}, 
};
use approx::assert_relative_eq;

// Include the common helper module
mod common;
use common::create_test_tensor;

#[test]
fn test_tensor_creation() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let t = create_test_tensor(data.clone(), shape.clone());
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 4);
    assert_eq!(t.strides(), vec![2, 1]);
    // Use get_f32_data
    let t_data = t.get_f32_data().unwrap();
    assert_relative_eq!(t_data[0], 1.0); // Index [0,0] -> linear 0*2 + 0*1 = 0
    assert_relative_eq!(t_data[3], 4.0); // Index [1,1] -> linear 1*2 + 1*1 = 3
}

#[test]
fn test_tensor_creation_error() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let shape = vec![2, 2];
    // Use non-generic Tensor::new
    let result = Tensor::new(data, shape);
    assert!(result.is_err());
    match result.err().unwrap() {
        NeuraRustError::TensorCreationError {
            data_len,
            shape: err_shape,
        } => {
            assert_eq!(data_len, 3);
            assert_eq!(err_shape, vec![2, 2]);
        }
        e => panic!("Expected TensorCreationError, got {:?}", e),
    }
}


#[test]
fn test_zeros_creation() {
    let shape = vec![2, 3];
    // Use zeros function (defaults to f32)
    let t = zeros(&shape).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.device(), StorageDevice::CPU); // Check device
    // Verify data is zero using get_f32_data
    let t_data = t.get_f32_data().unwrap();
    assert!(t_data.iter().all(|&x| x == 0.0), "Tensor not filled with zeros");
}

#[test]
fn test_ones_creation() {
    let shape = vec![1, 4];
    // Use ones function (defaults to f32)
    let t = ones(&shape).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 4);
    assert_eq!(t.device(), StorageDevice::CPU);
    // Verify data is one using get_f32_data
    let t_data = t.get_f32_data().unwrap();
    assert!(t_data.iter().all(|&x| x == 1.0), "Tensor not filled with ones");
}

#[test]
fn test_full_creation() {
    let shape = vec![3, 1, 2];
    let fill_val = 42.5_f32;
    // Use full function
    let t = full(&shape, fill_val).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.device(), StorageDevice::CPU);
    // Verify data is fill_val using get_f32_data
    let t_data = t.get_f32_data().unwrap();
    assert!(t_data.iter().all(|&x| (x - fill_val).abs() < 1e-6), "Tensor not filled with value");
} 