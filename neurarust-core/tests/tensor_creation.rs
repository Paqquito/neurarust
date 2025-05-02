use neurarust_core::{
    device::StorageDevice,
    error::NeuraRustError,
    tensor::Tensor,
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
    assert_relative_eq!(t.get(&[0, 0]).unwrap(), 1.0);
    assert_relative_eq!(t.get(&[1, 1]).unwrap(), 4.0);
}

#[test]
fn test_tensor_creation_error() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let shape = vec![2, 2];
    // Use Tensor::new directly here as create_test_tensor would panic
    let result = Tensor::<f32>::new(data, shape);
    assert!(result.is_err());
    match result.err().unwrap() {
        NeuraRustError::TensorCreationError {
            data_len,
            shape: err_shape,
        } => {
            assert_eq!(data_len, 3);
            assert_eq!(err_shape, vec![2, 2]);
        }
        e => panic!("Expected TensorCreationError, got {:?}", e), // Improved panic message
    }
}


#[test]
fn test_zeros_creation() {
    let shape = vec![2, 3];
    // Use Tensor::zeros and add type annotation for variable t
    let t: Tensor<f64> = Tensor::zeros(shape.clone()).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.device(), StorageDevice::CPU); // Check device
                                                // Verify data is zero (type known from t)
    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(t.get(&[i, j]).unwrap(), 0.0f64);
        }
    }
}

#[test]
fn test_ones_creation() {
    let shape = vec![1, 4];
    // Use Tensor::ones and add type annotation for variable t
    let t: Tensor<i32> = Tensor::ones(shape.clone()).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 4);
    assert_eq!(t.device(), StorageDevice::CPU);
    // Verify data is one (type known from t)
    for j in 0..4 {
        assert_eq!(t.get(&[0, j]).unwrap(), 1i32);
    }
}

#[test]
fn test_full_creation() {
    let shape = vec![3, 1, 2];
    let fill_val = 42.5_f32;
    // Use Tensor::full (type inferred from fill_val)
    let t = Tensor::full(shape.clone(), fill_val).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.device(), StorageDevice::CPU);
    // Verify data is fill_val
    for i in 0..3 {
        for j in 0..1 {
            for k in 0..2 {
                assert_relative_eq!(t.get(&[i, j, k]).unwrap(), fill_val);
            }
        }
    }
} 