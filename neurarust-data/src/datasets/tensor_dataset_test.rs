// neurarust-data/src/datasets/tensor_dataset_test.rs

use super::*;
use neurarust_core::tensor::create::from_vec_f32;
use neurarust_core::{tensor::Tensor, NeuraRustError};

// Helper to create a basic tensor for testing.
fn create_test_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    from_vec_f32(data, shape.to_vec()).unwrap()
}

fn create_test_tensor_requires_grad(data: Vec<f32>, shape: &[usize]) -> Tensor {
    let t = from_vec_f32(data, shape.to_vec()).unwrap();
    t.set_requires_grad(true).unwrap();
    t
}

#[test]
fn test_tensor_dataset_new_empty() {
    let dataset = TensorDataset::new(vec![]).unwrap();
    assert_eq!(dataset.len(), 0);
    assert!(dataset.is_empty());
}

#[test]
fn test_tensor_dataset_new_single_tensor() {
    let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let dataset = TensorDataset::new(vec![t1]).unwrap();
    assert_eq!(dataset.len(), 2);
}

#[test]
fn test_tensor_dataset_new_multiple_tensors_valid() {
    let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let t2 = create_test_tensor(vec![0.0, 1.0, 0.0], &[3, 1]);
    let dataset = TensorDataset::new(vec![t1, t2]).unwrap();
    assert_eq!(dataset.len(), 3);
}

#[test]
fn test_tensor_dataset_new_rank_mismatch_scalar() {
    let t1 = create_test_tensor(vec![1.0], &[]); // scalar
    let err = TensorDataset::new(vec![t1]).err().unwrap();
    match err {
        NeuraRustError::RankMismatch { expected, actual } => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 0);
        }
        _ => panic!("Expected RankMismatch error"),
    }
}

#[test]
fn test_tensor_dataset_new_rank_mismatch_in_list() {
    let t1 = create_test_tensor(vec![1.0, 2.0], &[2,1]);
    let t2 = create_test_tensor(vec![3.0], &[]); // scalar
    let err = TensorDataset::new(vec![t1, t2]).err().unwrap();
    match err {
        NeuraRustError::RankMismatch { expected, actual } => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 0);
        }
        _ => panic!("Expected RankMismatch error"),
    }
}

#[test]
fn test_tensor_dataset_new_shape_mismatch() {
    let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = create_test_tensor(vec![0.0, 1.0, 0.0], &[3, 1]); // Different first dimension
    let err = TensorDataset::new(vec![t1, t2]).err().unwrap();
    match err {
        NeuraRustError::ShapeMismatch { .. } => assert!(true),
        _ => panic!("Expected ShapeMismatch error"),
    }
}

#[test]
fn test_tensor_dataset_get_valid_index_single_tensor() {
    let t1 = create_test_tensor_requires_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let dataset = TensorDataset::new(vec![t1]).unwrap();

    let item0 = dataset.get(0).unwrap();
    assert_eq!(item0.len(), 1);
    // Slice in TensorDataset will produce a tensor of shape [1, 2] for this case,
    // as it slices the first dimension [0,1) and keeps remaining dims.
    assert_eq!(item0[0].shape(), &[1, 2]); 
    assert_eq!(item0[0].get_f32_data().unwrap(), vec![1.0, 2.0]);
    assert!(item0[0].requires_grad(), "Grad requirement should propagate");

    let item2 = dataset.get(2).unwrap();
    assert_eq!(item2.len(), 1);
    assert_eq!(item2[0].shape(), &[1, 2]);
    assert_eq!(item2[0].get_f32_data().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_tensor_dataset_get_valid_index_multiple_tensors() {
    let t1_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t1_shape = &[3, 2];
    let t1 = create_test_tensor(t1_data.clone(), t1_shape);

    let t2_data = vec![10.0, 20.0, 30.0];
    let t2_shape = &[3, 1];
    let t2 = create_test_tensor(t2_data.clone(), t2_shape);
    
    let dataset = TensorDataset::new(vec![t1, t2]).unwrap();

    let item1 = dataset.get(1).unwrap();
    assert_eq!(item1.len(), 2);
    
    assert_eq!(item1[0].shape(), &[1, 2]);
    assert_eq!(item1[0].get_f32_data().unwrap(), vec![3.0, 4.0]);

    assert_eq!(item1[1].shape(), &[1, 1]);
    assert_eq!(item1[1].get_f32_data().unwrap(), vec![20.0]);
}


#[test]
fn test_tensor_dataset_get_valid_index_multiple_tensors_grad_propagation() {
    let t1 = create_test_tensor_requires_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = create_test_tensor(vec![0.1, 0.2], &[2, 1]); // t2 no grad
    let t3 = create_test_tensor_requires_grad(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let dataset = TensorDataset::new(vec![t1, t2, t3]).unwrap();
    let item0 = dataset.get(0).unwrap();

    assert_eq!(item0.len(), 3);
    assert!(item0[0].requires_grad(), "t1 slice should require grad");
    assert!(!item0[1].requires_grad(), "t2 slice should not require grad");
    assert!(item0[2].requires_grad(), "t3 slice should require grad");

    assert_eq!(item0[0].get_f32_data().unwrap(), vec![1.0, 2.0]);
    assert_eq!(item0[1].get_f32_data().unwrap(), vec![0.1]);
    assert_eq!(item0[2].get_f32_data().unwrap(), vec![5.0, 6.0]);
}


#[test]
fn test_tensor_dataset_get_invalid_index() {
    let t1 = create_test_tensor(vec![1.0, 2.0], &[1, 2]);
    let dataset = TensorDataset::new(vec![t1]).unwrap();
    let err = dataset.get(1).err().unwrap();
    match err {
        NeuraRustError::IndexOutOfBounds { .. } => assert!(true),
        _ => panic!("Expected IndexOutOfBounds error"),
    }
}

#[test]
fn test_tensor_dataset_get_from_empty_dataset() {
    let dataset = TensorDataset::new(vec![]).unwrap();
    let err = dataset.get(0).err().unwrap(); // index 0 is out of bounds for empty
    match err {
        NeuraRustError::IndexOutOfBounds { index, shape } => {
            assert_eq!(index, vec![0]);
            assert_eq!(shape, vec![0]);
        }
        _ => panic!("Expected IndexOutOfBounds error for empty dataset get"),
    }
}


#[test]
fn test_tensor_dataset_slice_shape_consistency() {
    // Test with a tensor that would result in a 0-dim slice if not careful
    let t1 = create_test_tensor(vec![10.0, 20.0, 30.0], &[3]); // Rank 1 tensor
    let dataset = TensorDataset::new(vec![t1]).unwrap();
    
    let item0_vec = dataset.get(0).unwrap();
    assert_eq!(item0_vec.len(), 1);
    let item0_tensor_slice = &item0_vec[0];
    // Slicing the first (and only) dimension [0,1) from a [3] tensor results in a [1] tensor.
    assert_eq!(item0_tensor_slice.shape(), &[1], "Slice of 1D tensor should be [1]");
    assert_eq!(item0_tensor_slice.get_f32_data().unwrap(), vec![10.0]);
}

#[test]
fn test_tensor_dataset_slice_for_1d_tensor() {
    let features = create_test_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let labels = create_test_tensor(vec![0.0, 1.0, 0.0], &[3]);
    let dataset = TensorDataset::new(vec![features, labels]).unwrap();

    let item1 = dataset.get(1).unwrap();
    assert_eq!(item1.len(), 2);
    assert_eq!(item1[0].shape(), &[1]);
    assert_eq!(item1[0].get_f32_data().unwrap(), vec![2.0]);
    assert_eq!(item1[1].shape(), &[1]);
    assert_eq!(item1[1].get_f32_data().unwrap(), vec![1.0]);
} 