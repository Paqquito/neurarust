// neurarust-core/src/tensor/tests.rs
// This file contains the unit tests previously located in tensor/mod.rs

use super::*; // Import everything from the parent module (tensor/mod.rs)
use crate::error::NeuraRustError; // Import error enum
use approx::assert_relative_eq;
use num_traits::{Zero, One};
use std::ops::AddAssign;
use std::fmt::Debug;
use std::iter::Sum;
use std::cmp::PartialEq;
use std::default::Default;
use std::sync::Arc;
use crate::device::StorageDevice; // Added StorageDevice import for tests

// Helper function to create a basic tensor for testing
// It requires many trait bounds, ensure they are available or adjust tests.
fn create_test_tensor<T>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> Tensor<T>
where T: Clone + Debug + PartialEq + Zero + One + Copy + AddAssign + Sum + Default + Send + Sync + 'static
{
    Tensor::new(data, shape).expect("Test tensor creation failed")
}

// Tests remain largely the same, accessing methods like shape(), get()
// which now handle the locking internally.
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
        NeuraRustError::TensorCreationError { data_len, shape: err_shape } => {
            assert_eq!(data_len, 3);
            assert_eq!(err_shape, vec![2, 2]);
        }
        e => panic!("Expected TensorCreationError, got {:?}", e), // Improved panic message
    }
}

#[test]
fn test_tensor_equality() {
    let data1 = vec![1.0_f32, 2.0];
    let shape1 = vec![2];
    let t1 = create_test_tensor(data1.clone(), shape1.clone());
    let t2 = create_test_tensor(data1.clone(), shape1.clone());
    let t3 = t1.clone(); // Clones Arc<RwLock>, points to same allocation
    let t4 = create_test_tensor(vec![3.0, 4.0], shape1.clone());
    let t5 = create_test_tensor(data1.clone(), vec![1, 2]);

    assert_eq!(t1, t1); // Equal to self
    assert_eq!(t1, t3); // Equal to Arc clone

    // With the new PartialEq comparing metadata and buffer Arc, identical creations ARE equal now.
    // The previous assert_ne!(t1, t2) was testing pointer equality, which is not the semantic meaning anymore.
    // If we want to test content equality when buffers are different, the PartialEq needs enhancement.
    // For now, test that identical creations *are* equal based on current PartialEq.
    // We also test that they have different Arc pointers (different instances).
    assert_ne!(Arc::as_ptr(&t1.data), Arc::as_ptr(&t2.data), "t1 and t2 should have different Arc<RwLock> pointers");
    assert_eq!(t1, t2, "t1 and t2 should be equal by content and metadata (current PartialEq)");

    assert_ne!(t1, t4, "t1 and t4 should have different data content"); // Buffers different, content different
    assert_ne!(t1, t5, "t1 and t5 should have different shapes"); // Different shape
}

#[test]
fn test_get_element() {
    let t = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    assert_eq!(t.get(&[0, 0]).unwrap(), 1);
    assert_eq!(t.get(&[0, 2]).unwrap(), 3);
    assert_eq!(t.get(&[1, 0]).unwrap(), 4);
    assert_eq!(t.get(&[1, 2]).unwrap(), 6);
}

#[test]
fn test_get_element_out_of_bounds() {
    let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    assert!(t.get(&[2, 0]).is_err());
    assert!(t.get(&[0, 2]).is_err());
    match t.get(&[0, 2]).err().unwrap() {
        NeuraRustError::IndexOutOfBounds { index, shape } => {
            assert_eq!(index, vec![0, 2]);
            assert_eq!(shape, vec![2, 2]);
        }
        e => panic!("Expected IndexOutOfBounds, got {:?}", e),
    }
}

#[test]
fn test_get_element_wrong_ndim() {
    let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    assert!(t.get(&[0]).is_err());
    assert!(t.get(&[0, 0, 0]).is_err());
    match t.get(&[0]).err().unwrap() {
        NeuraRustError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        e => panic!("Expected DimensionMismatch, got {:?}", e),
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
    for i in 0..2 { for j in 0..3 { assert_relative_eq!(t.get(&[i, j]).unwrap(), 0.0f64); } }
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
    for j in 0..4 { assert_eq!(t.get(&[0, j]).unwrap(), 1i32); }
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
    for i in 0..3 { for j in 0..1 { for k in 0..2 { assert_relative_eq!(t.get(&[i, j, k]).unwrap(), fill_val); } } }
}

#[test]
fn test_simple_slice() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    // Use SliceArg type for clarity
    let ranges: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (0, 3), (0, 4)];
    let view = tensor.slice(&ranges).expect("Simple slice failed");

    assert_eq!(view.shape(), vec![1, 3, 4], "View shape mismatch");
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0, "Value mismatch at [0,0,0]");
    assert_eq!(view.get(&[0, 2, 3]).unwrap(), 11.0, "Value mismatch at [0,2,3]");
}

#[test]
fn test_slice_shares_data() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let ranges: Vec<crate::ops::view_ops::SliceArg> = vec![(1, 2), (1, 3), (0, 2)];
    let view = tensor.slice(&ranges).expect("Slice for data sharing test failed");

    let original_buffer_ptr = Arc::as_ptr(&tensor.borrow_data_buffer());
    let view_buffer_ptr = Arc::as_ptr(&view.borrow_data_buffer());

    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "View does not share the same data buffer Arc");
    assert_eq!(original_buffer_ptr, view_buffer_ptr, "View does not point to the same buffer allocation");
}

#[test]
fn test_slice_metadata() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]); // Shape [2, 3, 4], Strides [12, 4, 1], Offset 0
    let ranges: Vec<crate::ops::view_ops::SliceArg> = vec![(1, 2), (1, 3), (0, 2)]; // Slice: [1, 1:3, 0:2] -> Shape [1, 2, 2]
    let view = tensor.slice(&ranges).expect("Slice for metadata test failed");

    let view_data = view.read_data();

    assert_eq!(view_data.shape, vec![1, 2, 2], "View shape mismatch");
    assert_eq!(view_data.strides, vec![12, 4, 1], "View strides should be inherited");
    assert_eq!(view_data.offset, 16, "View offset calculation incorrect"); // Offset = 1*12 + 1*4 + 0*1 = 16

    // Drop guard before calling view.get() which also locks
    drop(view_data);
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 16.0, "First element value mismatch"); // 1*12 + 1*4 + 0*1 = 16
}

#[test]
fn test_slice_invalid_range() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Range end > dimension size
    let ranges_invalid_end: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (0, 4), (0, 4)]; // Dim 1 size is 3, end is 4
    let result_invalid_end = tensor.slice(&ranges_invalid_end);
    assert!(matches!(result_invalid_end, Err(NeuraRustError::SliceError { .. })), "Expected SliceError for end > dim size");

    // Range start > end
    let ranges_invalid_start: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (3, 2), (0, 4)]; // Dim 1 start > end
    let result_invalid_start = tensor.slice(&ranges_invalid_start);
    assert!(matches!(result_invalid_start, Err(NeuraRustError::SliceError { .. })), "Expected SliceError for start > end");

    // Test start == end (should work)
    let ranges_zero_size: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (2, 2), (0, 4)]; // Dim 1 becomes size 0
    let result_zero_size = tensor.slice(&ranges_zero_size);
    assert!(result_zero_size.is_ok(), "Slice with start == end should be Ok");
    assert_eq!(result_zero_size.unwrap().shape(), vec![1, 0, 4], "Slice start == end shape mismatch");

    // Incorrect number of ranges
    let ranges_wrong_ndim: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (0, 1)]; // Only 2 ranges for 3 dims
    let result_wrong_ndim = tensor.slice(&ranges_wrong_ndim);
    assert!(matches!(result_wrong_ndim, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for wrong number of ranges");
}

#[test]
fn test_transpose_2d() {
    // Tensor: [[0, 1, 2],
    //          [3, 4, 5]] Shape: [2, 3], Strides: [3, 1]
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);

    let view = tensor.transpose(0, 1).expect("Transpose 2D failed");

    // Expected: [[0, 3],
    //            [1, 4],
    //            [2, 5]] Shape: [3, 2], Strides: [1, 3]
    assert_eq!(view.shape(), vec![3, 2], "Transposed shape mismatch");
    assert_eq!(view.strides(), vec![1, 3], "Transposed strides mismatch");

    // Verify data sharing and offset
    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "Transpose view should share data");
    assert_eq!(view.read_data().offset, 0, "Transpose should not change offset");

    // Check values
    assert_eq!(view.get(&[0, 0]).unwrap(), 0.0); // Original [0, 0]
    assert_eq!(view.get(&[0, 1]).unwrap(), 3.0); // Original [1, 0]
    assert_eq!(view.get(&[1, 0]).unwrap(), 1.0); // Original [0, 1]
    assert_eq!(view.get(&[1, 1]).unwrap(), 4.0); // Original [1, 1]
    assert_eq!(view.get(&[2, 0]).unwrap(), 2.0); // Original [0, 2]
    assert_eq!(view.get(&[2, 1]).unwrap(), 5.0); // Original [1, 2]
}

#[test]
fn test_transpose_higher_dim() {
    // Shape: [2, 3, 4], Strides: [12, 4, 1]
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Transpose dims 1 and 2
    let view = tensor.transpose(1, 2).expect("Transpose higher dim failed");

    // Expected Shape: [2, 4, 3], Strides: [12, 1, 4]
    assert_eq!(view.shape(), vec![2, 4, 3], "Transposed shape mismatch");
    assert_eq!(view.strides(), vec![12, 1, 4], "Transposed strides mismatch");

    // Check a few values
    // view[0, 0, 0] -> original[0, 0, 0] = 0
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0);
    // view[0, 1, 2] -> original[0, 2, 1] = offset 0 + 2*4 + 1*1 = 9 -> val 9.0
    assert_eq!(view.get(&[0, 1, 2]).unwrap(), 9.0);
    // view[1, 2, 1] -> original[1, 1, 2] = offset 0 + 1*12 + 1*4 + 2*1 = 18 -> val 18.0
    assert_eq!(view.get(&[1, 2, 1]).unwrap(), 18.0);
}

#[test]
fn test_transpose_invalid_dims() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);

    // dim1 >= rank
    let result_dim1 = tensor.transpose(2, 0);
    assert!(matches!(result_dim1, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for dim1 >= rank");
    if let Err(NeuraRustError::DimensionMismatch { expected, actual }) = result_dim1 {
        assert_eq!(expected, 2, "Expected rank 2");
        assert_eq!(actual, 2, "Actual invalid dim was 2"); // Should report the invalid dimension index
    }

    // dim2 >= rank
    let result_dim2 = tensor.transpose(0, 2);
    assert!(matches!(result_dim2, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for dim2 >= rank");
    if let Err(NeuraRustError::DimensionMismatch { expected, actual }) = result_dim2 {
        assert_eq!(expected, 2, "Expected rank 2");
        assert_eq!(actual, 2, "Actual invalid dim was 2"); // Should report the invalid dimension index
    }

    // dim1 == dim2 (should ideally be a no-op view, but let's check if it errors or not)
    // Current implementation likely doesn't error, just creates an identical view.
    let result_same_dims = tensor.transpose(0, 0);
    assert!(result_same_dims.is_ok(), "Transposing same dimension should be Ok");
    let view_same_dims = result_same_dims.unwrap();
    // Check metadata is identical
    assert_eq!(view_same_dims.shape(), tensor.shape());
    assert_eq!(view_same_dims.strides(), tensor.strides());
    assert_eq!(view_same_dims.read_data().offset, tensor.read_data().offset);
    // Ensure it's a new view struct pointing to the same data
    assert_ne!(Arc::as_ptr(&tensor.data), Arc::as_ptr(&view_same_dims.data), "Transpose same dim should create new view struct");
    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view_same_dims.borrow_data_buffer()), "Transpose same dim view should share data buffer");

}

#[test]
fn test_permute_simple() {
    // Tensor: [[[ 0,  1],
    //            [ 2,  3]],
    //           [[ 4,  5],
    //            [ 6,  7]]] Shape: [2, 2, 2], Strides: [4, 2, 1]
    let data = (0..8).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 2, 2]);

    // Permute to [2, 0, 1]
    let view = tensor.permute(&[2, 0, 1]).expect("Simple permute failed");

    // Expected Shape: [2, 2, 2], Strides: [1, 4, 2]
    assert_eq!(view.shape(), vec![2, 2, 2], "Permuted shape mismatch");
    assert_eq!(view.strides(), vec![1, 4, 2], "Permuted strides mismatch");

    // Check values
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0); // Original [0, 0, 0]
    assert_eq!(view.get(&[1, 0, 0]).unwrap(), 1.0); // Original [0, 0, 1]
    assert_eq!(view.get(&[0, 1, 0]).unwrap(), 4.0); // Original [1, 0, 0]
    assert_eq!(view.get(&[1, 1, 1]).unwrap(), 7.0); // Original [1, 1, 1]
}

#[test]
fn test_permute_higher_dim() {
    // Shape: [2, 3, 4], Strides: [12, 4, 1]
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Permute to [1, 2, 0]
    let view = tensor.permute(&[1, 2, 0]).expect("Higher dim permute failed");

    // Expected Shape: [3, 4, 2], Strides: [4, 1, 12]
    assert_eq!(view.shape(), vec![3, 4, 2], "Permuted shape mismatch");
    assert_eq!(view.strides(), vec![4, 1, 12], "Permuted strides mismatch");

    // Check values
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0);  // Original [0, 0, 0]
    // view[1, 2, 0] -> original[0, 1, 2] = offset 0 + 0*12 + 1*4 + 2*1 = 6 -> val 6.0
    assert_eq!(view.get(&[1, 2, 0]).unwrap(), 6.0);
    // view[2, 3, 1] -> original[1, 2, 3] = offset 0 + 1*12 + 2*4 + 3*1 = 23 -> val 23.0
    assert_eq!(view.get(&[2, 3, 1]).unwrap(), 23.0);
}

#[test]
fn test_permute_identity() {
    // Shape: [2, 3, 4], Strides: [12, 4, 1]
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    let view = tensor.permute(&[0, 1, 2]).expect("Identity permute failed");

    // Expected Shape: [2, 3, 4], Strides: [12, 4, 1]
    assert_eq!(view.shape(), vec![2, 3, 4], "Identity permute shape mismatch");
    assert_eq!(view.strides(), vec![12, 4, 1], "Identity permute strides mismatch");

    // Should be a new view struct, but logically equal
    assert_ne!(Arc::as_ptr(&tensor.data), Arc::as_ptr(&view.data), "Identity permute should create new view struct");
    // Check equality using PartialEq impl
    assert_eq!(tensor, view, "Identity permute view should be equal to original based on PartialEq");


    // Check data sharing
    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "Identity permute view should share data");
    assert_eq!(view.read_data().offset, 0, "Identity permute view offset should be 0");
}

#[test]
fn test_permute_invalid_dims() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]); // Rank 2

    // Incorrect number of dims
    let result_wrong_len = tensor.permute(&[0]);
    assert!(matches!(result_wrong_len, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for wrong permute length");

    // Dimension out of bounds
    let result_oob = tensor.permute(&[0, 2]); // 2 is out of bounds for rank 2
    assert!(matches!(result_oob, Err(NeuraRustError::InvalidPermutation { .. })), "Expected InvalidPermutation for OOB dim");

    // Duplicate dimension
    let result_dup = tensor.permute(&[0, 0]);
    assert!(matches!(result_dup, Err(NeuraRustError::InvalidPermutation { .. })), "Expected InvalidPermutation for duplicate dim");
}

#[test]
fn test_is_contiguous() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();

    // Standard tensor
    let tensor_std = create_test_tensor(data.clone(), vec![2, 3, 4]);
    assert!(tensor_std.is_contiguous(), "Standard tensor should be contiguous");

    // Scalar tensor
    let tensor_scalar = create_test_tensor(vec![1.0], vec![]);
    assert!(tensor_scalar.is_contiguous(), "Scalar tensor should be contiguous");

    // Tensor with dimension size 1
    let tensor_dim1 = create_test_tensor(vec![1.0, 2.0], vec![1, 2]); // Strides [2, 1]
    assert!(tensor_dim1.is_contiguous(), "Tensor with dim size 1 should be contiguous");
    let tensor_dim1_end = create_test_tensor(vec![1.0, 2.0], vec![2, 1]); // Strides [1, 1]
    assert!(tensor_dim1_end.is_contiguous(), "Tensor with dim size 1 at end should be contiguous");
    let tensor_dim1_mid = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]); // Strides [2, 2, 1]
    assert!(tensor_dim1_mid.is_contiguous(), "Tensor with dim size 1 in middle should be contiguous");

    // Tensor with dimension size 0
    let tensor_dim0 = create_test_tensor(Vec::<f32>::new(), vec![2, 0, 3]);
    assert!(tensor_dim0.is_contiguous(), "Tensor with dim size 0 should be contiguous");

    // Transposed tensor (usually not contiguous)
    let tensor_transposed = tensor_std.transpose(1, 2).unwrap();
    assert!(!tensor_transposed.is_contiguous(), "Transposed tensor should not be contiguous");

    // Permuted tensor (usually not contiguous unless identity)
    let tensor_permuted = tensor_std.permute(&[2, 0, 1]).unwrap();
    assert!(!tensor_permuted.is_contiguous(), "Permuted tensor should not be contiguous");
    let tensor_permuted_id = tensor_std.permute(&[0, 1, 2]).unwrap();
    assert!(tensor_permuted_id.is_contiguous(), "Identity-permuted tensor view should be contiguous");

    // Sliced tensor (contiguous if slice covers full inner dimensions and starts at 0)
    let slice_contig1: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 2), (0, 3), (0, 4)]; // Full slice
    assert!(tensor_std.slice(&slice_contig1).unwrap().is_contiguous(), "Full slice should be contiguous");
    let slice_contig2: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (0, 3), (0, 4)]; // First outer slice
    assert!(tensor_std.slice(&slice_contig2).unwrap().is_contiguous(), "Slice of first outer dim should be contiguous");

    let slice_noncontig1: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 2), (0, 3), (1, 4)]; // Slice inner dim not at start
    assert!(!tensor_std.slice(&slice_noncontig1).unwrap().is_contiguous(), "Slice inner dim not at start should not be contiguous");
    let slice_noncontig2: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 2), (1, 3), (0, 4)]; // Slice middle dim not at start
    assert!(!tensor_std.slice(&slice_noncontig2).unwrap().is_contiguous(), "Slice middle dim not at start should not be contiguous");

    // Slice resulting in dim size 1
    let slice_dim1: Vec<crate::ops::view_ops::SliceArg> = vec![(0, 1), (0, 1), (0, 4)]; // Shape [1, 1, 4]
    assert!(tensor_std.slice(&slice_dim1).unwrap().is_contiguous(), "Slice resulting in dim size 1 should be contiguous");
}

#[test]
fn test_reshape_contiguous() {
    let data = (0..12).map(|x| x as i32).collect::<Vec<i32>>();
    let tensor = create_test_tensor(data.clone(), vec![2, 6]); // Contiguous

    // Reshape to [3, 4]
    let view = tensor.reshape(vec![3, 4]).expect("Reshape contiguous failed");
    assert_eq!(view.shape(), vec![3, 4], "Reshaped shape mismatch");
    assert_eq!(view.strides(), vec![4, 1], "Reshaped strides mismatch"); // Should be contiguous strides for new shape
    assert!(view.is_contiguous(), "Reshaped view should be contiguous");

    // Check data sharing and offset
    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "Reshape view should share data");
    assert_eq!(view.read_data().offset, 0, "Reshape view offset should be 0");

    // Check values
    assert_eq!(view.get(&[0, 0]).unwrap(), 0); // Original [0, 0] -> value 0
    assert_eq!(view.get(&[1, 1]).unwrap(), 5); // Index 1*4 + 1 = 5 -> value 5
    assert_eq!(view.get(&[2, 3]).unwrap(), 11); // Index 2*4 + 3 = 11 -> value 11
}

#[test]
fn test_reshape_to_scalar() {
    let tensor = create_test_tensor(vec![42.0_f64], vec![1]);
    let view = tensor.reshape(vec![]).expect("Reshape to scalar failed");
    assert_eq!(view.shape(), Vec::<usize>::new(), "Scalar shape mismatch");
    assert_eq!(view.strides(), Vec::<usize>::new(), "Scalar strides mismatch");
    assert!(view.is_contiguous(), "Scalar view should be contiguous");
    assert_eq!(view.get(&[]).unwrap(), 42.0, "Scalar value mismatch");
}

#[test]
fn test_reshape_from_scalar() {
    let tensor = Tensor::scalar(42.0_f64); // Create using scalar helper
    let view = tensor.reshape(vec![1, 1]).expect("Reshape from scalar failed");
    assert_eq!(view.shape(), vec![1, 1], "Reshaped scalar shape mismatch");
    assert_eq!(view.strides(), vec![1, 1], "Reshaped scalar strides mismatch");
    assert!(view.is_contiguous(), "Reshaped scalar view should be contiguous");
    assert_eq!(view.get(&[0, 0]).unwrap(), 42.0, "Reshaped scalar value mismatch");
}

#[test]
fn test_reshape_non_contiguous_error() {
    let data = (0..12).map(|x| x as i32).collect::<Vec<i32>>();
    let tensor_std = create_test_tensor(data.clone(), vec![2, 6]);

    // Create a non-contiguous view (transpose)
    let tensor_noncontig = tensor_std.transpose(0, 1).unwrap();
    assert!(!tensor_noncontig.is_contiguous());

    // Attempt to reshape the non-contiguous view
    let result = tensor_noncontig.reshape(vec![3, 4]);
    assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(_))), "Expected UnsupportedOperation for reshape non-contiguous");
    if let Err(NeuraRustError::UnsupportedOperation(msg)) = result {
        assert!(msg.contains("contiguous"), "Error message should mention contiguity");
    }

    // Attempt to reshape a likely non-contiguous slice (e.g., slicing middle dim)
    // Use a slice known to be non-contiguous from test_is_contiguous
    let slice_noncontig = tensor_std.slice(&[(0, 2), (1, 5)]).unwrap_or_else(|e| {
         panic!("Slice creation failed unexpectedly: {:?}", e); // Should not fail here
    });
    assert!(!slice_noncontig.is_contiguous(), "Slice [(0, 2), (1, 5)] should be non-contiguous");
    let result_slice = slice_noncontig.reshape(vec![8]); // Target shape with 2*(5-1) = 8 elements
    assert!(matches!(result_slice, Err(NeuraRustError::UnsupportedOperation(_))), "Expected UnsupportedOperation for reshape non-contiguous slice");
}

#[test]
fn test_reshape_numel_mismatch() {
    let data = (0..12).map(|x| x as i32).collect::<Vec<i32>>();
    let original_shape = vec![2, 6];
    let target_shape = vec![3, 5];
    let tensor = create_test_tensor(data.clone(), original_shape.clone()); // 12 elements

    // Attempt to reshape to shape with different number of elements
    let result = tensor.reshape(target_shape.clone()); // 15 elements
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })), "Expected ShapeMismatch for wrong number of elements");
    // Correct the field names and compare shapes
    if let Err(NeuraRustError::ShapeMismatch { expected, actual, operation, .. }) = result {
         assert_eq!(expected, original_shape); // Check original shape
         assert_eq!(actual, target_shape); // Check target shape
         assert_eq!(operation, "reshape");
     }
}

#[test]
fn test_contiguous_on_contiguous() {
    let data = (0..6).map(|x| x as i32).collect::<Vec<i32>>();
    let tensor = create_test_tensor(data.clone(), vec![2, 3]);
    assert!(tensor.is_contiguous());

    let contiguous_tensor = tensor.contiguous().expect(".contiguous() failed");

    // Should return a clone of the original tensor (same Arc pointer to RwLock)
    assert!(Arc::ptr_eq(&tensor.data, &contiguous_tensor.data), "Contiguous on contiguous should return same Arc<RwLock>");
    assert_eq!(tensor, contiguous_tensor);
}

#[test]
fn test_contiguous_on_transpose() {
    // Tensor: [[0, 1, 2],
    //          [3, 4, 5]] Shape: [2, 3], Strides: [3, 1]
    let data = (0..6).map(|x| x as i32).collect::<Vec<i32>>();
    let tensor = create_test_tensor(data.clone(), vec![2, 3]);

    // Transposed view: [[0, 3],
    //                  [1, 4],
    //                  [2, 5]] Shape: [3, 2], Strides: [1, 3] (Not contiguous)
    let transposed_view = tensor.transpose(0, 1).unwrap();
    assert!(!transposed_view.is_contiguous());

    let contiguous_tensor = transposed_view.contiguous().expect(".contiguous() on transpose failed");

    // Should be a new tensor with copied data, contiguous strides
    assert!(!Arc::ptr_eq(&transposed_view.data, &contiguous_tensor.data), "Contiguous on transpose should return new Arc<RwLock>");
    assert_eq!(contiguous_tensor.shape(), vec![3, 2], "Contiguous tensor shape mismatch");
    assert_eq!(contiguous_tensor.strides(), vec![2, 1], "Contiguous tensor strides mismatch");
    assert!(contiguous_tensor.is_contiguous(), "Result of .contiguous() must be contiguous");

    // Check data content
    let expected_data = vec![0, 3, 1, 4, 2, 5];
    // Access data via .get() or a method that reads the buffer respecting new layout
    let mut actual_data = Vec::with_capacity(6);
    for i in 0..3 {
        for j in 0..2 {
            actual_data.push(contiguous_tensor.get(&[i,j]).unwrap());
        }
    }
    assert_eq!(actual_data, expected_data, "Contiguous tensor data mismatch");
    // Optional: Also check the raw buffer if needed, but .get() is safer
    // let contiguous_buffer_data = contiguous_tensor.borrow_data_buffer().cpu_data().unwrap().clone();
    // assert_eq!(*contiguous_buffer_data, expected_data, "Contiguous tensor raw buffer data mismatch");
}

#[test]
fn test_contiguous_on_permute() {
    // Shape: [2, 2, 2], Strides: [4, 2, 1]
    let data = (0..8).map(|x| x as i32).collect::<Vec<i32>>();
    let tensor = create_test_tensor(data, vec![2, 2, 2]);

    // Permuted view [2, 0, 1] -> Shape [2, 2, 2], Strides [1, 4, 2] (Not contiguous)
    let permuted_view = tensor.permute(&[2, 0, 1]).unwrap();
    assert!(!permuted_view.is_contiguous());

    let contiguous_tensor = permuted_view.contiguous().expect(".contiguous() on permute failed");

    // Should be a new tensor with copied data, contiguous strides
    assert!(!Arc::ptr_eq(&permuted_view.data, &contiguous_tensor.data));
    assert_eq!(contiguous_tensor.shape(), vec![2, 2, 2]);
    assert_eq!(contiguous_tensor.strides(), vec![4, 2, 1]); // Contiguous strides for [2, 2, 2]
    assert!(contiguous_tensor.is_contiguous());

    // Check data content carefully
    // Expected order based on iterating through the permuted view's logical indices:
    // Indices (permuted): [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]
    // Corresponds to orig: [0,0,0], [0,1,0], [1,0,0], [1,1,0], [0,0,1], [0,1,1], [1,0,1], [1,1,1]
    // Original values:     0,       2,       4,       6,       1,       3,       5,       7
    let expected_data = vec![0, 2, 4, 6, 1, 3, 5, 7];

    // Access data via .get()
    let mut actual_data = Vec::with_capacity(8);
     for i in 0..2 { for j in 0..2 { for k in 0..2 { actual_data.push(contiguous_tensor.get(&[i,j,k]).unwrap()); } } }
    assert_eq!(actual_data, expected_data, "Contiguous tensor data mismatch");
}

#[test]
fn test_contiguous_on_slice() {
    // Shape [2, 3, 4], Strides [12, 4, 1]
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Slice: [1, 1:3, 0:2] -> Shape [1, 2, 2], Strides [12, 4, 1], Offset 16 (Not contiguous)
    let slice_view = tensor.slice(&[(1, 2), (1, 3), (0, 2)]).unwrap();
    assert!(!slice_view.is_contiguous());

    let contiguous_tensor = slice_view.contiguous().expect(".contiguous() on slice failed");

    // Should be a new tensor
    assert!(!Arc::ptr_eq(&slice_view.data, &contiguous_tensor.data));
    assert_eq!(contiguous_tensor.shape(), vec![1, 2, 2]);
    assert_eq!(contiguous_tensor.strides(), vec![4, 2, 1]); // Contiguous strides for [1, 2, 2]
    assert!(contiguous_tensor.is_contiguous());

    // Expected data by iterating the slice view:
    // Indices (slice): [0,0,0], [0,0,1], [0,1,0], [0,1,1]
    // Corresponds to orig: [1,1,0], [1,1,1], [1,2,0], [1,2,1]
    // Original values: 1*12+1*4+0*1=16, 16+1=17, 1*12+2*4+0*1=20, 20+1=21
    let expected_data = vec![16.0, 17.0, 20.0, 21.0];

    // Access data via .get()
    let mut actual_data = Vec::with_capacity(4);
    for i in 0..1 { for j in 0..2 { for k in 0..2 { actual_data.push(contiguous_tensor.get(&[i,j,k]).unwrap()); } } }
    assert_eq!(actual_data, expected_data, "Contiguous tensor data mismatch");
}

#[test]
fn test_view_ops_on_scalar() {
    let scalar = Tensor::scalar(5.0_f32);

    // Slice (invalid ndim)
    assert!(matches!(scalar.slice(&[(0, 0)]), Err(NeuraRustError::DimensionMismatch { .. })));

    // Transpose (invalid ndim)
    assert!(matches!(scalar.transpose(0, 0), Err(NeuraRustError::DimensionMismatch { .. })));

    // Permute (invalid ndim)
    assert!(matches!(scalar.permute(&[]), Err(NeuraRustError::DimensionMismatch { .. }))); // Should fail on len mismatch

    // Reshape
    let reshaped = scalar.reshape(vec![1, 1]);
    assert!(reshaped.is_ok());
    assert_eq!(reshaped.unwrap().shape(), vec![1, 1]);

    // Contiguous
    let contiguous_scalar = scalar.contiguous().unwrap();
    assert!(Arc::ptr_eq(&scalar.data, &contiguous_scalar.data)); // Should be clone
    assert_eq!(scalar, contiguous_scalar);
}

#[test]
fn test_view_ops_on_zero_dim_tensor() {
    // Test case for tensor created with shape like [2, 0, 3]
    let zero_dim_tensor = Tensor::<i32>::new(vec![], vec![2, 0, 3]).unwrap();

    assert_eq!(zero_dim_tensor.numel(), 0);
    assert!(zero_dim_tensor.is_contiguous());

    // Slice
    let sliced = zero_dim_tensor.slice(&[(0, 1), (0, 0), (1, 2)]);
    assert!(sliced.is_ok());
    let sliced_tensor = sliced.unwrap();
    assert_eq!(sliced_tensor.shape(), vec![1, 0, 1]);
    assert_eq!(sliced_tensor.numel(), 0);

    // Transpose
    let transposed = zero_dim_tensor.transpose(0, 1);
    assert!(transposed.is_ok());
    let transposed_tensor = transposed.unwrap();
    assert_eq!(transposed_tensor.shape(), vec![0, 2, 3]);
    assert_eq!(transposed_tensor.numel(), 0);

    // Permute
    let permuted = zero_dim_tensor.permute(&[1, 2, 0]);
    assert!(permuted.is_ok());
    let permuted_tensor = permuted.unwrap();
    assert_eq!(permuted_tensor.shape(), vec![0, 3, 2]);
    assert_eq!(permuted_tensor.numel(), 0);

    // Reshape (to another 0-element shape)
    let reshaped = zero_dim_tensor.reshape(vec![6, 0]);
    assert!(reshaped.is_ok());
    let reshaped_tensor = reshaped.unwrap();
    assert_eq!(reshaped_tensor.shape(), vec![6, 0]);
    assert_eq!(reshaped_tensor.numel(), 0);

    // Reshape (to non-zero elements -> should fail)
    let reshape_fail = zero_dim_tensor.reshape(vec![1]);
    assert!(matches!(reshape_fail, Err(NeuraRustError::ShapeMismatch { .. })));

    // Contiguous
    let contiguous_zero = zero_dim_tensor.contiguous().unwrap();
    assert!(Arc::ptr_eq(&zero_dim_tensor.data, &contiguous_zero.data)); // Should be clone
    assert_eq!(zero_dim_tensor, contiguous_zero);
} 