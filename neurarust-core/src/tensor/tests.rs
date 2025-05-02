// neurarust-core/src/tensor/tests.rs
// This file contains the unit tests previously located in tensor/mod.rs

use super::*; // Import everything from the parent module (tensor/mod.rs)
use approx::assert_relative_eq;
use num_traits::{Zero, One};
use std::ops::AddAssign;
use std::fmt::Debug;
use std::iter::Sum;
use std::cmp::PartialEq;
use std::default::Default;
use std::sync::Arc;

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
    let result = Tensor::<f32>::new(data, shape);
    assert!(result.is_err());
    match result.err().unwrap() {
        NeuraRustError::TensorCreationError { data_len, shape: err_shape } => {
            assert_eq!(data_len, 3);
            assert_eq!(err_shape, vec![2, 2]);
        }
        _ => panic!("Expected TensorCreationError"),
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
    assert_ne!(t1, t2, "t1 and t2 should have different Arc<RwLock> pointers");

    assert_ne!(t1, t4); // Different data Arc pointer
    assert_ne!(t1, t5); // Different shape
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
        _ => panic!("Expected IndexOutOfBounds"),
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
        _ => panic!("Expected DimensionMismatch"),
    }
}

#[test]
fn test_zeros_creation() {
    let shape = vec![2, 3];
    let t = zeros::<f64>(shape.clone()).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 6);
    for i in 0..2 { for j in 0..3 { assert_relative_eq!(t.get(&[i, j]).unwrap(), 0.0); } }
}

#[test]
fn test_ones_creation() {
    let shape = vec![1, 4];
    let t = ones::<i32>(shape.clone()).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 4);
    for j in 0..4 { assert_eq!(t.get(&[0, j]).unwrap(), 1); }
}

#[test]
fn test_full_creation() {
    let shape = vec![3, 1, 2];
    let fill_val = 42.5_f32;
    let t = full(shape.clone(), fill_val).unwrap();
    assert_eq!(t.shape(), shape);
    assert_eq!(t.numel(), 6);
    for i in 0..3 { for j in 0..1 { for k in 0..2 { assert_relative_eq!(t.get(&[i, j, k]).unwrap(), fill_val); } } }
}

#[test]
fn test_simple_slice() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let ranges = vec![(0, 1), (0, 3), (0, 4)];
    let view = tensor.slice(&ranges).expect("Simple slice failed");

    assert_eq!(view.shape(), vec![1, 3, 4], "View shape mismatch");
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0, "Value mismatch at [0,0,0]");
    assert_eq!(view.get(&[0, 2, 3]).unwrap(), 11.0, "Value mismatch at [0,2,3]");
}

#[test]
fn test_slice_shares_data() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let ranges = vec![(1, 2), (1, 3), (0, 2)];
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
    let ranges = vec![(1, 2), (1, 3), (0, 2)]; // Slice: [1, 1:3, 0:2] -> Shape [1, 2, 2]
    let view = tensor.slice(&ranges).expect("Slice for metadata test failed");

    let view_data = view.read_data();

    assert_eq!(view_data.shape, vec![1, 2, 2], "View shape mismatch");
    assert_eq!(view_data.strides, vec![12, 4, 1], "View strides should be inherited");
    assert_eq!(view_data.offset, 16, "View offset calculation incorrect");

    // Drop guard before calling view.get() which also locks
    drop(view_data);
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 16.0, "First element value mismatch");
}

#[test]
fn test_slice_invalid_range() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Range end > dimension size
    let ranges_invalid_end = vec![(0, 1), (0, 4), (0, 4)]; // Dim 1 size is 3, end is 4
    let result_invalid_end = tensor.slice(&ranges_invalid_end);
    assert!(matches!(result_invalid_end, Err(NeuraRustError::SliceError { .. })), "Expected SliceError for end > dim size");

    // Range start >= end
    let ranges_invalid_start = vec![(0, 1), (2, 2), (0, 4)]; // Dim 1 start == end
    let result_invalid_start = tensor.slice(&ranges_invalid_start);
    assert!(matches!(result_invalid_start, Err(NeuraRustError::SliceError { .. })), "Expected SliceError for start >= end");

    // Incorrect number of ranges
    let ranges_wrong_ndim = vec![(0, 1), (0, 1)]; // Only 2 ranges for 3 dims
    let result_wrong_ndim = tensor.slice(&ranges_wrong_ndim);
    // Use DimensionMismatch as per the updated slice_op
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
    // view[0, 1, 2] -> original[0, 2, 1] = 4*2 + 1 = 9
    assert_eq!(view.get(&[0, 1, 2]).unwrap(), 9.0);
    // view[1, 3, 0] -> original[1, 0, 3] = 12*1 + 4*0 + 1*3 = 15
    assert_eq!(view.get(&[1, 3, 0]).unwrap(), 15.0);
    // view[1, 2, 1] -> original[1, 1, 2] = 12*1 + 4*1 + 1*2 = 18
     assert_eq!(view.get(&[1, 2, 1]).unwrap(), 18.0);
}

#[test]
fn test_transpose_invalid_dims() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]); // Rank 2

    // Dim >= rank
    let result1 = tensor.transpose(0, 2);
    assert!(matches!(result1, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for dim >= rank");

    let result2 = tensor.transpose(2, 1);
    assert!(matches!(result2, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for dim >= rank");

    // Transposing same dimension should ideally work (no-op), let's test it
    let view_same_dim = tensor.transpose(1, 1).expect("Transposing same dimension failed");
    assert_eq!(view_same_dim.shape(), vec![2, 3]);
    assert_eq!(view_same_dim.strides(), vec![3, 1]);
    assert_eq!(view_same_dim, tensor, "Transposing same dim should be identity (data pointer check)");
    // Note: The above assert_eq checks pointer equality, not necessarily value equality or metadata equality.
    // Let's check metadata explicitly
    let view_data = view_same_dim.read_data();
    let tensor_data = tensor.read_data();
    assert_eq!(view_data.shape, tensor_data.shape);
    assert_eq!(view_data.strides, tensor_data.strides);
    assert_eq!(view_data.offset, tensor_data.offset);

} 