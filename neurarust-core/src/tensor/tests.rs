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

#[test]
fn test_permute_simple() {
    // Shape: [2, 3], Strides: [3, 1]
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);

    // Permute dims [0, 1] -> [1, 0] (equivalent to transpose(0, 1))
    let view = tensor.permute(&[1, 0]).expect("Permute simple failed");

    // Expected Shape: [3, 2], Strides: [1, 3]
    assert_eq!(view.shape(), vec![3, 2], "Permuted shape mismatch");
    assert_eq!(view.strides(), vec![1, 3], "Permuted strides mismatch");

    // Verify data sharing and offset
    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "Permute view should share data");
    assert_eq!(view.read_data().offset, 0, "Permute should not change offset");

    // Check values (should match transpose test)
    assert_eq!(view.get(&[0, 0]).unwrap(), 0.0); // Original [0, 0]
    assert_eq!(view.get(&[1, 0]).unwrap(), 1.0); // Original [0, 1]
    assert_eq!(view.get(&[2, 0]).unwrap(), 2.0); // Original [0, 2]
    assert_eq!(view.get(&[0, 1]).unwrap(), 3.0); // Original [1, 0]
    assert_eq!(view.get(&[1, 1]).unwrap(), 4.0); // Original [1, 1]
    assert_eq!(view.get(&[2, 1]).unwrap(), 5.0); // Original [1, 2]
}

#[test]
fn test_permute_higher_dim() {
    // Shape: [2, 3, 4], Strides: [12, 4, 1]
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Permute [0, 1, 2] -> [2, 0, 1]
    let view = tensor.permute(&[2, 0, 1]).expect("Permute higher dim failed");

    // Expected Shape: [4, 2, 3], Strides: [1, 12, 4]
    assert_eq!(view.shape(), vec![4, 2, 3], "Permuted shape mismatch");
    assert_eq!(view.strides(), vec![1, 12, 4], "Permuted strides mismatch");

    // Check a few values
    // view[0, 0, 0] -> original[0, 0, 0] = 0
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 0.0);
    // view[1, 0, 2] -> original[0, 2, 1] = 0*12 + 2*4 + 1*1 = 9
    assert_eq!(view.get(&[1, 0, 2]).unwrap(), 9.0);
    // view[3, 1, 0] -> original[1, 0, 3] = 1*12 + 0*4 + 3*1 = 15
    assert_eq!(view.get(&[3, 1, 0]).unwrap(), 15.0);
    // view[2, 1, 1] -> original[1, 1, 2] = 1*12 + 1*4 + 2*1 = 18
    assert_eq!(view.get(&[2, 1, 1]).unwrap(), 18.0);
}

#[test]
fn test_permute_identity() {
    // Shape: [2, 3, 4], Strides: [12, 4, 1]
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // Permute [0, 1, 2] -> [0, 1, 2]
    let view = tensor.permute(&[0, 1, 2]).expect("Permute identity failed");

    assert_eq!(view.shape(), vec![2, 3, 4]);
    assert_eq!(view.strides(), vec![12, 4, 1]);
    assert_eq!(view, tensor, "Permute identity should be equal (pointer check)");

    // Check metadata explicitly
    let view_data = view.read_data();
    let tensor_data = tensor.read_data();
    assert_eq!(view_data.shape, tensor_data.shape);
    assert_eq!(view_data.strides, tensor_data.strides);
    assert_eq!(view_data.offset, tensor_data.offset);
}

#[test]
fn test_permute_invalid_dims() {
    // Shape: [2, 3]
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);

    // Incorrect number of dims
    let result_wrong_len = tensor.permute(&[1, 0, 2]);
    assert!(matches!(result_wrong_len, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for wrong number of dims");

    let result_wrong_len2 = tensor.permute(&[0]);
    assert!(matches!(result_wrong_len2, Err(NeuraRustError::DimensionMismatch { .. })), "Expected DimensionMismatch for wrong number of dims");

    // Invalid permutation (duplicate dim)
    let result_duplicate = tensor.permute(&[1, 1]);
    assert!(matches!(result_duplicate, Err(NeuraRustError::InvalidPermutation { .. })), "Expected InvalidPermutation for duplicate dims");

    // Invalid permutation (dim out of bounds)
    let result_oob = tensor.permute(&[0, 2]);
    assert!(matches!(result_oob, Err(NeuraRustError::InvalidPermutation { .. })), "Expected InvalidPermutation for out-of-bounds dim");
}

#[test]
fn test_is_contiguous() {
    // Standard contiguous tensor
    let data1 = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor1 = create_test_tensor(data1, vec![2, 3]); // Strides [3, 1]
    assert!(tensor1.is_contiguous(), "Standard 2x3 tensor should be contiguous");

    // Scalar
    let tensor_scalar = create_test_tensor(vec![5.0], vec![]);
    assert!(tensor_scalar.is_contiguous(), "Scalar tensor should be contiguous");

    // Tensor with dimension size 1
    let tensor_dim1 = create_test_tensor((0..4).map(|x| x as f32).collect(), vec![4, 1]); // Strides [1, 1]
    assert!(tensor_dim1.is_contiguous(), "Tensor with dim size 1 can be contiguous");
    let tensor_dim1_b = create_test_tensor((0..4).map(|x| x as f32).collect(), vec![1, 4]); // Strides [4, 1]
    assert!(tensor_dim1_b.is_contiguous(), "Tensor with dim size 1 (leading) should be contiguous");

    // Non-contiguous: Transpose
    let view_transposed = tensor1.transpose(0, 1).expect("Transpose failed"); // Shape [3, 2], Strides [1, 3]
    assert!(!view_transposed.is_contiguous(), "Transposed tensor should not be contiguous");

    // Non-contiguous: Slice with step > 1 (if slicing supported steps)
    // For now, test slice that results in non-contiguous strides implicitly
    let data2 = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor2 = create_test_tensor(data2, vec![2, 3, 4]); // Strides [12, 4, 1]
    // Slice [:, :, 0:4:2] -> Not possible with current slice, but imagine strides [12, 4, 2]
    // Let's use permute to create a non-contiguous case
    let view_permuted = tensor2.permute(&[2, 0, 1]).expect("Permute failed"); // Shape [4, 2, 3], Strides [1, 12, 4]
    assert!(!view_permuted.is_contiguous(), "Permuted tensor [2,0,1] should not be contiguous");

    // Slice that *remains* contiguous
    let view_contig_slice = tensor2.slice(&[(0, 1), (0, 3), (0, 4)]).expect("Slice failed"); // Shape [1, 3, 4], Strides [12, 4, 1], Offset 0
    // Even though it's a view, its logical layout matches C order for its shape
    assert!(view_contig_slice.is_contiguous(), "Slice resulting in shape [1, 3, 4] should be contiguous");

    let view_contig_slice2 = tensor2.slice(&[(0, 2), (1, 3), (0, 4)]).expect("Slice failed"); // Shape [2, 2, 4], Strides[12, 4, 1], Offset 4
    // This slice is NOT contiguous because the stride for the first dimension (12)
    // is not equal to the product of the remaining dimensions (2 * 4 = 8).
    assert!(!view_contig_slice2.is_contiguous(), "Slice shape [2, 2, 4] from [2, 3, 4] should NOT be contiguous");

    // Slice that becomes non-contiguous
    let view_noncontig_slice = tensor2.slice(&[(0, 2), (0, 3), (1, 3)]).expect("Slice failed"); // Shape [2, 3, 2], Strides[12, 4, 1], Offset 1
    assert!(!view_noncontig_slice.is_contiguous(), "Slice shape [2, 3, 2] from [2, 3, 4] (offset 1) should not be contiguous");

    // Tensor with zero dimension
    let tensor_zero_dim = create_test_tensor(Vec::<f32>::new(), vec![2, 0, 3]);
    assert!(tensor_zero_dim.is_contiguous(), "Tensor with zero dimension should be contiguous");
}

#[test]
fn test_reshape_contiguous() {
    // Shape [2, 6], Strides [6, 1]
    let data = (0..12).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 6]);
    assert!(tensor.is_contiguous());

    // Reshape to [3, 4]
    let view = tensor.reshape(vec![3, 4]).expect("Reshape contiguous failed");

    // Expected Shape: [3, 4], Strides: [4, 1]
    assert_eq!(view.shape(), vec![3, 4], "Reshaped shape mismatch");
    assert_eq!(view.strides(), vec![4, 1], "Reshaped strides mismatch");

    // Verify data sharing and offset
    assert!(Arc::ptr_eq(&tensor.borrow_data_buffer(), &view.borrow_data_buffer()), "Reshape view should share data");
    assert_eq!(view.read_data().offset, 0, "Reshape should not change offset for contiguous input");

    // Check values
    assert_eq!(view.get(&[0, 0]).unwrap(), 0.0); // Original [0, 0]
    assert_eq!(view.get(&[1, 1]).unwrap(), 5.0); // Original [0, 5]
    assert_eq!(view.get(&[2, 3]).unwrap(), 11.0); // Original [1, 5]
}

#[test]
fn test_reshape_to_scalar() {
    let tensor = create_test_tensor(vec![42.0], vec![1, 1]);
    assert!(tensor.is_contiguous());
    let view = tensor.reshape(vec![]).expect("Reshape to scalar failed");
    assert_eq!(view.shape(), vec![], "Reshaped shape mismatch");
    assert_eq!(view.strides(), vec![], "Reshaped strides mismatch");
    assert_eq!(view.get(&[]).unwrap(), 42.0);
}

#[test]
fn test_reshape_from_scalar() {
    let tensor = create_test_tensor(vec![42.0], vec![]);
    assert!(tensor.is_contiguous());
    let view = tensor.reshape(vec![1, 1, 1]).expect("Reshape from scalar failed");
    assert_eq!(view.shape(), vec![1, 1, 1], "Reshaped shape mismatch");
    assert_eq!(view.strides(), vec![1, 1, 1], "Reshaped strides mismatch");
    assert_eq!(view.get(&[0, 0, 0]).unwrap(), 42.0);
}

#[test]
fn test_reshape_non_contiguous_error() {
    // Create a non-contiguous tensor via transpose
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);
    let transposed_view = tensor.transpose(0, 1).expect("Transpose failed");
    assert!(!transposed_view.is_contiguous());

    // Attempt to reshape the non-contiguous view
    let result = transposed_view.reshape(vec![6]);
    assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(_))), "Expected UnsupportedOperation for reshaping non-contiguous tensor");
    // Check error message content if needed
    if let Err(NeuraRustError::UnsupportedOperation(msg)) = result {
        assert!(msg.contains("contiguous"), "Error message should mention contiguity");
    }
}

#[test]
fn test_reshape_numel_mismatch() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);

    // Attempt reshape with wrong number of elements
    let result = tensor.reshape(vec![2, 2]);
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })), "Expected ShapeMismatch for wrong number of elements");

    if let Err(NeuraRustError::ShapeMismatch { expected, actual }) = result {
        assert_eq!(expected, vec![2, 3]);
        assert_eq!(actual, vec![2, 2]);
    }
} 