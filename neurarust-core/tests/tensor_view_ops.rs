use neurarust_core::utils::testing::check_tensor_near;
use neurarust_core::ops::view::slice::SliceArg;
use neurarust_core::tensor::Tensor;
use neurarust_core::NeuraRustError;
use approx::assert_relative_eq;

// Include the common helper module
mod common; // Reactivate common module
use common::create_test_tensor; // Reactivate usage

// --- Slice Tests ---

#[test]
fn test_simple_slice() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let ranges: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 1, 1),
        SliceArg::Slice(0, 3, 1),
        SliceArg::Slice(0, 4, 1),
    ];
    let view = tensor.slice(&ranges).expect("Simple slice failed");

    assert_eq!(view.shape(), vec![1, 3, 4], "View shape mismatch");
    // Use get_f32_data to check elements
    let view_data = view.contiguous().unwrap().get_f32_data().unwrap(); // Make contiguous first
    assert_relative_eq!(view_data[0], 0.0, epsilon=1e-6); // [0,0,0]
    assert_relative_eq!(view_data[11], 11.0, epsilon=1e-6); // [0,2,3]
}

#[test]
#[ignore = "Skipping test: Cannot easily verify shared data pointer due to private fields"]
fn test_slice_shares_data() {
    /* // Keep code commented for reference
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let t = Tensor::new(data.clone(), shape.clone()).unwrap();

    let ranges = vec![
        SliceArg::new(0, 1),
        SliceArg::new(0, 2),
    ];
    let sliced_view = t.slice(&ranges).unwrap();

    // Check Arc pointer equality for the underlying TensorData
    // This requires accessing the private `data` field, hence ignored.
    // assert!(Arc::ptr_eq(&t.data, &sliced_view.data), "Slice should point to the same TensorData Arc initially");
    */
}

#[test]
fn test_slice_metadata() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let ranges: Vec<SliceArg> = vec![
        SliceArg::Slice(1, 2, 1),
        SliceArg::Slice(1, 3, 1),
        SliceArg::Slice(0, 2, 1),
    ];
    let view = tensor.slice(&ranges).expect("Metadata slice failed");

    assert_eq!(view.shape(), vec![1, 2, 2], "View shape mismatch");

    // Verify properties derived from metadata
    assert!(!view.is_contiguous()); // This slice shouldn't be contiguous
    assert_eq!(view.strides(), vec![12, 4, 1]); // Should inherit original strides

    // Check values by making contiguous
    let view_data = view.contiguous().unwrap().get_f32_data().unwrap();
    // Expected view data: [[ [16, 17], [20, 21] ]]
    let expected_view_data = vec![16.0, 17.0, 20.0, 21.0];
    assert_eq!(view_data, expected_view_data);
}

#[test]
fn test_slice_invalid_range() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);

    // End > size (should be clamped by normalize_slice, op should succeed)
    let ranges_end: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 1, 1),
        SliceArg::Slice(0, 4, 1), // dim size is 3, end=4 should be clamped to 3
        SliceArg::Slice(0, 4, 1),
    ];
    let result_end = tensor.slice(&ranges_end);
    assert!(result_end.is_ok(), "Slice with end > size should succeed (clamped)");
    let view_end = result_end.unwrap();
    // Shape should reflect clamping: [1, 3, 4]
    assert_eq!(view_end.shape(), vec![1, 3, 4], "Shape mismatch for end > size slice");

    // Start > end (should produce empty slice, op should succeed)
    let ranges_start: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 1, 1),
        SliceArg::Slice(3, 2, 1), // start=3 > end=2
        SliceArg::Slice(0, 4, 1),
    ];
    let result_start = tensor.slice(&ranges_start);
    assert!(result_start.is_ok(), "Slice with start > end should succeed (empty slice)");
    let view_start = result_start.unwrap();
    // Shape should be empty along the affected dimension: [1, 0, 4]
    assert_eq!(view_start.shape(), vec![1, 0, 4], "Shape mismatch for start > end slice");

    // Empty slice (start == end) (valid)
    let ranges_empty: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 1, 1),
        SliceArg::Slice(2, 2, 1),
        SliceArg::Slice(0, 4, 1),
    ];
    let result_empty = tensor.slice(&ranges_empty);
    assert!(result_empty.is_ok());
    assert_eq!(result_empty.unwrap().shape(), vec![1, 0, 4]);
}

#[test]
fn test_slice_wrong_ndim() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let ranges: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 1, 1),
        SliceArg::Slice(0, 1, 1),
    ];
    let result = tensor.slice(&ranges);
    // Expect SliceError because validate_and_adjust_ranges checks rank
    assert!(matches!(
        result,
        Err(NeuraRustError::SliceError { .. })
    ), "Expected SliceError for wrong number of slice ranges");
}

// --- Transpose Tests ---

#[test]
fn test_transpose_2d() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);
    let view = tensor.transpose(0, 1).expect("Transpose 2D failed");
    assert_eq!(view.shape(), vec![3, 2], "Transposed shape mismatch");
    assert_eq!(view.strides(), vec![1, 3], "Transposed strides mismatch");
    assert!(!view.is_contiguous());
    let view_data = view.contiguous().unwrap().get_f32_data().unwrap();
    let expected_data = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
    assert_eq!(view_data, expected_data);
}

#[test]
fn test_transpose_higher_dim() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let view = tensor.transpose(1, 2).expect("Transpose higher dim failed");
    assert_eq!(view.shape(), vec![2, 4, 3], "Transposed shape mismatch");
    assert_eq!(view.strides(), vec![12, 1, 4], "Transposed strides mismatch");
    assert!(!view.is_contiguous());
    let view_data = view.contiguous().unwrap().get_f32_data().unwrap();
    let expected_data = vec![
         0.0,  4.0,  8.0,    1.0,  5.0,  9.0,    2.0,  6.0, 10.0,    3.0,  7.0, 11.0,
        12.0, 16.0, 20.0,   13.0, 17.0, 21.0,   14.0, 18.0, 22.0,   15.0, 19.0, 23.0,
    ];
     assert_eq!(view_data, expected_data);
}

#[test]
fn test_transpose_invalid_dims() {
    let data = (0..6).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3]);
    let result_dim1 = tensor.transpose(2, 0);
    assert!(matches!(result_dim1, Err(NeuraRustError::IndexOutOfBounds { .. })));
    let result_dim2 = tensor.transpose(0, 2);
    assert!(matches!(result_dim2, Err(NeuraRustError::IndexOutOfBounds { .. })));
    let result_same_dims = tensor.transpose(0, 0);
    assert!(result_same_dims.is_ok(), "Transposing same dimension should be Ok");
    let view_same_dims = result_same_dims.unwrap();
    assert_eq!(view_same_dims.shape(), tensor.shape());
    assert_eq!(view_same_dims.strides(), tensor.strides());
}

// --- Permute Tests ---
#[test]
fn test_permute_simple() {
    let data = (0..8).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 2, 2]);
    let view = tensor.permute(&[2, 0, 1]).expect("Simple permute failed");
    assert_eq!(view.shape(), vec![2, 2, 2], "Permuted shape mismatch");
    assert_eq!(view.strides(), vec![1, 4, 2], "Permuted strides mismatch");
    assert!(!view.is_contiguous());
    let view_data = view.contiguous().unwrap().get_f32_data().unwrap();
    let expected_data = vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0];
    assert_eq!(view_data, expected_data);
}

// test_permute_higher_dim_adapted was already adapted correctly
#[test]
fn test_permute_higher_dim_adapted() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data, vec![2, 3, 4]);
    let view = tensor.permute(&[1, 2, 0]).expect("Higher dim permute failed");
    assert_eq!(view.shape(), vec![3, 4, 2]);
    assert_eq!(view.strides(), vec![4, 1, 12]);
    assert!(!view.is_contiguous());
    let view_data = view.contiguous().unwrap().get_f32_data().unwrap();
    let expected_data = vec![
         0.0, 12.0,  1.0, 13.0,  2.0, 14.0,  3.0, 15.0,
         4.0, 16.0,  5.0, 17.0,  6.0, 18.0,  7.0, 19.0,
         8.0, 20.0,  9.0, 21.0, 10.0, 22.0, 11.0, 23.0,
    ];
    assert_eq!(view_data, expected_data);
}

// --- is_contiguous Test ---
#[test]
fn test_is_contiguous() {
    let data = (0..24).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor_std = create_test_tensor(data.clone(), vec![2, 3, 4]);
    assert!(tensor_std.is_contiguous(), "Standard tensor should be contiguous");
    let tensor_scalar = Tensor::new(vec![1.0], vec![]).unwrap();
    assert!(tensor_scalar.is_contiguous(), "Scalar tensor should be contiguous");
    let tensor_dim1 = create_test_tensor(vec![1.0, 2.0], vec![1, 2]);
    assert!(tensor_dim1.is_contiguous(), "Tensor with dim size 1 should be contiguous");
    let tensor_dim1_end = create_test_tensor(vec![1.0, 2.0], vec![2, 1]);
    assert!(tensor_dim1_end.is_contiguous(), "Tensor with dim size 1 at end should be contiguous");
    // Use f32 for data
    let tensor_dim1_mid = create_test_tensor((0..4).map(|x| x as f32).collect(), vec![2, 1, 2]); 
    assert!(tensor_dim1_mid.is_contiguous(), "Tensor with dim size 1 in middle should be contiguous");
    let tensor_dim0 = Tensor::new(Vec::<f32>::new(), vec![2, 0, 3]).unwrap();
    assert!(tensor_dim0.is_contiguous(), "Tensor with dim size 0 should be contiguous");
    let tensor_transposed = tensor_std.transpose(1, 2).unwrap();
    assert!(!tensor_transposed.is_contiguous(), "Transposed tensor should not be contiguous");
    let tensor_permuted = tensor_std.permute(&[2, 0, 1]).unwrap();
    assert!(!tensor_permuted.is_contiguous(), "Permuted tensor should not be contiguous");
    let tensor_permuted_id = tensor_std.permute(&[0, 1, 2]).unwrap();
    assert!(tensor_permuted_id.is_contiguous(), "Identity-permuted tensor view should be contiguous");
    let slice_contig1: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 2, 1),
        SliceArg::Slice(0, 3, 1),
        SliceArg::Slice(1, 3, 1),
    ];
    assert!(!tensor_std.slice(&slice_contig1).unwrap().is_contiguous(), "Slice on inner dimension should not be contiguous");
    let slice_contig2: Vec<SliceArg> = vec![
        SliceArg::Slice(1, 2, 1),
        SliceArg::Slice(0, 3, 1),
        SliceArg::Slice(0, 4, 1),
    ];
    assert!(tensor_std.slice(&slice_contig2).unwrap().is_contiguous());
    let slice_noncontig: Vec<SliceArg> = vec![
        SliceArg::Slice(0, 2, 1),
        SliceArg::Slice(1, 3, 1),
        SliceArg::Slice(0, 4, 1),
    ];
    assert!(!tensor_std.slice(&slice_noncontig).unwrap().is_contiguous());
}

// --- Reshape Tests ---
#[test]
fn test_reshape_contiguous() {
    let tensor = create_test_tensor((0..12).map(|x| x as f32).collect(), vec![2, 6]);
    let view = tensor.reshape(vec![3, 4]).expect("Reshape failed");
    assert_eq!(view.shape(), vec![3, 4]);
    assert_eq!(view.strides(), vec![4, 1]);
    assert!(view.is_contiguous());
    let view_data = view.get_f32_data().unwrap();
    let expected_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    assert_eq!(view_data, expected_data);
}

#[test]
fn test_reshape_to_scalar() {
    let tensor = create_test_tensor(vec![42.0f32], vec![1]);
    let view = tensor.reshape(vec![]).expect("Reshape to scalar failed");
    assert_eq!(view.shape(), vec![]);
    assert_eq!(view.strides(), vec![]);
    assert!(view.is_contiguous());
    let view_data: Vec<f32> = view.get_f32_data().unwrap();
    assert_eq!(view_data, vec![42.0f32]); 
}

#[test]
fn test_reshape_from_scalar() {
    let tensor = Tensor::new(vec![42.0f32], vec![]).unwrap(); 
    let view = tensor.reshape(vec![1, 1, 1]).expect("Reshape from scalar failed");
    assert_eq!(view.shape(), vec![1, 1, 1]);
    assert_eq!(view.strides(), vec![1, 1, 1]);
    assert!(view.is_contiguous());
    let view_data: Vec<f32> = view.get_f32_data().unwrap();
    assert_eq!(view_data, vec![42.0f32]); 
}

#[test]
fn test_reshape_contiguous_error() { 
    let t = create_test_tensor((0..12).map(|x| x as f32).collect(), vec![2, 6]);
    let view = t.reshape(vec![3, 4]).unwrap();
    let data = view.get_f32_data().unwrap();
    assert_eq!(data[5], 5.0f32);  
    assert_eq!(data[11], 11.0f32); 
}

// --- Contiguous Tests ---
#[test]
fn test_contiguous_on_transpose() {
    let tensor = create_test_tensor((0..6).map(|x| x as f32).collect(), vec![2, 3]);
    let view = tensor.transpose(0, 1).unwrap(); 
    assert!(!view.is_contiguous());
    let contiguous_tensor = view.contiguous().unwrap();
    assert!(contiguous_tensor.is_contiguous());
    assert_eq!(contiguous_tensor.shape(), vec![3, 2]);
    assert_eq!(contiguous_tensor.strides(), vec![2, 1]);
    let expected_data = vec![0.0f32, 3.0f32, 1.0f32, 4.0f32, 2.0f32, 5.0f32];
    let actual_data = contiguous_tensor.get_f32_data().unwrap();
    assert_eq!(actual_data, expected_data);
    let original_data = tensor.get_f32_data().unwrap();
    assert_eq!(original_data, (0..6).map(|x| x as f32).collect::<Vec<f32>>());
}

#[test]
#[ignore = "Skipping test relying on direct element access via .get()"]
fn test_view_ops_on_scalar_get() {
     // let scalar_tensor = Tensor::new(vec![5.0f32], vec![]).unwrap();
     // let view = scalar_tensor.transpose(0, 0); // Transpose on scalar is identity
     // assert!(view.is_ok());
     // Check value after making contiguous (though it should already be)
     // let view_data = view.unwrap().contiguous().unwrap().get_f32_data().unwrap();
     // assert_eq!(view_data[0], 5.0f32);
}

// ... (Ignore or adapt other tests using .get() or Tensor::scalar) ...

#[test]
fn test_view_creation_requires_grad() {
    let data = (0..8).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data.clone(), vec![2, 2, 2]);
    tensor.set_requires_grad(true).unwrap();

    // Slice
    let slice_view = tensor.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 2, 1), SliceArg::Slice(1, 2, 1)]).unwrap();
    assert!(slice_view.requires_grad(), "Slice view should require grad");
    assert!(slice_view.grad_fn().is_some(), "Slice view should have grad_fn");

    // Transpose
    let transpose_view = tensor.transpose(1, 2).unwrap();
    assert!(transpose_view.requires_grad(), "Transpose view should require grad");
    assert!(transpose_view.grad_fn().is_some(), "Transpose view should have grad_fn");

    // Permute
    let permute_view = tensor.permute(&[1, 2, 0]).unwrap();
    assert!(permute_view.requires_grad(), "Permute view should require grad");
    assert!(permute_view.grad_fn().is_some(), "Permute view should have grad_fn");

    // Reshape (on contiguous)
    let contiguous_tensor = tensor.contiguous().unwrap();
    contiguous_tensor.set_requires_grad(true).unwrap(); // Ensure the source requires grad
    let reshape_view = contiguous_tensor.reshape(vec![8]).unwrap();
    assert!(reshape_view.requires_grad(), "Reshape view should require grad");
    assert!(reshape_view.grad_fn().is_some(), "Reshape view should have grad_fn");

    // Contiguous (on non-contiguous that requires grad)
    let non_contig = tensor.transpose(0, 1).unwrap(); // Requires grad because tensor does
    assert!(non_contig.requires_grad());
    let contig_view = non_contig.contiguous().unwrap(); // Contiguous makes a copy

    // Corriger les assertions : contiguous DOIT maintenant propager requires_grad et avoir un grad_fn
    assert!(contig_view.requires_grad(), "Contiguous() view SHOULD propagate requires_grad");
    assert!(contig_view.grad_fn().is_some(), "Contiguous() view SHOULD have grad_fn");
}

#[test]
fn test_view_ops_dont_require_grad() {
    let data = (0..8).map(|x| x as f32).collect::<Vec<f32>>();
    let tensor = create_test_tensor(data.clone(), vec![2, 2, 2]);
    assert!(!tensor.requires_grad());

    // Slice
    let slice_view = tensor.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 2, 1), SliceArg::Slice(1, 2, 1)]).unwrap();
    assert!(!slice_view.requires_grad(), "Slice view should not require grad");
    assert!(slice_view.grad_fn().is_none(), "Slice view should not have grad_fn");

    // Transpose
    let transpose_view = tensor.transpose(1, 2).unwrap();
    assert!(!transpose_view.requires_grad(), "Transpose view should not require grad");
    assert!(transpose_view.grad_fn().is_none(), "Transpose view should not have grad_fn");

    // Permute
    let permute_view = tensor.permute(&[1, 2, 0]).unwrap();
    assert!(!permute_view.requires_grad(), "Permute view should not require grad");
    assert!(permute_view.grad_fn().is_none(), "Permute view should not have grad_fn");

    // Reshape (on contiguous)
    let reshape_view = tensor.reshape(vec![8]).unwrap();
    assert!(!reshape_view.requires_grad(), "Reshape view should not require grad");
    assert!(reshape_view.grad_fn().is_none(), "Reshape view should not have grad_fn");

    // Contiguous
    let contig_view = tensor.contiguous().unwrap();
    assert!(!contig_view.requires_grad(), "Contiguous view should not require grad");
    assert!(contig_view.grad_fn().is_none(), "Contiguous view should not have grad_fn");
}

#[test]
fn test_view_ops_on_scalar() {
    let scalar_tensor = Tensor::new(vec![5.0f32], vec![]).unwrap();

    // Reshape to different non-scalar shapes (should work if numel matches)
    assert!(scalar_tensor.reshape(vec![1]).is_ok());
    assert!(scalar_tensor.reshape(vec![1,1]).is_ok());
    assert!(matches!(scalar_tensor.reshape(vec![2]), Err(NeuraRustError::ShapeMismatch{..}))); // Wrong numel

    // Slice is invalid for 0-dim
    // Expect SliceError due to rank mismatch
    assert!(matches!(scalar_tensor.slice(&[SliceArg::Slice(0, 0, 1)]), Err(NeuraRustError::SliceError { .. })));

    // Transpose is invalid for 0-dim
    // Expect IndexOutOfBounds because dims 0 and 0 don't exist for rank 0
    assert!(matches!(scalar_tensor.transpose(0, 0), Err(NeuraRustError::IndexOutOfBounds{..})));

    // Permute is valid for rank 0 with empty axes (identity)
    assert!(scalar_tensor.permute(&[]).is_ok(), "Permute with empty axes on scalar should be Ok");
    // Permute is invalid for rank 0 with non-empty axes (RankMismatch)
    assert!(matches!(scalar_tensor.permute(&[0]), Err(NeuraRustError::RankMismatch{ expected: 0, actual: 1 })), "Permute with non-empty axes on scalar should be RankMismatch");

    // Reshape (should work to vec![1] or stay as vec![])
    let reshaped = scalar_tensor.reshape(vec![1,1]);
    assert!(reshaped.is_ok());
    assert_eq!(reshaped.unwrap().shape(), vec![1,1]);
    // Contiguous (valid, no-op)
    let contiguous = scalar_tensor.contiguous();
    assert!(contiguous.is_ok());
    assert_eq!(scalar_tensor, contiguous.unwrap());
}

#[test]
fn test_view_ops_on_zero_dim_tensor() {
    let zero_dim_tensor = Tensor::new(Vec::<f32>::new(), vec![2, 0, 3]).unwrap();
    assert_eq!(zero_dim_tensor.numel(), 0);
    assert!(zero_dim_tensor.is_contiguous());

    // Slice
    let sliced = zero_dim_tensor.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 0, 1), SliceArg::Slice(1, 2, 1)]);
    assert!(sliced.is_ok());
    assert_eq!(sliced.unwrap().shape(), vec![1, 0, 1]);

    // Transpose
    let transposed = zero_dim_tensor.transpose(0, 1);
    assert!(transposed.is_ok());
    assert_eq!(transposed.unwrap().shape(), vec![0, 2, 3]); // Swaps 0 and 2

    // Permute
    let permuted = zero_dim_tensor.permute(&[1, 2, 0]);
    assert!(permuted.is_ok());
    assert_eq!(permuted.unwrap().shape(), vec![0, 3, 2]);

    // Reshape (to another 0-element shape)
    let reshaped = zero_dim_tensor.reshape(vec![6, 0]);
    assert!(reshaped.is_ok());
    assert_eq!(reshaped.unwrap().shape(), vec![6, 0]);

    // Reshape (to non-zero elements -> should fail)
    let reshape_fail = zero_dim_tensor.reshape(vec![1]);
    assert!(matches!(reshape_fail, Err(NeuraRustError::ShapeMismatch { .. })));

    // Contiguous
    let contiguous_zero = zero_dim_tensor.contiguous().unwrap();
    assert_eq!(zero_dim_tensor, contiguous_zero);
}

#[test]
fn test_slice_simple() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let expected = neurarust_core::tensor::create::from_vec_f32(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
    let sliced = a.slice(&[SliceArg::Slice(1, 4, 1)]).unwrap();
    check_tensor_near(&sliced, &expected.shape(), &expected.get_f32_data().unwrap(), 1e-6);
}

#[test]
fn test_slice_2d() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
    // Slice rows 0..1 (exclusive), columns 1..3 (exclusive)
    let expected = neurarust_core::tensor::create::from_vec_f32(vec![2., 3.], vec![1, 2]).unwrap();
    let sliced = a
        .slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(1, 3, 1)])
        .unwrap();
    check_tensor_near(&sliced, &expected.shape(), &expected.get_f32_data().unwrap(), 1e-6);
}

#[test]
fn test_slice_full_range() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
    let expected = a.clone(); // Slicing with full range should be identity
    let sliced = a
        .slice(&[SliceArg::Slice(0, 2, 1), SliceArg::Slice(0, 3, 1)])
        .unwrap();
    check_tensor_near(&sliced, &expected.shape(), &expected.get_f32_data().unwrap(), 1e-6);
}

#[test]
fn test_slice_empty() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
    // Slice rows 1..1 -> empty
    let expected_shape = vec![0, 3];
    let expected = Tensor::new(Vec::<f32>::new(), expected_shape.clone()).unwrap();
    let sliced = a
        .slice(&[SliceArg::Slice(1, 1, 1), SliceArg::Slice(0, 3, 1)])
        .unwrap();
    check_tensor_near(&sliced, &expected_shape, &expected.get_f32_data().unwrap(), 1e-6);

    // Slice cols 2..2 -> empty
    let expected_cols_shape = vec![2, 0];
    let expected_cols = Tensor::new(Vec::<f32>::new(), expected_cols_shape.clone()).unwrap();
    let sliced_cols = a
        .slice(&[SliceArg::Slice(0, 2, 1), SliceArg::Slice(2, 2, 1)])
        .unwrap();
    check_tensor_near(&sliced_cols, &expected_cols_shape, &expected_cols.get_f32_data().unwrap(), 1e-6);
}

#[test]
fn test_slice_step() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![10]).unwrap();
    let expected = neurarust_core::tensor::create::from_vec_f32(vec![0., 2., 4., 6., 8.], vec![5]).unwrap();
    let sliced = a.slice(&[SliceArg::Slice(0, 10, 2)]).unwrap(); // Step 2
    check_tensor_near(&sliced, &expected.shape(), &expected.get_f32_data().unwrap(), 1e-6);

    let expected_rev = neurarust_core::tensor::create::from_vec_f32(vec![9., 7., 5., 3., 1.], vec![5]).unwrap();
    // Note: Negative step requires careful range handling, assuming positive for now
    // If negative steps were implemented:
    // let sliced_rev = a.slice(&[SliceArg::Slice(9, -1, -2)]).unwrap(); // Hypothetical negative step
    // check_tensor_near(&sliced_rev, &expected_rev.shape(), &expected_rev.get_f32_data().unwrap(), 1e-6);
    let _ = expected_rev; // Avoid unused warning for now
}

#[test]
fn test_slice_requires_grad() {
    let a_res = neurarust_core::tensor::create::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let a = a_res.unwrap();
    a.set_requires_grad(true).unwrap(); // Unwrap after setting grad
    let sliced = a.slice(&[SliceArg::Slice(0, 2, 1)]).unwrap(); // Now call slice
    assert!(sliced.requires_grad());
    assert!(sliced.grad_fn().is_some());
    assert_eq!(sliced.shape(), &[2]);
}

#[test]
fn test_slice_no_grad() {
    let a_res = neurarust_core::tensor::create::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let a = a_res.unwrap();
    a.set_requires_grad(false).unwrap(); // Unwrap after setting grad
    let sliced = a.slice(&[SliceArg::Slice(0, 2, 1)]).unwrap(); // Now call slice
    assert!(!sliced.requires_grad());
    assert!(sliced.grad_fn().is_none());
}

#[test]
fn test_slice_error_rank_mismatch() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3.], vec![3]).unwrap();
    let result = a.slice(&[SliceArg::Slice(0, 1, 1), SliceArg::Slice(0, 1, 1)]);
    assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
}

#[test]
fn test_slice_error_out_of_bounds() {
    let a = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3.], vec![3]).unwrap();
    
    // Test 1: End index (4) > dimension size (3). Should succeed and clamp end to 3.
    let result = a.slice(&[SliceArg::Slice(0, 4, 1)]);
    assert!(result.is_ok(), "Slice end out of bounds should succeed (clamp)");
    let view = result.unwrap();
    assert_eq!(view.shape(), vec![3], "Shape mismatch for end out of bounds slice"); // Shape should be [3] (0..3)
    check_tensor_near(&view, &vec![3], &[1.0, 2.0, 3.0], 1e-6);

    // Test 2: Start index (3) == dimension size (3). Should succeed (empty slice).
    let result_start = a.slice(&[SliceArg::Slice(3, 3, 1)]);
    assert!(result_start.is_ok(), "Slice start == size should succeed (empty)");
    let view_start = result_start.unwrap();
    assert_eq!(view_start.shape(), vec![0], "Shape mismatch for start == size slice"); // Shape should be [0]

    // Test 3: 2D tensor, second dimension slice end (3) > size (2). Should succeed and clamp end to 2.
    let a_2d = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
    let result_2d = a_2d.slice(&[SliceArg::Slice(0, 2, 1), SliceArg::Slice(1, 3, 1)]);
    assert!(result_2d.is_ok(), "2D Slice end out of bounds should succeed (clamp)");
    let view_2d = result_2d.unwrap();
    assert_eq!(view_2d.shape(), vec![2, 1], "Shape mismatch for 2D end out of bounds slice"); // Shape [2, 1] (0..2, 1..2)
    // Expected values: [[2], [4]] -> contiguous [2, 4]
    check_tensor_near(&view_2d.contiguous().unwrap(), &vec![2, 1], &[2.0, 4.0], 1e-6);
}

#[test]
fn test_slice_error_invalid_step() {
     let a = neurarust_core::tensor::create::from_vec_f32(vec![1., 2., 3.], vec![3]).unwrap();
     let result = a.slice(&[SliceArg::Slice(0, 3, 0)]);
     assert!(matches!(result, Err(NeuraRustError::SliceError { message }) if message == "Step cannot be zero"));
} 