#[test]
fn test_slice_basic() {
    let t = crate::tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]).unwrap();
    let sliced = t.slice(&[0..1, 0..2, 1..3]).unwrap(); // Slice to [1, 2, 2]
    assert_eq!(sliced.shape(), &[1, 2, 2]);
}

#[test]
fn test_slice_full() {
    let t = crate::tensor::from_vec_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]).unwrap();
    let sliced = t.slice(&[0..2, 0..3]).unwrap(); // Full slice
    assert_eq!(sliced.shape(), &[2, 3]);
}

#[test]
fn test_slice_empty_dim() {
    let t = crate::tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]).unwrap();
    let sliced = t.slice(&[1..1, 0..2, 0..3]).unwrap(); // Slice dim 0 to be empty
    assert_eq!(sliced.shape(), &[0, 2, 3]);
}

#[test]
fn test_slice_invalid_range_start_gt_end() {
    let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = t.slice(&[0..2, 1..0]); // Invalid range 1..0
    assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
}

#[test]
fn test_slice_invalid_range_end_gt_size() {
    let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = t.slice(&[0..3, 0..2]); // Invalid range 0..3 for dim 0 size 2
    assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
} 