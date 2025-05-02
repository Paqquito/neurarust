// These tests were originally in src/tensor/utils.rs or src/tensor/tests.rs
// They test utility functions related to tensors, now moved to integration tests.

use neurarust_core::tensor::utils::{broadcast_shapes, calculate_strides};
use neurarust_core::error::NeuraRustError;

#[test]
fn test_broadcast_shapes_equal() {
    let shape1 = vec![2, 3];
    let shape2 = vec![2, 3];
    let expected = vec![2, 3];
    assert_eq!(broadcast_shapes(&shape1, &shape2).unwrap(), expected);
}

#[test]
fn test_broadcast_shapes_scalar() {
    let shape1 = vec![2, 3];
    let shape2 = vec![]; // Scalar
    let expected = vec![2, 3];
    assert_eq!(broadcast_shapes(&shape1, &shape2).unwrap(), expected);
    assert_eq!(broadcast_shapes(&shape2, &shape1).unwrap(), expected);
}

#[test]
fn test_broadcast_shapes_one_dimension() {
    let shape1 = vec![2, 3];
    let shape2 = vec![3];
    let expected = vec![2, 3];
    assert_eq!(broadcast_shapes(&shape1, &shape2).unwrap(), expected);
    assert_eq!(broadcast_shapes(&shape2, &shape1).unwrap(), expected);
}

#[test]
fn test_broadcast_shapes_prepend_ones() {
    let shape1 = vec![3, 4];
    let shape2 = vec![1, 3, 4];
    let expected = vec![1, 3, 4];
    assert_eq!(broadcast_shapes(&shape1, &shape2).unwrap(), expected);
    assert_eq!(broadcast_shapes(&shape2, &shape1).unwrap(), expected);
}

#[test]
fn test_broadcast_shapes_complex() {
    let shape1 = vec![5, 1, 6];
    let shape2 = vec![   4, 1];
    let expected = vec![5, 4, 6];
    assert_eq!(broadcast_shapes(&shape1, &shape2).unwrap(), expected);
    assert_eq!(broadcast_shapes(&shape2, &shape1).unwrap(), expected);
}

#[test]
fn test_broadcast_shapes_with_zero() {
    let shape1 = vec![5, 0, 6];
    let shape2 = vec![   1, 1];
    let expected = vec![5, 0, 6];
    assert_eq!(broadcast_shapes(&shape1, &shape2).unwrap(), expected);
    assert_eq!(broadcast_shapes(&shape2, &shape1).unwrap(), expected);

    let shape3 = vec![5, 1, 6];
    let shape4 = vec![   0, 1];
    let expected2 = vec![5, 0, 6];
     assert_eq!(broadcast_shapes(&shape3, &shape4).unwrap(), expected2);
     assert_eq!(broadcast_shapes(&shape4, &shape3).unwrap(), expected2);
}

// Example of a broadcast failure test
#[test]
fn test_broadcast_failure() {
    let shape1 = vec![2, 3];
    let shape2 = vec![2, 4];
    let result = broadcast_shapes(&shape1, &shape2);
    assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
}

#[test]
fn test_calculate_strides_simple() {
    let shape = vec![2, 3, 4];
    let expected = vec![12, 4, 1];
    assert_eq!(calculate_strides(&shape), expected);
}

#[test]
fn test_calculate_strides_empty() {
    let shape = vec![];
    let expected = vec![];
    assert_eq!(calculate_strides(&shape), expected);
}

#[test]
fn test_calculate_strides_includes_zero_dim() {
    let shape = vec![2, 0, 3];
    // Stride calculation usually proceeds normally, even with zero dimensions.
    // The last dim stride is 1.
    // Dim 1 stride = stride[2] * shape[2] = 1 * 3 = 3.
    // Dim 0 stride = stride[1] * shape[1] = 3 * 0 = 0.
    let expected = vec![0, 3, 1];
    assert_eq!(calculate_strides(&shape), expected);
}

#[test]
fn test_calculate_strides_single_zero_dim() {
     let shape = vec![0];
     // Stride for the last (and only) dimension is 1.
     let expected = vec![1];
     assert_eq!(calculate_strides(&shape), expected);
} 