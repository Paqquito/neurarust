#[cfg(test)]
// Fichier de test pour src/tensor/utils.rs

use super::*; // Importe les fonctions du module parent (calculate_strides, broadcast_shapes)
use crate::error::NeuraRustError; // Import de l'erreur pour les tests de broadcast

#[test]
fn test_calculate_strides_simple() {
    assert_eq!(calculate_strides(&[2, 3]), vec![3, 1]);
    assert_eq!(calculate_strides(&[4, 5, 6]), vec![30, 6, 1]);
    assert_eq!(calculate_strides(&[5]), vec![1]);
    assert_eq!(calculate_strides(&[1, 5]), vec![5, 1]);
    assert_eq!(calculate_strides(&[5, 1]), vec![1, 1]);
}

#[test]
fn test_calculate_strides_empty() {
    assert_eq!(calculate_strides(&[]), Vec::<usize>::new());
}

#[test]
fn test_calculate_strides_single_zero_dim() {
    assert_eq!(calculate_strides(&[0]), vec![1]);
}

#[test]
fn test_calculate_strides_includes_zero_dim() {
    assert_eq!(calculate_strides(&[2, 0, 3]), vec![0, 3, 1]);
    assert_eq!(calculate_strides(&[2, 3, 0]), vec![0, 0, 1]);
}

#[test]
fn test_broadcast_shapes_equal() {
    assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]), Ok(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[5], &[5]), Ok(vec![5]));
    assert_eq!(broadcast_shapes(&[], &[]), Ok(vec![]));
}

#[test]
fn test_broadcast_shapes_scalar() {
    assert_eq!(broadcast_shapes(&[2, 3], &[]), Ok(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[], &[2, 3]), Ok(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[1], &[]), Ok(vec![1]));
}

#[test]
fn test_broadcast_shapes_one_dimension() {
    assert_eq!(broadcast_shapes(&[4, 1], &[4, 5]), Ok(vec![4, 5]));
    assert_eq!(broadcast_shapes(&[4, 5], &[1, 5]), Ok(vec![4, 5]));
    assert_eq!(broadcast_shapes(&[4, 5], &[4, 1]), Ok(vec![4, 5]));
    assert_eq!(broadcast_shapes(&[1, 5], &[4, 5]), Ok(vec![4, 5]));
}

#[test]
fn test_broadcast_shapes_prepend_ones() {
    assert_eq!(broadcast_shapes(&[4, 5], &[5]), Ok(vec![4, 5]));
    assert_eq!(broadcast_shapes(&[5], &[4, 5]), Ok(vec![4, 5]));
    assert_eq!(broadcast_shapes(&[2, 3, 4], &[3, 1]), Ok(vec![2, 3, 4]));
    assert_eq!(broadcast_shapes(&[3, 4], &[2, 1, 4]), Ok(vec![2, 3, 4]));
}

#[test]
fn test_broadcast_shapes_complex() {
    assert_eq!(
        broadcast_shapes(&[5, 1, 4, 1], &[4, 5]),
        Ok(vec![5, 1, 4, 5])
    );
    assert_eq!(
        broadcast_shapes(&[4, 5], &[5, 1, 4, 1]),
        Ok(vec![5, 1, 4, 5])
    );
}

#[test]
fn test_broadcast_shapes_with_zero() {
    assert_eq!(broadcast_shapes(&[2, 3], &[2, 0]), Ok(vec![2, 0]));
    assert_eq!(broadcast_shapes(&[2, 0], &[2, 3]), Ok(vec![2, 0]));
    assert_eq!(broadcast_shapes(&[1, 0], &[5, 1]), Ok(vec![5, 0]));
    assert_eq!(broadcast_shapes(&[5, 1], &[1, 0]), Ok(vec![5, 0]));
    assert_eq!(broadcast_shapes(&[5, 0], &[5, 0]), Ok(vec![5, 0]));
    assert_eq!(broadcast_shapes(&[0], &[5]), Ok(vec![0]));
    assert_eq!(broadcast_shapes(&[5], &[0]), Ok(vec![0]));
    assert_eq!(broadcast_shapes(&[], &[0]), Ok(vec![0]));
    assert_eq!(broadcast_shapes(&[1, 0], &[5, 3]), Ok(vec![5, 0]));
    let res1 = broadcast_shapes(&[2, 3], &[2, 4]);
    assert!(matches!(res1, Err(NeuraRustError::BroadcastError { .. })));
    let res2 = broadcast_shapes(&[3, 0], &[2, 0]);
    assert!(matches!(res2, Err(NeuraRustError::BroadcastError { .. })));
    let res3 = broadcast_shapes(&[2, 0], &[5, 3]);
    assert!(matches!(res3, Err(NeuraRustError::BroadcastError { .. })));
}

// Tests for index_to_coord
#[test]
fn test_index_to_coord_simple() {
    assert_eq!(index_to_coord(0, &[2, 3]), vec![0, 0]);
    assert_eq!(index_to_coord(1, &[2, 3]), vec![0, 1]);
    assert_eq!(index_to_coord(2, &[2, 3]), vec![0, 2]);
    assert_eq!(index_to_coord(3, &[2, 3]), vec![1, 0]);
    assert_eq!(index_to_coord(4, &[2, 3]), vec![1, 1]);
    assert_eq!(index_to_coord(5, &[2, 3]), vec![1, 2]);
}

#[test]
fn test_index_to_coord_higher_dim() {
    assert_eq!(index_to_coord(0, &[2, 2, 2]), vec![0, 0, 0]);
    assert_eq!(index_to_coord(3, &[2, 2, 2]), vec![0, 1, 1]);
    assert_eq!(index_to_coord(7, &[2, 2, 2]), vec![1, 1, 1]);
}

#[test]
fn test_index_to_coord_scalar() {
    assert_eq!(index_to_coord(0, &[]), vec![]);
}

#[test]
#[should_panic]
fn test_index_to_coord_scalar_panic() {
    index_to_coord(1, &[]); // Index > 0 for scalar
}

#[test]
#[should_panic]
fn test_index_to_coord_out_of_bounds() {
    index_to_coord(6, &[2, 3]); // Index >= numel
}

#[test]
fn test_index_to_coord_with_zero_dim() {
    assert_eq!(index_to_coord(0, &[2, 0, 3]), vec![0, 0, 0]); // Index 0 is valid for empty tensor
}

#[test]
#[should_panic]
fn test_index_to_coord_with_zero_dim_panic() {
    index_to_coord(1, &[2, 0, 3]); // Index > 0 for empty tensor
}

#[test]
fn test_coord_to_index_basic() {
    let shape = &[2, 3, 4];
    assert_eq!(coord_to_index(&[0, 0, 0], shape), 0);
    assert_eq!(coord_to_index(&[0, 1, 0], shape), 4);
    assert_eq!(coord_to_index(&[0, 2, 3], shape), 11);
    assert_eq!(coord_to_index(&[1, 0, 0], shape), 12);
    assert_eq!(coord_to_index(&[1, 2, 3], shape), 23);
}

#[test]
fn test_coord_to_index_scalar() {
    assert_eq!(coord_to_index(&[], &[]), 0);
}

#[test]
#[should_panic(expected = "Coordinate 2 out of bounds for dimension 0 with size 2")]
fn test_coord_to_index_out_of_bounds() {
    let shape = &[2, 3];
    coord_to_index(&[2, 0], shape);
}

#[test]
#[should_panic(expected = "Number of coordinates must match shape rank")]
fn test_coord_to_index_rank_mismatch() {
    let shape = &[2, 3];
    coord_to_index(&[0], shape);
} 