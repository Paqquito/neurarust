#[cfg(test)]
// Les tests sont directement dans ce fichier, pas besoin de `mod tests { ... }` ici.

use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use std::sync::Arc;

#[test]
fn test_flatten_basic() -> Result<(), NeuraRustError> {
    let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
    let flattened_t = t.flatten(1, 2)?;
    assert_eq!(flattened_t.shape(), &[2, 12]);
    assert_eq!(flattened_t.strides(), &[12, 1]);
    assert_eq!(flattened_t.numel(), 24);
    let t_lock = t.read_data();
    let ft_lock = flattened_t.read_data();
    assert!(Arc::ptr_eq(t_lock.buffer(), ft_lock.buffer()));
    Ok(())
}

#[test]
fn test_flatten_all() -> Result<(), NeuraRustError> {
    let t = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3])?;
    let flattened_t = t.flatten(0, 1)?;
    assert_eq!(flattened_t.shape(), &[6]);
    assert_eq!(flattened_t.strides(), &[1]);
    let t_lock = t.read_data();
    let ft_lock = flattened_t.read_data();
    assert!(Arc::ptr_eq(t_lock.buffer(), ft_lock.buffer()));
    Ok(())
}

#[test]
fn test_flatten_scalar() -> Result<(), NeuraRustError> {
    let scalar = Tensor::new(vec![42.0f32], vec![])?;
    let flattened_scalar = scalar.flatten(0, 0)?;
    assert_eq!(flattened_scalar.shape(), &[1]);
    assert_eq!(flattened_scalar.strides(), &[1]);
    assert_eq!(flattened_scalar.item_f32()?, 42.0f32);

    let flattened_scalar_neg_idx = scalar.flatten(-1, -1)?;
    assert_eq!(flattened_scalar_neg_idx.shape(), &[1]);
    assert_eq!(flattened_scalar_neg_idx.strides(), &[1]);
    Ok(())
}

#[test]
fn test_flatten_start_eq_end() -> Result<(), NeuraRustError> {
    let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
    let flattened_t = t.flatten(1, 1)?;
    assert_eq!(flattened_t.shape(), &[2, 3, 4]);
    let t_lock = t.read_data();
    let ft_lock = flattened_t.read_data();
    assert!(Arc::ptr_eq(t_lock.buffer(), ft_lock.buffer()));
    Ok(())
}

#[test]
fn test_flatten_negative_indices() -> Result<(), NeuraRustError> {
    let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
    let flattened_t = t.flatten(1, -1)?;
    assert_eq!(flattened_t.shape(), &[2, 12]);
    let t_lock = t.read_data();
    let ft_lock = flattened_t.read_data();
    assert!(Arc::ptr_eq(t_lock.buffer(), ft_lock.buffer()));

    let flattened_all = t.flatten(0, -1)?;
    assert_eq!(flattened_all.shape(), &[24]);
    let t_all_lock = t.read_data();
    let ft_all_lock = flattened_all.read_data();
    assert!(Arc::ptr_eq(t_all_lock.buffer(), ft_all_lock.buffer()));
    Ok(())
}

#[test]
fn test_flatten_dim_with_zero() -> Result<(), NeuraRustError> {
    let t = Tensor::new(Vec::<f32>::new(), vec![2, 0, 3])?;
    let flattened_t = t.flatten(0, 1)?;
    assert_eq!(flattened_t.shape(), &[0, 3]);
    assert_eq!(flattened_t.numel(), 0);

    let flattened_all = t.flatten(0, -1)?;
    assert_eq!(flattened_all.shape(), &[0]);
    Ok(())
}

#[test]
fn test_flatten_error_start_gt_end() {
    let t2 = Tensor::new(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![2,3]).unwrap();
    let result2 = t2.flatten(1,0);
    assert!(matches!(result2, Err(NeuraRustError::UnsupportedOperation(_))));
}

#[test]
fn test_flatten_error_out_of_bounds() {
    let t = Tensor::new(vec![1.0f32], vec![1]).unwrap();
    assert!(matches!(t.flatten(1, 1), Err(NeuraRustError::InvalidAxis { axis: 1, rank: 1 }) | Err(NeuraRustError::UnsupportedOperation(_)) ));
    assert!(matches!(t.flatten(0, 1), Err(NeuraRustError::InvalidAxis { axis: 1, rank: 1 }) | Err(NeuraRustError::UnsupportedOperation(_)) ));
    assert!(matches!(t.flatten(-2, 0), Err(NeuraRustError::UnsupportedOperation(_))));
    
    let scalar = Tensor::new(vec![1.0f32], vec![]).unwrap();
    assert!(matches!(scalar.flatten(1,1), Err(NeuraRustError::UnsupportedOperation(_))));
    assert!(matches!(scalar.flatten(0,1), Err(NeuraRustError::UnsupportedOperation(_))));
    assert!(matches!(scalar.flatten(-2,0), Err(NeuraRustError::UnsupportedOperation(_))));
}

#[test]
fn test_flatten_non_contiguous() -> Result<(), NeuraRustError> {
    let t = Tensor::new((0..12).map(|x| x as f32).collect(), vec![3, 4])?;
    let transposed_t = t.transpose(0, 1)?;
    assert!(!transposed_t.is_contiguous());

    let result = transposed_t.flatten(0, 1);
    assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(_))));
    Ok(())
}

#[test]
fn test_flatten_propagates_requires_grad() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?;
    t.set_requires_grad(true)?;
    let flattened_t = t.flatten(0, 0)?;
    assert!(flattened_t.requires_grad());
    Ok(())
} 