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

#[test]
fn test_unsqueeze_basic() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    // Original shape: [2, 3], Original strides: [3, 1]

    // Unsqueeze at dim 0
    let u0 = t.unsqueeze(0)?;
    assert_eq!(u0.shape(), vec![1, 2, 3]);
    assert_eq!(u0.strides(), vec![3, 3, 1]); // Stride[0]=input_strides[0]=3. input_strides=[3,1] copied after new stride.
    assert_eq!(u0.get_f32_data()?, t.get_f32_data()?);

    // Unsqueeze at dim 1
    let u1 = t.unsqueeze(1)?;
    assert_eq!(u1.shape(), vec![2, 1, 3]);
    assert_eq!(u1.strides(), vec![3, 1, 1]); // Stride[0]=input_strides[0]=3. Stride[1]=input_strides[1]=1. input_strides[1] copied after new stride.
    assert_eq!(u1.get_f32_data()?, t.get_f32_data()?);

    // Unsqueeze at dim 2 (end)
    let u2 = t.unsqueeze(2)?;
    assert_eq!(u2.shape(), vec![2, 3, 1]);
    assert_eq!(u2.strides(), vec![3, 1, 1]); // Stride[0]=input_strides[0]=3. Stride[1]=input_strides[1]=1. Stride[2]=1.
    assert_eq!(u2.get_f32_data()?, t.get_f32_data()?);

    Ok(())
}

#[test]
fn test_unsqueeze_scalar() -> Result<(), NeuraRustError> {
    let scalar = Tensor::new(vec![42.0f32], vec![])?;
    // Strides: []

    let u0 = scalar.unsqueeze(0)?;
    assert_eq!(u0.shape(), vec![1]);
    assert_eq!(u0.strides(), vec![1]);
    assert_eq!(u0.item_f32()?, 42.0f32);
    Ok(())
}

#[test]
fn test_unsqueeze_already_1d() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    // Strides: [1]

    let u0 = t.unsqueeze(0)?;
    assert_eq!(u0.shape(), vec![1, 3]);
    assert_eq!(u0.strides(), vec![1, 1]); // Stride[0]=input_strides[0]=1. input_strides[0] copied after new stride.
    assert_eq!(u0.get_f32_data()?, vec![1.0, 2.0, 3.0]);

    let u1 = t.unsqueeze(1)?;
    assert_eq!(u1.shape(), vec![3, 1]);
    assert_eq!(u1.strides(), vec![1, 1]); // Stride[0]=input_strides[0]=1. Stride[1]=1.
    assert_eq!(u1.get_f32_data()?, vec![1.0, 2.0, 3.0]);

    Ok(())
}

#[test]
fn test_unsqueeze_error_invalid_dim() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
    // Rank is 2, so valid dim for unsqueeze is 0, 1, 2.
    assert!(t.unsqueeze(3).is_err());
    match t.unsqueeze(3) {
        Err(NeuraRustError::InvalidAxis { axis, rank }) => {
            assert_eq!(axis, 3);
            assert_eq!(rank, 2); // The error rank should be original rank for dim > rank check
        }
        _ => panic!("Expected InvalidAxis error"),
    }
    Ok(())
}

#[test]
fn test_unsqueeze_propagates_requires_grad() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0f32, 2.0], vec![2])?;
    t.set_requires_grad(true)?;

    let u = t.unsqueeze(0)?;
    assert!(u.requires_grad());
    assert!(u.grad_fn().is_some());
    let grad_fn_name = format!("{:?}", u.grad_fn().unwrap());
    assert!(grad_fn_name.contains("UnsqueezeBackward"));

    Ok(())
}

#[test]
fn test_unsqueeze_data_integrity_multiple_unsqueezes() -> Result<(), NeuraRustError> {
    let t_orig = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3])?;
    let data_orig = t_orig.get_f32_data()?;

    let u0 = t_orig.unsqueeze(0)?; // [1, 2, 3]
    assert_eq!(u0.get_f32_data()?, data_orig);

    let u01 = u0.unsqueeze(1)?; // Shape: [1,1,2,3]. Strides u0: [3,3,1]. Strides u01: [3,3,3,1]
    assert_eq!(u01.get_f32_data()?, data_orig);

    let u013 = u01.unsqueeze(3)?; // Shape: [1,1,2,1,3]. Strides u01: [3,3,3,1]. Strides u013: [3,3,3,1,1]
    assert_eq!(u013.get_f32_data()?, data_orig);

    assert_eq!(u013.shape(), vec![1, 1, 2, 1, 3]);

    Ok(())
}

#[test]
fn test_unsqueeze_strides_complex() -> Result<(), NeuraRustError> {
    // Tensor: shape [2, 3, 4], strides [12, 4, 1]
    let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
    assert_eq!(t.strides(), vec![12, 4, 1]);

    // Unsqueeze at dim 0 -> shape [1, 2, 3, 4], expected strides [12, 12, 4, 1]
    let u0 = t.unsqueeze(0)?;
    assert_eq!(u0.shape(), vec![1, 2, 3, 4]);
    assert_eq!(u0.strides(), vec![12, 12, 4, 1]);

    // Unsqueeze at dim 1 -> shape [2, 1, 3, 4], expected strides [12, 4, 4, 1]
    let u1 = t.unsqueeze(1)?;
    assert_eq!(u1.shape(), vec![2, 1, 3, 4]);
    assert_eq!(u1.strides(), vec![12, 4, 4, 1]);

    // Unsqueeze at dim 2 -> shape [2, 3, 1, 4], expected strides [12, 4, 1, 1]
    let u2 = t.unsqueeze(2)?;
    assert_eq!(u2.shape(), vec![2, 3, 1, 4]);
    assert_eq!(u2.strides(), vec![12, 4, 1, 1]);

    // Unsqueeze at dim 3 (end) -> shape [2, 3, 4, 1], expected strides [12, 4, 1, 1]
    let u3 = t.unsqueeze(3)?;
    assert_eq!(u3.shape(), vec![2, 3, 4, 1]);
    assert_eq!(u3.strides(), vec![12, 4, 1, 1]);

    Ok(())
}

#[test]
fn test_squeeze_basic_none() -> Result<(), NeuraRustError> {
    // Shape [1, 2, 1, 3, 1]
    let t = Tensor::new((0..6).map(|x| x as f32).collect(), vec![1, 2, 1, 3, 1])?;
    let s = t.squeeze(None)?;
    assert_eq!(s.shape(), vec![2, 3]);
    // Strides: originaux [6, 3, 3, 1, 1]. Squeezés: [3,1] (ceux de dim 2 et 3)
    assert_eq!(s.strides(), vec![3, 1]);
    assert_eq!(s.get_f32_data()?, t.get_f32_data()?);

    // Shape [2, 3]
    let t2 = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3])?;
    let s2 = t2.squeeze(None)?;
    assert_eq!(s2.shape(), vec![2, 3]); // No change
    assert_eq!(s2.strides(), vec![3, 1]);
    assert_eq!(s2.get_f32_data()?, t2.get_f32_data()?);
    Ok(())
}

#[test]
fn test_squeeze_specific_dim() -> Result<(), NeuraRustError> {
    let t = Tensor::new((0..6).map(|x| x as f32).collect(), vec![1, 2, 1, 3, 1])?;
    // Original shape: [1, 2, 1, 3, 1], Strides: [6, 3, 3, 1, 1]

    // Squeeze dim 0 (size 1)
    let s0 = t.squeeze(Some(0))?;
    assert_eq!(s0.shape(), vec![2, 1, 3, 1]);
    assert_eq!(s0.strides(), vec![3, 3, 1, 1]);
    assert_eq!(s0.get_f32_data()?, t.get_f32_data()?);

    // Squeeze dim 2 (original index, size 1)
    let s2 = t.squeeze(Some(2))?;
    assert_eq!(s2.shape(), vec![1, 2, 3, 1]);
    assert_eq!(s2.strides(), vec![6, 3, 1, 1]); // Stride de dim 1 ([6]), dim 2 ([3]), dim 4 ([1]), dim 5 ([1])
                                            // Strides originaux: [6, 3, 3, 1, 1]
                                            // Squeezing dim 2 (original stride 3) -> [6, 3, 1, 1]
    assert_eq!(s2.get_f32_data()?, t.get_f32_data()?);

    // Attempt to squeeze dim 1 (size 2) - should be no-op
    let s1_noop = t.squeeze(Some(1))?;
    assert_eq!(s1_noop.shape(), vec![1, 2, 1, 3, 1]);
    assert_eq!(s1_noop.strides(), vec![6, 3, 3, 1, 1]);
    assert_eq!(s1_noop.get_f32_data()?, t.get_f32_data()?);
    Ok(())
}

#[test]
fn test_squeeze_to_scalar() -> Result<(), NeuraRustError> {
    let t1 = Tensor::new(vec![5.0f32], vec![1, 1, 1, 1])?;
    let s1 = t1.squeeze(None)?;
    assert_eq!(s1.shape(), vec![]);
    assert_eq!(s1.strides(), vec![]);
    assert_eq!(s1.item_f32()?, 5.0f32);

    let t2 = Tensor::new(vec![6.0f32], vec![1])?;
    let s2 = t2.squeeze(None)?;
    assert_eq!(s2.shape(), vec![]);
    assert_eq!(s2.strides(), vec![]);
    assert_eq!(s2.item_f32()?, 6.0f32);

    let t3 = Tensor::new(vec![7.0f32], vec![])?;
    let s3 = t3.squeeze(None)?;
    assert_eq!(s3.shape(), vec![]); // Scalar remains scalar
    assert_eq!(s3.strides(), vec![]);
    assert_eq!(s3.item_f32()?, 7.0f32);
    Ok(())
}

#[test]
fn test_squeeze_error_invalid_dim() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0f32], vec![1, 1])?;
    // Rank is 2, so valid dim for squeeze is 0, 1
    assert!(t.squeeze(Some(2)).is_err());
    match t.squeeze(Some(2)) {
        Err(NeuraRustError::InvalidAxis { axis, rank }) => {
            assert_eq!(axis, 2);
            assert_eq!(rank, 2);
        }
        _ => panic!("Expected InvalidAxis error"),
    }
    Ok(())
}

#[test]
fn test_squeeze_propagates_requires_grad() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0f32, 2.0], vec![1, 2, 1])?;
    t.set_requires_grad(true)?;

    let s_none = t.squeeze(None)?;
    assert!(s_none.requires_grad());
    assert!(s_none.grad_fn().is_some());
    let grad_fn_name_none = format!("{:?}", s_none.grad_fn().unwrap());
    assert!(grad_fn_name_none.contains("SqueezeBackward"));

    let s_some = t.squeeze(Some(0))?;
    assert!(s_some.requires_grad());
    assert!(s_some.grad_fn().is_some());
    let grad_fn_name_some = format!("{:?}", s_some.grad_fn().unwrap());
    assert!(grad_fn_name_some.contains("SqueezeBackward"));

    // No-op squeeze should still propagate grad info if input required grad
    let t_no_squeeze_needed = Tensor::new(vec![1.,2.,3.,4.], vec![2,2])?;
    t_no_squeeze_needed.set_requires_grad(true)?;
    let s_noop = t_no_squeeze_needed.squeeze(Some(0))?; // dim 0 is size 2
    assert!(s_noop.requires_grad());
    assert!(s_noop.grad_fn().is_some()); // Even if it's a no-op view, it should track
    Ok(())
} 