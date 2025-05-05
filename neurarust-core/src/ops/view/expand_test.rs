#[test]
fn test_expand_basic() {
    let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let expanded = t.expand(&[2, 3]).unwrap();
    assert_eq!(expanded.shape(), &[2, 3]);
}

#[test]
fn test_expand_add_dim() {
    let t = crate::tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
    let expanded = t.expand(&[3, 1, 2]).unwrap();
    assert_eq!(expanded.shape(), &[3, 1, 2]);
}

#[test]
fn test_expand_existing_dim() {
    let t = crate::tensor::from_vec_f32(vec![1.0], vec![1]).unwrap();
    let expanded = t.expand(&[3, 1]).unwrap();
    assert_eq!(expanded.shape(), &[3, 1]);
}

#[test]
fn test_expand_no_change() {
    let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let expanded = t.expand(&[2, 2]).unwrap();
    assert_eq!(expanded.shape(), &[2, 2]);
}

#[test]
fn test_expand_mixed() {
    let t = crate::tensor::from_vec_f32(vec![1.0, 2.0], vec![1, 2]).unwrap(); // Shape [1, 2]
    let expanded = t.expand(&[3, 2]).unwrap(); // Expand dim 0
    assert_eq!(expanded.shape(), &[3, 2]);
}

#[test]
fn test_expand_backward() {
    let t_data = vec![1.0, 2.0, 3.0];
    let t_shape = vec![3];
    let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
    t.set_requires_grad(true).unwrap();
}

#[test]
fn test_expand_backward_add_dims() {
    let t_data = vec![1.0, 2.0];
    let t_shape = vec![2];
    let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
    t.set_requires_grad(true).unwrap();
} 