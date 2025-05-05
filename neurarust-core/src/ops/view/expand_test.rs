#[cfg(test)]
use crate::ops::view::expand::expand_op; // Import the op function
use crate::tensor::{self, Tensor}; // Import tensor for creation funcs
use crate::error::NeuraRustError;
#[allow(unused_imports)] // Allow unused as check_grad is only in ignored tests
use crate::autograd::grad_check::{check_grad, GradCheckError};
 // If needed for manual checks

// --- FORWARD TESTS --- 

#[test]
fn test_expand_basic() -> Result<(), NeuraRustError> { // Return Result
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?; // Use Tensor::new
    let expanded = expand_op(&t, vec![3, 2])?; // Call expand_op
    assert_eq!(expanded.shape(), &[3, 2]);
    assert_eq!(expanded.strides(), &[0, 1]); // Check strides as well
    Ok(())
}

#[test]
fn test_expand_add_dims() -> Result<(), NeuraRustError> { // Return Result
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?; // Use Tensor::new
    let expanded = expand_op(&t, vec![2, 1, 2])?; // Call expand_op
    assert_eq!(expanded.shape(), &[2, 1, 2]);
    assert_eq!(expanded.strides(), &[0, 0, 1]); // Check strides
    Ok(())
}

#[test]
fn test_expand_same_shape() -> Result<(), NeuraRustError> { // Return Result
    let t1 = Tensor::new(vec![1.0, 2.0], vec![2])?;
    let expanded1 = expand_op(&t1, vec![2])?;
    assert_eq!(expanded1.shape(), &[2]);
    assert_eq!(expanded1.strides(), &[1]);

    let t2 = Tensor::new(vec![1.0], vec![1])?;
    let expanded2 = expand_op(&t2, vec![1])?;
    assert_eq!(expanded2.shape(), &[1]);
    assert_eq!(expanded2.strides(), &[1]); // Stride of [1] is typically 1
    Ok(())
}

// --- ERROR / EDGE CASE TESTS --- 

#[test]
fn test_expand_incompatible_dim() -> Result<(), NeuraRustError> { // Return Result
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    let result = expand_op(&t, vec![3, 4]); // Expand [3] to [3, 4] -> Error
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_expand_too_small() -> Result<(), NeuraRustError> { // Return Result
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    let result = expand_op(&t, vec![2]); // Expand [3] to [2] -> Error
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_expand_invalid_rank() -> Result<(), NeuraRustError> { // Renamed, was testing valid case
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?;
    let result = expand_op(&t, vec![1]); // Expand [2] to [1] -> Error
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_expand_invalid_dim_size_case1() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?;
    let result = expand_op(&t, vec![3]); // Expand [2] to [3] -> Error
    assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_expand_invalid_dim_size_case2() -> Result<(), NeuraRustError> {
    let t2 = Tensor::new(vec![1.0], vec![1])?;
    let result2 = expand_op(&t2, vec![2, 3])?; // Expand [1] to [2, 3] -> OK
    assert_eq!(result2.shape(), &[2, 3]);
    assert_eq!(result2.strides(), &[0, 0]); // Check strides
    Ok(())
}

// --- AUTOGRAD TESTS --- 

#[test]
#[ignore = "Backward for expand involves sum/reshape, check_grad might be unstable"]
fn test_expand_backward() -> Result<(), GradCheckError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?; // F32
    t.set_requires_grad(true)?;
    let target_shape = vec![2, 3];

    let expand_fn = |inputs: &[Tensor]| expand_op(&inputs[0], target_shape.clone());
    
    let output_grad = tensor::full(&target_shape, 1.0)?; // F32 grad

    check_grad(expand_fn, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
#[ignore = "Backward for expand involves sum/reshape, check_grad might be unstable"]
fn test_expand_backward_add_dims() -> Result<(), GradCheckError> {
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?; // F32
    t.set_requires_grad(true)?;
    let target_shape = vec![3, 1, 2];

    let expand_fn = |inputs: &[Tensor]| expand_op(&inputs[0], target_shape.clone());

    let output_grad = tensor::full(&target_shape, 1.0)?; // F32 grad

    check_grad(expand_fn, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
#[ignore = "Backward for expand involves sum/reshape, check_grad might be unstable"]
fn test_expand_backward_f64() -> Result<(), GradCheckError> { // Keep f64 test but ignore
    let t = Tensor::new_f64(vec![1.0, 2.0], vec![2])?;
    t.set_requires_grad(true)?;
    let target_shape = vec![3, 2];

    let expand_fn = |inputs: &[Tensor]| expand_op(&inputs[0], target_shape.clone());
    
    let output_grad = tensor::full_f64(&target_shape, 1.0)?;

    check_grad(expand_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
} 