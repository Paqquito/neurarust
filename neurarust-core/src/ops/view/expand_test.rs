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
    let expanded = expand_op(&t, &[3, 2])?; // Pass as slice of isize
    assert_eq!(expanded.shape(), &[3, 2]);
    assert_eq!(expanded.strides(), &[0, 1]); // Check strides as well
    Ok(())
}

#[test]
fn test_expand_with_neg_one() -> Result<(), NeuraRustError> {
    let data = vec![1.0f32, 2.0];
    let t_original_2_1 = Tensor::new(data.clone(), vec![2, 1])?; // Shape [2,1]

    // Cas 1: target_shape = [-1, 0] sur source [2,1]
    let res_case1 = expand_op(&t_original_2_1, &[-1, 0]);
    let expected_msg_case1 = "Target dimension 1 is 0, but corresponding source dimension 1 is not 0 (it's 1). Only a 0-sized source dim can be targeted with 0.".to_string();
    match res_case1 {
        Err(NeuraRustError::UnsupportedOperation(actual_msg)) => {
            assert_eq!(actual_msg, expected_msg_case1, "Mismatch in error message for case 1");
        }
        _ => {
            panic!("Case 1: Expected Err(UnsupportedOperation), got {:?}", res_case1);
        }
    }

    // Cas 2: target_shape = [-1, 3] sur source [2] (t2)
    let data2 = vec![7.0f32, 8.0];
    let t2 = Tensor::new(data2, vec![2])?; // Shape [2]
    let res_case2 = expand_op(&t2, &[-1, 3]);
    let expected_msg_case2 = "Dimension size -1 not allowed for new dimensions in expand.".to_string();
    match res_case2 {
        Err(NeuraRustError::UnsupportedOperation(actual_msg)) => {
            assert_eq!(actual_msg, expected_msg_case2, "Mismatch in error message for case 2");
        }
        _ => {
            panic!("Case 2: Expected Err(UnsupportedOperation), got {:?}", res_case2);
        }
    }

    // Cas 3: target_shape = [-1, -1, 0] sur source [1] (t3)
    let t3 = Tensor::new(vec![1.0f32], vec![1])?;
    let res_case3 = expand_op(&t3, &[-1, -1, 0]);
    let expected_msg_case3 = "Dimension size -1 not allowed for new dimensions in expand.".to_string();
    assert!(matches!(
        res_case3,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg_case3
    ));
    
    // Cas 4: target_shape = [0] sur source [2,1] (t_original_2_1)
    let res_case4 = expand_op(&t_original_2_1, &[0]);
    let expected_msg_case4 = "Target rank cannot be less than source rank for expand.".to_string();
     assert!(matches!(
        res_case4,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg_case4
    ));

    Ok(())
}

#[test]
fn test_expand_error_neg_one_for_new_dim() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0], vec![1])?;
    let result = expand_op(&t, &[-1, 1]); // Trying to use -1 for a new leading dimension.
    assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(_))));
    Ok(())
}

#[test]
fn test_expand_error_negative_dim_size() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0], vec![1])?;
    let result = expand_op(&t, &[-2, 1]);
    assert!(matches!(result, Err(NeuraRustError::UnsupportedOperation(_))));
    Ok(())
}

#[test]
fn test_expand_add_dims() -> Result<(), NeuraRustError> { // Return Result
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?; // Use Tensor::new
    let expanded = expand_op(&t, &[2, 1, 2])?; // Call expand_op
    assert_eq!(expanded.shape(), &[2, 1, 2]);
    assert_eq!(expanded.strides(), &[0, 0, 1]); // Check strides
    Ok(())
}

#[test]
fn test_expand_same_shape() -> Result<(), NeuraRustError> { // Return Result
    let t1 = Tensor::new(vec![1.0, 2.0], vec![2])?;
    let expanded1 = expand_op(&t1, &[2])?;
    assert_eq!(expanded1.shape(), &[2]);
    assert_eq!(expanded1.strides(), &[1]);

    let t2 = Tensor::new(vec![1.0], vec![1])?;
    let expanded2 = expand_op(&t2, &[1])?;
    assert_eq!(expanded2.shape(), &[1]);
    assert_eq!(expanded2.strides(), &[1]); // Stride of [1] is typically 1
    Ok(())
}

// --- ERROR / EDGE CASE TESTS --- 

#[test]
fn test_expand_incompatible_dim() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?; // Shape [3]
    let result = expand_op(&t, &[2]);
    let expected_msg1 = "Target dimension 0 size 2 is smaller than source dimension 0 size 3.".to_string();
    assert!(matches!(
        result,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg1
    ));
    
    let result2 = expand_op(&t, &[4]);
    let expected_msg2 = "Cannot expand source dimension 0 (size 3) to target size 4 because source is not 1.".to_string();
     assert!(matches!(
        result2,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg2
    ));
    Ok(())
}

#[test]
fn test_expand_too_small() -> Result<(), NeuraRustError> { // Renommé pour refléter l'erreur de rang
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let result = expand_op(&t, &[2]);
    let expected_msg = "Target rank cannot be less than source rank for expand.".to_string();
    assert!(matches!(
        result,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg
    ));
    Ok(())
}

#[test]
fn test_expand_invalid_rank() -> Result<(), NeuraRustError> { // Ce test est redondant avec test_expand_too_small
                                                              // mais je le garde pour vérifier le message exact si besoin.
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let result = expand_op(&t, &[1]); // Target rank 1 < source rank 2
    let expected_msg = "Target rank cannot be less than source rank for expand.".to_string();
     assert!(matches!(
        result,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg
    ));
    Ok(())
}

#[test]
fn test_expand_invalid_dim_size_case1() -> Result<(), NeuraRustError> {
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?; // Shape [2]
    let result = expand_op(&t, &[1]);
    let expected_msg = "Target dimension 0 size 1 is smaller than source dimension 0 size 2.".to_string();
    assert!(matches!(
        result,
        Err(NeuraRustError::UnsupportedOperation(msg)) if msg == expected_msg
    ));
    Ok(())
}

#[test]
fn test_expand_invalid_dim_size_case2() -> Result<(), NeuraRustError> {
    let t2 = Tensor::new(vec![1.0], vec![1])?;
    let result2 = expand_op(&t2, &[2, 3])?; // Expand [1] to [2, 3] -> OK
    assert_eq!(result2.shape(), &[2, 3]);
    assert_eq!(result2.strides(), &[0, 0]); // Check strides
    Ok(())
}

// --- AUTOGRAD TESTS --- 

#[test]
fn test_expand_backward() -> Result<(), GradCheckError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?; // F32
    t.set_requires_grad(true)?;
    let target_shape_isize: Vec<isize> = vec![2, 3];

    let expand_fn = |inputs: &[Tensor]| expand_op(&inputs[0], &target_shape_isize);
    
    let output_grad_shape: Vec<usize> = target_shape_isize.iter().map(|&x| x as usize).collect();
    let output_grad = tensor::full(&output_grad_shape, 1.0)?; // F32 grad

    check_grad(expand_fn, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
fn test_expand_backward_add_dims() -> Result<(), GradCheckError> {
    let t = Tensor::new(vec![1.0, 2.0], vec![2])?; // F32
    t.set_requires_grad(true)?;
    let target_shape_isize: Vec<isize> = vec![3, 1, 2];

    let expand_fn = |inputs: &[Tensor]| expand_op(&inputs[0], &target_shape_isize);

    let output_grad_shape: Vec<usize> = target_shape_isize.iter().map(|&x| x as usize).collect();
    let output_grad = tensor::full(&output_grad_shape, 1.0)?; // F32 grad

    check_grad(expand_fn, &[t], &output_grad, 1e-3, 1e-4, 1e-3)
}

#[test]
fn test_expand_backward_f64() -> Result<(), GradCheckError> { // Keep f64 test but ignore
    let t = Tensor::new_f64(vec![1.0, 2.0], vec![2])?;
    t.set_requires_grad(true)?;
    let target_shape_isize: Vec<isize> = vec![3, 2];

    let expand_fn = |inputs: &[Tensor]| expand_op(&inputs[0], &target_shape_isize);
    
    let output_grad_shape: Vec<usize> = target_shape_isize.iter().map(|&x| x as usize).collect();
    let output_grad = tensor::full_f64(&output_grad_shape, 1.0)?;

    check_grad(expand_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
} 