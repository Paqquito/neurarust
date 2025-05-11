use super::*; // Import items from parent module (Tensor, NeuraRustError, etc.)
// Import necessary ops for testing
use crate::ops::arithmetic::add::add_op;

#[test]
fn test_detach_basic() {
    let t1_res = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(t1_res.is_ok());
    let t1 = t1_res.unwrap();
    t1.set_requires_grad(true).unwrap();
    // Use add_op to create a grad_fn
    // Note: Using add_op directly here is okay because this is an internal module test
    let t1_added = add_op(&t1, &t1);
    assert!(t1_added.is_ok());
    // Check the *result* of add_op for grad_fn
    let t1_added = t1_added.unwrap(); 
    assert!(t1_added.requires_grad(), "Result of add should require grad");
    assert!(t1_added.grad_fn().is_some(), "Result of add should have grad_fn");

    // Detach the result of the operation
    let t2 = t1_added.detach();

    // Check detached properties
    assert!(!t2.requires_grad(), "Detached tensor should not require grad");
    assert!(t2.grad_fn().is_none(), "Detached tensor should not have grad_fn");
    assert!(t2.grad().is_none(), "Detached tensor should not have grad");

    // Check metadata and data sharing
    assert_eq!(t1_added.shape(), t2.shape(), "Shapes should be equal");
    assert_eq!(t1_added.dtype(), t2.dtype(), "DTypes should be equal");
    assert_eq!(t1_added.device(), t2.device(), "Devices should be equal");
    assert_eq!(t1_added.strides(), t2.strides(), "Strides should be equal");
    
    // Verify data content equality
    let t1_added_data = t1_added.get_f32_data().unwrap();
    let t2_data = t2.get_f32_data().unwrap();
    assert_eq!(t1_added_data, t2_data, "Data content should be equal");
}

// TODO: Add more tests for autograd methods if needed 