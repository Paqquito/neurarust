use neurarust_core::{
    error::NeuraRustError,
    // tensor::Tensor, // Removed unused import
};
// use std::sync::Arc; // Removed unused import

// Include the common helper module
mod common;
use common::create_test_tensor;

#[test]
fn test_tensor_equality() {
    let data1 = vec![1.0_f32, 2.0];
    let shape1 = vec![2];
    let t1 = create_test_tensor(data1.clone(), shape1.clone());
    let t2 = create_test_tensor(data1.clone(), shape1.clone()); // Creates a separate Tensor instance
    let t3 = t1.clone(); // Clones Arc<RwLock>, points to same allocation, new Tensor instance
    let t4 = create_test_tensor(vec![3.0, 4.0], shape1.clone());
    let t5 = create_test_tensor(data1.clone(), vec![1, 2]);

    assert_eq!(t1, t1); // Equal to self
    assert_eq!(t1, t3); // Equal to Arc clone (same data, same metadata)

    // t1 and t2 were created separately. Verify they are distinct instances but logically equal.
    // We cannot directly compare Arc pointers anymore as `data` is private.
    // Rely on PartialEq for logical equality.
    // The fact they come from separate `create_test_tensor` calls implies different underlying allocations
    // unless some unexpected optimization occurs. The main point is logical equality.
    assert_eq!(
        t1, t2,
        "t1 and t2 should be equal by content and metadata (PartialEq)"
    );
    // Optional: Check that the Tensor structs themselves are at different memory addresses
    assert_ne!(std::ptr::addr_of!(t1), std::ptr::addr_of!(t2), "t1 and t2 should be distinct Tensor instances");

    assert_ne!(t1, t4, "t1 and t4 should have different data content");
    assert_ne!(t1, t5, "t1 and t5 should have different shapes");
}

#[test]
fn test_get_element() {
    let t = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    assert_eq!(t.get(&[0, 0]).unwrap(), 1);
    assert_eq!(t.get(&[0, 2]).unwrap(), 3);
    assert_eq!(t.get(&[1, 0]).unwrap(), 4);
    assert_eq!(t.get(&[1, 2]).unwrap(), 6);
}

#[test]
fn test_get_element_out_of_bounds() {
    let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    assert!(t.get(&[2, 0]).is_err());
    assert!(t.get(&[0, 2]).is_err());
    match t.get(&[0, 2]).err().unwrap() {
        NeuraRustError::IndexOutOfBounds { index, shape } => {
            assert_eq!(index, vec![0, 2]);
            assert_eq!(shape, vec![2, 2]);
        }
        e => panic!("Expected IndexOutOfBounds, got {:?}", e),
    }
}

#[test]
fn test_get_element_wrong_ndim() {
    let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    assert!(t.get(&[0]).is_err());
    assert!(t.get(&[0, 0, 0]).is_err());
    match t.get(&[0]).err().unwrap() {
        NeuraRustError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        e => panic!("Expected DimensionMismatch, got {:?}", e),
    }
} 