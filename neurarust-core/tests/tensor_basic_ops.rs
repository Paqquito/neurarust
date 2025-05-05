use neurarust_core::Tensor;

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

    // NOTE: PartialEq is not derived for Tensor as pointer equality is not semantic equality.
    // Need a custom comparison function or use check_tensor_near.
    // For now, let's test some properties.
    assert_eq!(t1.shape(), t3.shape());
    assert_eq!(t1.get_f32_data().unwrap(), t3.get_f32_data().unwrap());

    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.get_f32_data().unwrap(), t2.get_f32_data().unwrap());

    // Optional: Check that the Tensor structs themselves are at different memory addresses
    assert_ne!(std::ptr::addr_of!(t1), std::ptr::addr_of!(t2), "t1 and t2 should be distinct Tensor instances");

    assert_ne!(t1.get_f32_data().unwrap(), t4.get_f32_data().unwrap(), "t1 and t4 should have different data content");
    assert_ne!(t1.shape(), t5.shape(), "t1 and t5 should have different shapes");
}

#[test]
fn test_get_element_via_data() { // Renamed test as .get() doesn't exist
    let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let data = t.get_f32_data().unwrap();
    // Original indices [0,0], [0,2], [1,0], [1,2]
    // Strides are [3, 1]
    // Linear indices: 0*3+0*1=0, 0*3+2*1=2, 1*3+0*1=3, 1*3+2*1=5
    assert_eq!(data[0], 1.0);
    assert_eq!(data[2], 3.0);
    assert_eq!(data[3], 4.0);
    assert_eq!(data[5], 6.0);
}

#[test]
#[ignore = "Skipping until an element access method (e.g., Tensor::at(&[...])) exists"] // .get() doesn't exist
fn test_get_element_out_of_bounds() {
    /*
    let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    assert!(t.at(&[2, 0]).is_err()); // Replace .get with hypothetical .at
    assert!(t.at(&[0, 2]).is_err());
    match t.at(&[0, 2]).err().unwrap() {
        NeuraRustError::IndexOutOfBounds { index, shape } => {
            assert_eq!(index, vec![0, 2]);
            assert_eq!(shape, vec![2, 2]);
        }
        e => panic!("Expected IndexOutOfBounds, got {:?}", e),
    }
    */
}

#[test]
#[ignore = "Skipping until an element access method (e.g., Tensor::at(&[...])) exists"] // .get() doesn't exist
fn test_get_element_wrong_ndim() {
    /*
    let t = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    assert!(t.at(&[0]).is_err()); // Replace .get with hypothetical .at
    assert!(t.at(&[0, 0, 0]).is_err());
    match t.at(&[0]).err().unwrap() {
        NeuraRustError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        e => panic!("Expected DimensionMismatch, got {:?}", e),
    }
    */
}

#[test]
fn test_detach_basic() {
    let t1_res = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(t1_res.is_ok());
    let t1 = t1_res.unwrap();
    t1.set_requires_grad(true).unwrap();
    // Use add_op to create a grad_fn
    let t1_added = neurarust_core::ops::arithmetic::add_op(&t1, &t1);
    assert!(t1_added.is_ok());
    // Check the *result* of add_op for grad_fn, or original tensor if add was in-place (not the case here)
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

    // REMOVED Pointer comparison tests due to private fields
    // Verify underlying buffer is shared (via pointer comparison of the Arc<Buffer>)
    // let t1_buffer_ptr = Arc::as_ptr(&t1_added.read_data().buffer);
    // let t2_buffer_ptr = Arc::as_ptr(&t2.read_data().buffer);
    // assert_eq!(t1_buffer_ptr, t2_buffer_ptr, "Detached tensor should share the same buffer Arc");
    // Verify TensorData Arcs are different
    // assert!(!Arc::ptr_eq(&t1_added.data, &t2.data), "Detached tensor should have a different TensorData Arc");
}

// TODO: Add test for detach modifying shared data? (Maybe not needed if buffer sharing is confirmed) 