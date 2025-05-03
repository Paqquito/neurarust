use crate::tensor::Tensor;

/// Checks if two tensors are approximately equal (shape and data within tolerance).
/// Assumes the actual tensor is F32 and on the CPU.
/// Panics if shapes differ or data differs significantly.
pub fn check_tensor_near(
    actual: &Tensor,
    expected_shape: &[usize],
    expected_data: &[f32],
    tolerance: f32,
) {
    assert_eq!(actual.shape(), expected_shape, "Shape mismatch");

    // Use get_f32_data to access data
    let actual_data_vec = actual
        .get_f32_data()
        .expect("Failed to get F32 CPU data in check_tensor_near");

    assert_eq!(
        actual_data_vec.len(),
        expected_data.len(),
        "Data length mismatch"
    );

    // Iterate over the f32 data slice
    for (i, (a, e)) in actual_data_vec.iter().zip(expected_data.iter()).enumerate() {
        let diff = (*a - *e).abs(); // Use abs() for f32 comparison
        if diff > tolerance {
            panic!(
                "Data mismatch at index {}: actual={:?}, expected={:?}, diff={:?}, tolerance={:?}",
                i, a, e, diff, tolerance
            );
        }
    }
}

/// Helper to create a simple f32 tensor for testing purposes.
pub(crate) fn create_test_tensor(
    data: Vec<f32>,
    shape: Vec<usize>,
) -> Tensor {
    Tensor::new(data, shape).expect("Failed to create test tensor")
}

/// Helper to create a simple f32 tensor that requires gradient for testing.
pub(crate) fn create_test_tensor_with_grad(
    data: Vec<f32>,
    shape: Vec<usize>,
) -> Tensor {
    let tensor = Tensor::new(data, shape).expect("Failed to create test tensor with grad");
    // Manually set requires_grad on TensorData
    {
        let mut tensor_data_guard = tensor.data.write().unwrap();
        tensor_data_guard.requires_grad = true;
    }
    tensor
}
