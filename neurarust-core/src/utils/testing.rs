use crate::tensor::Tensor;
use approx::AbsDiffEq;

/// Checks if two tensors are approximately equal element-wise.
// Adapté pour le nouveau Tensor non-générique (implicitement F32 CPU)
pub fn check_tensor_near(tensor: &Tensor, expected_shape: &[usize], expected_data: &[f32], tol: f32) {
    assert_eq!(
        tensor.shape(),
        expected_shape,
        "Shape mismatch: expected {:?}, got {:?}",
        expected_shape,
        tensor.shape()
    );

    let actual_data = tensor.get_f32_data().expect("Failed to get F32 data for comparison");

    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "Data length mismatch: expected {}, got {}",
        expected_data.len(),
        actual_data.len()
    );

    for (i, (a, b)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            AbsDiffEq::abs_diff_eq(a, b, tol),
            "Data mismatch at index {}: expected {}, got {}. Difference: {}",
            i,
            b,
            a,
            (a - b).abs()
        );
    }
}
