use crate::tensor::Tensor;
use std::fmt::Debug;

/// Checks if two tensors are approximately equal (shape and data within tolerance).
/// Panics if shapes differ or data differs significantly.
pub fn check_tensor_near<T>(
    actual: &Tensor<T>,
    expected_shape: &[usize],
    expected_requires_grad: bool,
    expected_data: &[T],
    tolerance: T,
)
where
    T: PartialEq + PartialOrd + Debug + Copy + std::ops::Sub<Output = T> + Zero + One + Neg<Output=T>,
{
    assert_eq!(actual.shape(), expected_shape, "Shape mismatch");
    assert_eq!(actual.requires_grad(), expected_requires_grad, "requires_grad mismatch");
    let actual_data = actual.data();
    assert_eq!(actual_data.len(), expected_data.len(), "Data length mismatch");

    for (i, (a, e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        let diff = *a - *e;
        // Check absolute difference for approximate equality
        // Handle potential negative diff using Neg trait bound or by comparing abs value
        let abs_diff = if diff < T::zero() { -diff } else { diff }; 
        assert!(
            abs_diff <= tolerance,
            "Data mismatch at index {}: actual={:?}, expected={:?}, diff={:?}, tolerance={:?}",
            i, a, e, diff, tolerance
        );
    }
}

// Placeholder for Zero and One for the trait bound
use num_traits::{Zero, One};
// Placeholder for Neg for the trait bound
use std::ops::Neg; 