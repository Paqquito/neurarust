use crate::tensor::Tensor;
use std::fmt::Debug;
use num_traits::{Zero, One};

/// Checks if two tensors are approximately equal (shape and data within tolerance).
/// Assumes the actual tensor is on the CPU.
/// Panics if shapes differ or data differs significantly.
pub fn check_tensor_near<T>(
    actual: &Tensor<T>,
    expected_shape: &[usize],
    expected_data: &[T],
    tolerance: T,
)
where
    T: PartialEq + PartialOrd + Debug + Copy + std::ops::Sub<Output = T> + Zero + One,
{
    assert_eq!(actual.shape(), expected_shape, "Shape mismatch");
    
    // Access CPU data buffer
    let actual_buffer_arc = actual.borrow_data_buffer();
    let actual_data_arc = actual_buffer_arc.cpu_data().expect("Actual tensor not on CPU in check_tensor_near");
    
    assert_eq!(actual_data_arc.len(), expected_data.len(), "Data length mismatch");

    // Iterate over the CPU data slice
    for (i, (a, e)) in actual_data_arc.iter().zip(expected_data.iter()).enumerate() {
        let diff = *a - *e;
        // Check if the difference is outside the tolerance range [ -tolerance, +tolerance ]
        // Equivalent to checking !(diff >= -tolerance && diff <= tolerance)
        // Requires PartialOrd trait
        let is_outside_tolerance = if diff > T::zero() { 
            diff > tolerance // Check against positive tolerance
        } else { 
            diff < T::zero() - tolerance // Check against negative tolerance equivalent (0 - tol)
        };

        if is_outside_tolerance {
            panic!(
                "Data mismatch at index {}: actual={:?}, expected={:?}, diff={:?}, tolerance={:?}",
                i, a, e, diff, tolerance
            );
        }
    }
} 