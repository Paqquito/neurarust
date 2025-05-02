// neurarust-core/src/ops/arithmetic/add.rs

use crate::tensor::Tensor;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use std::ops::{Add, AddAssign};
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::default::Default;
use crate::error::NeuraRustError;
use std::cmp::PartialEq;
use crate::device::StorageDevice;

// --- Forward Operation --- 

/// Performs element-wise addition for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
/// This operation creates a new Tensor with copied data on the same device.
pub fn add_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Add<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + PartialEq,
{
    // Acquire read locks for inputs
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check --- 
    // Ensure both tensors are on the same device (and implicitly CPU for now)
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("Cannot add tensors on different devices: {:?} and {:?}", a_guard.device, b_guard.device)
        ));
    }
    let device = a_guard.device; // Device for the output tensor
    // Ensure the operation is supported on this device (currently only CPU)
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            format!("Addition is currently only supported on CPU, not {:?}", device)
        ));
    }
    // --- Get CPU Data Buffers --- 
    // We know they are on CPU, so unwrap the result of cpu_data()
    // Note: cpu_data() returns Result<&Arc<Vec<T>>, _>, so we dereference it
    // and clone the Arc for safe access later. 
    let a_data_arc = a_guard.data.cpu_data()?.clone(); // Clone the Arc<Vec<T>>
    let b_data_arc = b_guard.data.cpu_data()?.clone(); // Clone the Arc<Vec<T>>

    // --- Shape and Broadcasting --- 
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    
    let output_shape = broadcast_shapes(a_shape, b_shape)
        .map_err(|_e| NeuraRustError::BroadcastError { 
            shape1: a_shape.clone(), 
            shape2: b_shape.clone()
        })?;

    // --- Calculation --- 
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    
    let result_strides = calculate_strides(&output_shape);
    let rank_diff_a = output_shape.len().saturating_sub(a_guard.shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_guard.shape.len());
    
    let mut input_a_coords = vec![0; a_guard.shape.len()];
    let mut input_b_coords = vec![0; b_guard.shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, &output_shape);
        
        for dim_idx in 0..a_guard.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_guard.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        // Get offset using the guard's metadata
        let offset_a = a_guard.get_offset(&input_a_coords);
        // Access data using the cloned Arc<Vec<T>>
        let val_a = a_data_arc[offset_a]; 
        
        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        // Get offset using the guard's metadata
        let offset_b = b_guard.get_offset(&input_b_coords);
        // Access data using the cloned Arc<Vec<T>>
        let val_b = b_data_arc[offset_b]; 

        result_data_vec.push(val_a + val_b);
    }

    // Drop read locks explicitly (although they drop implicitly at end of scope)
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor --- 
    // The result tensor is created on the same device as the inputs (CPU here)
    Tensor::new(result_data_vec, output_shape.clone())
}

/// REMOVED: In-place AddAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.
// impl<'a, T> AddAssign<&'a Tensor<T>> for Tensor<T>
// where
//     T: AddAssign + Copy + Clone,
// {
//     fn add_assign(&mut self, other: &'a Tensor<T>) { ... }
// }

// --- Backward Operation (REMOVED for Phase 0) --- 
// #[derive(Debug)]
// struct AddBackward<T: 'static> { ... }
// impl<T> BackwardOp<T> for AddBackward<T> { ... }

// --- Tests --- 

#[cfg(test)]
mod tests {
    use super::*; 
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Add, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    use crate::error::NeuraRustError;
    use std::default::Default;

    // Helpers remain the same
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + Sum>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> { 
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    // REMOVED: fn create_test_tensor_with_grad(...)

    #[test]
    fn test_add_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![6_i32, 8, 10, 12];
        let expected_shape = vec![2, 2];
        
        let result = add_op(&t1, &t2); // Use add_op now
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        
        // Compare data: borrow_data_buffer returns Arc<Buffer<T>>
        // Need to access cpu_data() within it.
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
        // REMOVED: assert!(!res_tensor.requires_grad());
    }

    #[test]
    fn test_add_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        
        let result = add_op(&t1, &t_non_broadcast); // Use add_op
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::BroadcastError { shape1, shape2 } => {
                assert_eq!(shape1, vec![2, 2]);
                assert_eq!(shape2, vec![2, 3]);
            },
            _ => panic!("Incorrect error type returned"),
        }
    }
    
    #[test]
    fn test_add_broadcasting() {
        let t1 = create_test_tensor(vec![1_i32, 2], vec![1, 2]); // Shape [1, 2]
        let t2 = create_test_tensor(vec![10_i32, 20], vec![2, 1]); // Shape [2, 1]
        let expected_data = vec![11_i32, 12, 21, 22];
        let expected_shape = vec![2, 2];

        let result = add_op(&t1, &t2).expect("Broadcasting add failed");
        assert_eq!(result.shape(), expected_shape);
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        // Test adding a scalar
        let t_mat = create_test_tensor(vec![1_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_scalar = Tensor::scalar(10.0_f32);
        let expected_scalar_add = vec![11.0_f32, 12.0, 13.0, 14.0];
        
        let result_scalar = add_op(&t_mat, &t_scalar).expect("Scalar add failed");
        assert_eq!(result_scalar.shape(), vec![2, 2]);
        // Updated data access
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc.cpu_data().expect("Scalar add result not on CPU");
        assert_eq!(scalar_res_cpu_data.as_slice(), expected_scalar_add.as_slice());
         
        let result_scalar_rev = add_op(&t_scalar, &t_mat).expect("Scalar add reverse failed");
        assert_eq!(result_scalar_rev.shape(), vec![2, 2]);
        // Updated data access
        let scalar_rev_res_buffer_arc = result_scalar_rev.borrow_data_buffer();
        let scalar_rev_res_cpu_data = scalar_rev_res_buffer_arc.cpu_data().expect("Scalar add reverse result not on CPU");
        assert_eq!(scalar_rev_res_cpu_data.as_slice(), expected_scalar_add.as_slice());
    }

    // REMOVED: Backward tests
    // #[test]
    // fn test_add_backward() -> Result<(), NeuraRustError> { ... }
    // #[test]
    // fn test_add_backward_broadcast() -> Result<(), NeuraRustError> { ... }
} 