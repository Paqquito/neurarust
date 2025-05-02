use crate::tensor::Tensor;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use std::ops::{Sub, AddAssign};
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::default::Default;
use crate::error::NeuraRustError;
use std::cmp::PartialEq;
use crate::device::StorageDevice;

// --- Forward Operation --- 

/// Performs element-wise subtraction for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
/// This operation creates a new Tensor with copied data on the same device.
pub fn sub_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Sub<Output = T> + Copy + Clone + Debug + Default + Zero + One + Sum + AddAssign + 'static + PartialEq,
{
    // Acquire read locks for inputs
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check --- 
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("Cannot subtract tensors on different devices: {:?} and {:?}", a_guard.device, b_guard.device)
        ));
    }
    let device = a_guard.device; 
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            format!("Subtraction is currently only supported on CPU, not {:?}", device)
        ));
    }
    // --- Get CPU Data Buffers --- 
    let a_data_arc = a_guard.data.cpu_data()?.clone(); 
    let b_data_arc = b_guard.data.cpu_data()?.clone(); 

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
        let offset_a = a_guard.get_offset(&input_a_coords);
        let val_a = a_data_arc[offset_a]; // Use cpu data arc
        
        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        let offset_b = b_guard.get_offset(&input_b_coords);
        let val_b = b_data_arc[offset_b]; // Use cpu data arc

        result_data_vec.push(val_a - val_b); // Perform subtraction
    }

    // Drop read locks 
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor --- 
    Tensor::new(result_data_vec, output_shape.clone())
}

/// REMOVED: In-place SubAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.

// --- Tests --- 

#[cfg(test)]
mod tests {
    use super::*; // Import new `sub_op` function
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Sub, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    use crate::error::NeuraRustError;
    use std::default::Default;
    use std::cmp::PartialEq;

    // Update test helpers bounds if necessary
    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> 
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Sub<Output=T> + Default + Sum + 'static
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = create_test_tensor(vec![10_i32, 20, 30, 40], vec![2, 2]);
        let t2 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let expected_data = vec![9_i32, 18, 27, 36];
        let expected_shape = vec![2, 2];
        
        let result = sub_op(&t1, &t2); // Use sub_op
        assert!(result.is_ok());
        let res_tensor = result.unwrap();

        // Use borrow_data_buffer and cpu_data for comparison
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice(), "Data mismatch");
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
    }

    #[test]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        
        let result = sub_op(&t1, &t_non_broadcast); // Use sub_op
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::BroadcastError { .. }));
    }
    
    #[test]
    fn test_sub_broadcasting() {
        let t1 = create_test_tensor(vec![10_i32, 20], vec![1, 2]); // Shape [1, 2]
        let t2 = create_test_tensor(vec![1_i32, 2], vec![2, 1]); // Shape [2, 1]
        let expected_data = vec![9_i32, 19, 8, 18];
        let expected_shape = vec![2, 2];

        let result = sub_op(&t1, &t2).expect("Broadcasting sub failed");
        assert_eq!(result.shape(), expected_shape);
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        // Test subtracting a scalar
        let t_mat = create_test_tensor(vec![10_f32, 20.0], vec![2, 1]);
        let t_scalar = Tensor::scalar(3.0_f32);
        let expected_scalar_sub = vec![7.0_f32, 17.0];
        let result_scalar = sub_op(&t_mat, &t_scalar).expect("Scalar sub failed");
        assert_eq!(result_scalar.shape(), vec![2, 1]);
        // Updated data access
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc.cpu_data().expect("Scalar sub result not on CPU");
        assert_eq!(scalar_res_cpu_data.as_slice(), expected_scalar_sub.as_slice());
         
        // Test scalar - tensor
        let expected_scalar_sub_rev = vec![-7.0_f32, -17.0];
        let result_scalar_rev = sub_op(&t_scalar, &t_mat).expect("Scalar sub reverse failed");
        assert_eq!(result_scalar_rev.shape(), vec![2, 1]);
        // Updated data access
        let scalar_rev_res_buffer_arc = result_scalar_rev.borrow_data_buffer();
        let scalar_rev_res_cpu_data = scalar_rev_res_buffer_arc.cpu_data().expect("Scalar sub reverse result not on CPU");
        assert_eq!(scalar_rev_res_cpu_data.as_slice(), expected_scalar_sub_rev.as_slice());
    }

    // Remove SubAssign tests
} 