use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use std::sync::Arc;

use std::fmt::Debug;
use std::sync::RwLock;

// --- Backward Operation Structure ---
#[derive(Debug)]
struct SubBackward {
    // Store Arcs directly for thread safety and graph linkage
    a_node: Arc<RwLock<TensorData>>,
    b_node: Arc<RwLock<TensorData>>,
    a_requires_grad: bool, // Store flags to return only necessary pointers in inputs()
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::with_capacity(2);

        // Use reduce_to_shape for gradients
        if self.a_requires_grad {
            let a_guard = self.a_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock A node in SubBackward".to_string()))?;
            let grad_a = grad_output.reduce_to_shape(&a_guard.shape)?;
            grads.push(grad_a);
        }

        if self.b_requires_grad {
            let b_guard = self.b_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock B node in SubBackward".to_string()))?;
            let grad_b_unreduced = crate::ops::arithmetic::neg_op(grad_output)?;
            let grad_b = grad_b_unreduced.reduce_to_shape(&b_guard.shape)?;
            grads.push(grad_b);
        }
        
        Ok(grads)
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        // Return pointers only for inputs that required grad
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

// --- Forward Operation ---
pub fn sub_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor A data for reading".to_string()))?;
    let b_guard = b.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor B data for reading".to_string()))?;

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "sub_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "sub_op currently only supports CPU".to_string()
        ));
    }

    // --- DType Check & Promotion (Simplified for F32 only) ---
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("sub_op currently only supports F32, got {:?} and {:?}", a_guard.dtype, b_guard.dtype)
        ));
    }
    let _output_dtype = DType::F32;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Extract Data & Metadata ---
    let a_shape = a_guard.shape.clone(); // Keep original shapes for potential backward reduction
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;
    let a_requires_grad = a_guard.requires_grad;
    let b_requires_grad = b_guard.requires_grad;

    let a_buffer_data_arc = a_guard.buffer().try_get_cpu_f32()?.clone(); 
    let b_buffer_data_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    // Keep input TensorData Arcs if needed for backward pass
    let a_node_arc = if a_requires_grad || b_requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if a_requires_grad || b_requires_grad { Some(b.data.clone()) } else { None };

    // Drop guards before computation
    drop(a_guard);
    drop(b_guard);

    // --- Calculation Logic (Manual Broadcasting) ---
    let numel_out = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_out);

    let a_data = a_buffer_data_arc.as_slice();
    let b_data = b_buffer_data_arc.as_slice();

    // Prepare indices and strides for iteration (similar to add_op)
    let mut a_indices = vec![0; a_shape.len()];
    let mut b_indices = vec![0; b_shape.len()];
    let mut current_indices = vec![0; output_shape.len()];
    let output_rank = output_shape.len();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    for i in 0..numel_out {
        // Calculate multi-dimensional index from linear index i for output_shape
        let mut current_linear = i;
        for dim in (0..output_rank).rev() {
            let shape_val = output_shape[dim];
            if shape_val > 0 { // Avoid division by zero for empty dimensions
                 current_indices[dim] = current_linear % shape_val;
                 current_linear /= shape_val;
            } else {
                 current_indices[dim] = 0;
            }
        }

        // Calculate corresponding indices for a and b considering broadcasting rules
        for dim in 0..output_rank {
            let out_idx = current_indices[dim];
            
            // Index for a (handle rank difference)
            let a_dim_idx = (dim as isize) - (output_rank as isize - a_rank as isize);
            if a_dim_idx >= 0 {
                let a_dim_idx = a_dim_idx as usize;
                a_indices[a_dim_idx] = if a_shape[a_dim_idx] == 1 { 0 } else { out_idx };
            }

            // Index for b (handle rank difference)
            let b_dim_idx = (dim as isize) - (output_rank as isize - b_rank as isize);
            if b_dim_idx >= 0 {
                 let b_dim_idx = b_dim_idx as usize;
                b_indices[b_dim_idx] = if b_shape[b_dim_idx] == 1 { 0 } else { out_idx };
            }
        }

        // Calculate physical offsets using strides
        let a_physical_offset = a_offset + a_indices.iter().zip(a_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        let b_physical_offset = b_offset + b_indices.iter().zip(b_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        
        // Perform subtraction
        result_data_vec.push(a_data[a_physical_offset] - b_data[b_physical_offset]);
    }
    let result_buffer_arc = Arc::new(result_data_vec); // Arc the final Vec

    // --- Create Output TensorData ---
    // Correct call to TensorData::new using the signature from tensor_data.rs
    let output_td = TensorData::new(
        result_buffer_arc.as_ref().clone(), // Pass the owned Vec<f32> 
        output_shape, // Pass the shape
    )?;
    let result_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    // --- Autograd Setup ---
    if a_requires_grad || b_requires_grad {
         if let (Some(a_arc), Some(b_arc)) = (a_node_arc, b_node_arc) {
             let mut output_guard = result_tensor.data.write().map_err(|_| NeuraRustError::InternalError("Failed to lock output tensor data for writing".to_string()))?;
             output_guard.requires_grad = true;
             output_guard.grad_fn = Some(Arc::new(SubBackward {
                 a_node: a_arc,
                 b_node: b_arc,
                 a_requires_grad, // Pass flags
                 b_requires_grad,
             }));
             println!("SubBackward grad_fn set for sub result."); // Temporary debug print
         }
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    
     
    // Import testing utils
    use crate::utils::testing::check_tensor_near;

    #[test]
    fn test_sub_tensors_ok() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let result = sub_op(&t1, &t2).unwrap();
        // Use the Tensor method directly
        let result_data = result.get_f32_data().expect("Failed to get f32 data in test");
        assert_eq!(result_data, vec![-3.0, -3.0, -3.0]);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_sub_tensors_shape_mismatch() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = sub_op(&t1, &t2);
        assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
    }

    #[test]
    fn test_sub_broadcasting() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::new(vec![5.0], vec![1]).unwrap();
        let result = sub_op(&t1, &t2).unwrap();
        // Use the Tensor method directly
        let result_data = result.get_f32_data().expect("Failed to get f32 data in test");
        assert_eq!(result_data, vec![-4.0, -3.0, -2.0]);
        assert_eq!(result.shape(), &[3]);

        let t3 = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let result2 = sub_op(&t3, &t1).unwrap();
        // Use the Tensor method directly
        let result2_data = result2.get_f32_data().expect("Failed to get f32 data in test");
        assert_eq!(result2_data, vec![9.0, 8.0, 7.0, 19.0, 18.0, 17.0]);
        assert_eq!(result2.shape(), &[2, 3]);
    }

    #[test]
    fn test_sub_backward_simple() {
        let a_data = vec![1.0, 2.0, 3.0];
        let b_data = vec![4.0, 5.0, 6.0];
        let shape = vec![3];
        let a = Tensor::new(a_data.clone(), shape.clone()).unwrap();
        let b = Tensor::new(b_data.clone(), shape.clone()).unwrap();
        a.set_requires_grad(true).unwrap();
        b.set_requires_grad(true).unwrap();

        let output = sub_op(&a, &b).unwrap();
        
        let grad_output_data = vec![0.1, 0.2, 0.3];
        let grad_output = Tensor::new(grad_output_data.clone(), shape.clone()).unwrap();

        output.backward(Some(grad_output)).expect("Backward pass failed");

        // Check grad_a
        let grad_a_contig = a.grad().unwrap().contiguous().unwrap();
        let expected_grad_a = grad_output_data.clone(); // grad_a = grad_output
        check_tensor_near(&grad_a_contig, &shape, &expected_grad_a, 1e-6);

        // Check grad_b
        let grad_b_contig = b.grad().unwrap().contiguous().unwrap();
        let expected_grad_b: Vec<f32> = grad_output_data.iter().map(|&g| -g).collect(); // grad_b = -grad_output
        check_tensor_near(&grad_b_contig, &shape, &expected_grad_b, 1e-6);
    }

    #[test]
    fn test_sub_backward_broadcast() {
        // Test case: a [2, 1] - b [3] -> output [2, 3]
        let a_data = vec![10.0, 20.0];
        let b_data = vec![1.0, 2.0, 3.0];
        let a_shape = vec![2, 1];
        let b_shape = vec![3];
        let output_shape = vec![2, 3];

        let a = Tensor::new(a_data.clone(), a_shape.clone()).unwrap();
        let b = Tensor::new(b_data.clone(), b_shape.clone()).unwrap();
        a.set_requires_grad(true).unwrap();
        b.set_requires_grad(true).unwrap();

        let output = sub_op(&a, &b).unwrap();
        
        let grad_output_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // Shape [2, 3]
        let grad_output = Tensor::new(grad_output_data.clone(), output_shape.clone()).unwrap();

        output.backward(Some(grad_output)).expect("Backward pass failed");

        // Check grad_a (needs reduction from [2, 3] to [2, 1])
        let grad_a_contig = a.grad().unwrap().contiguous().unwrap();
        let expected_grad_a = vec![0.6, 1.5]; // Sum cols: [0.1+0.2+0.3, 0.4+0.5+0.6]
        check_tensor_near(&grad_a_contig, &a_shape, &expected_grad_a, 1e-6);

        // Check grad_b (needs reduction from [2, 3] to [3])
        let grad_b_contig = b.grad().unwrap().contiguous().unwrap();
        // Sum rows of -grad_output: [-(0.1+0.4), -(0.2+0.5), -(0.3+0.6)]
        let expected_grad_b = vec![-0.5, -0.7, -0.9]; 
        check_tensor_near(&grad_b_contig, &b_shape, &expected_grad_b, 1e-6);
    }
}
