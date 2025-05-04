use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use crate::ops::arithmetic::mul::mul_op;
use crate::ops::arithmetic::neg::neg_op;

// Keep Zero trait for division check
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use num_traits::Zero;

// --- Backward Operation Structure ---
#[derive(Debug)]
struct DivBackward {
    a_node: Arc<RwLock<TensorData>>,
    b_node: Arc<RwLock<TensorData>>,
    b_tensor_clone: Tensor,
    a_requires_grad: bool,
    b_requires_grad: bool,
}

// --- Backward Operation Implementation ---
impl BackwardOp for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // For z = a / b:
        // grad(a) = grad_output * dz/da = grad_output * (1 / b)
        // grad(b) = grad_output * dz/db = grad_output * (-a / b^2)
        let mut grads = Vec::with_capacity(2);

        if self.a_requires_grad {
            let a_guard = self.a_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock A node in DivBackward".to_string()))?;
            let grad_a_unreduced = div_op(grad_output, &self.b_tensor_clone)?;
            let grad_a = grad_a_unreduced.reduce_to_shape(&a_guard.shape)?;
            grads.push(grad_a);
        }

        if self.b_requires_grad {
            let b_guard = self.b_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock B node in DivBackward".to_string()))?;
            let a_tensor = Tensor { data: self.a_node.clone() };
            
            let b_squared = mul_op(&self.b_tensor_clone, &self.b_tensor_clone)?;
            let neg_a = neg_op(&a_tensor)?;
            let inner_term = div_op(&neg_a, &b_squared)?;
            let grad_b_unreduced = mul_op(grad_output, &inner_term)?;
            let grad_b = grad_b_unreduced.reduce_to_shape(&b_guard.shape)?;
            grads.push(grad_b);
        }

        Ok(grads)
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut ids = Vec::new();
        if self.a_requires_grad { ids.push(Arc::as_ptr(&self.a_node)); }
        if self.b_requires_grad { ids.push(Arc::as_ptr(&self.b_node)); }
        ids
    }
}

// --- Forward Operation ---
pub fn div_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor A data for reading".to_string()))?;
    let b_guard = b.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock tensor B data for reading".to_string()))?;

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "div_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            "div_op currently only supports CPU tensors.".to_string(),
        ));
    }

    // --- DType Check ---
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "div_op currently only supports F32 tensors.".to_string(),
        ));
    }
    let _output_dtype = DType::F32;

    // --- Shape Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Extract Data & Metadata ---
    let a_shape = a_guard.shape.clone(); 
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;
    let a_requires_grad = a_guard.requires_grad;
    let b_requires_grad = b_guard.requires_grad;

    let a_buffer_data_arc = a_guard.buffer().try_get_cpu_f32()?.clone(); 
    let b_buffer_data_arc = b_guard.buffer().try_get_cpu_f32()?.clone();
    let a_node_arc = if a_requires_grad || b_requires_grad { Some(a.data.clone()) } else { None };
    let b_node_arc = if a_requires_grad || b_requires_grad { Some(b.data.clone()) } else { None };
    let b_tensor_clone = if b_requires_grad { Some(b.clone()) } else { None };

    drop(a_guard);
    drop(b_guard);

    // --- Calculation Logic (Manual Broadcasting) ---
    let numel_out = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_out);
    let a_data = a_buffer_data_arc.as_slice();
    let b_data = b_buffer_data_arc.as_slice();

    let mut a_indices = vec![0; a_shape.len()];
    let mut b_indices = vec![0; b_shape.len()];
    let mut current_indices = vec![0; output_shape.len()];
    let output_rank = output_shape.len();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    for i in 0..numel_out {
        let mut current_linear = i;
        for dim in (0..output_rank).rev() {
            let shape_val = output_shape[dim];
            if shape_val > 0 { current_indices[dim] = current_linear % shape_val; current_linear /= shape_val; } else { current_indices[dim] = 0; }
        }
        for dim in 0..output_rank {
            let out_idx = current_indices[dim];
            let a_dim_idx = (dim as isize) - (output_rank as isize - a_rank as isize); if a_dim_idx >= 0 { a_indices[a_dim_idx as usize] = if a_shape[a_dim_idx as usize] == 1 { 0 } else { out_idx }; }
            let b_dim_idx = (dim as isize) - (output_rank as isize - b_rank as isize); if b_dim_idx >= 0 { b_indices[b_dim_idx as usize] = if b_shape[b_dim_idx as usize] == 1 { 0 } else { out_idx }; }
        }
        let a_physical_offset = a_offset + a_indices.iter().zip(a_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        let b_physical_offset = b_offset + b_indices.iter().zip(b_strides.iter()).map(|(&idx, &stride)| idx * stride).sum::<usize>();
        
        let divisor = b_data[b_physical_offset];
        if divisor.is_zero() {
            return Err(NeuraRustError::DivisionByZero); 
        }
        result_data_vec.push(a_data[a_physical_offset] / divisor);
    }
    let result_buffer_arc = Arc::new(result_data_vec);

    // --- Create Output TensorData ---
    let output_td = TensorData::new(
        result_buffer_arc.as_ref().clone(),
        output_shape,
    )?;
    let result_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    // --- Autograd Setup ---
    // Determine if autograd is needed based on original inputs
    let autograd_needed = a_requires_grad || b_requires_grad;

    if autograd_needed {
        // We NEED the original Arcs and potentially the b clone if autograd is needed
        // Retrieve them from the Options created earlier.
        let a_arc = a_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing a_node_arc when autograd needed".to_string()))?;
        let b_arc = b_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing b_node_arc when autograd needed".to_string()))?;
        // Only need b_clone if b itself requires grad for the backward calculation
        let b_clone_for_backward = if b_requires_grad { 
            b_tensor_clone.ok_or_else(|| NeuraRustError::InternalError("Missing b_tensor_clone when b requires grad".to_string()))?
        } else {
            // If b doesn't require grad, we still need *a* tensor b for the backward pass of a.
            // Clone it here if it wasn't cloned earlier.
            b.clone() 
        };

        let mut output_guard = result_tensor.data.write().map_err(|_| NeuraRustError::InternalError("Failed to lock output tensor data for writing".to_string()))?;
        output_guard.requires_grad = true; // Set requires_grad on the output
        let backward_context = DivBackward { 
            a_node: a_arc, 
            b_node: b_arc, 
            b_tensor_clone: b_clone_for_backward, // Pass the correctly obtained clone
            a_requires_grad, 
            b_requires_grad 
        };
        output_guard.grad_fn = Some(Arc::new(backward_context));
        println!("DivBackward grad_fn set for div result."); // Debug print
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    
    
    // Import testing utils
    use crate::utils::testing::check_tensor_near;

    // No local get_f32_data needed, use Tensor::get_f32_data()

    #[test]
    fn test_div_tensors_ok() {
        let a = Tensor::new(vec![10.0, 20.0, -30.0], vec![3]).unwrap();
        let b = Tensor::new(vec![2.0, 5.0, 10.0], vec![3]).unwrap();
        let result = div_op(&a, &b).unwrap();
        let result_data = result.get_f32_data().expect("Failed to get result data");
        assert_eq!(result_data, vec![5.0, 4.0, -3.0]);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_div_broadcasting() {
        let a = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![2.0, 4.0], vec![2]).unwrap();
        let result = div_op(&a, &b).unwrap();
        let result_data = result.get_f32_data().expect("Failed to get result data");
        assert_eq!(result_data, vec![5.0, 2.5, 10.0, 5.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_div_by_zero() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![0.0, 1.0], vec![2]).unwrap();
        let result = div_op(&a, &b);
        assert!(matches!(result, Err(NeuraRustError::DivisionByZero)));
    }

    // --- Autograd Tests ---
    #[test]
    fn test_div_backward_simple() {
        let a_data = vec![6.0, 10.0];
        let b_data = vec![2.0, 5.0];
        let shape = vec![2];
        let a = Tensor::new(a_data.clone(), shape.clone()).unwrap();
        let b = Tensor::new(b_data.clone(), shape.clone()).unwrap();
        a.set_requires_grad(true).unwrap();
        b.set_requires_grad(true).unwrap();

        let output = div_op(&a, &b).unwrap();
        
        let grad_output_data = vec![0.1, 0.2];
        let grad_output = Tensor::new(grad_output_data.clone(), shape.clone()).unwrap();

        output.backward(Some(grad_output)).expect("Backward pass failed");

        // Check grad_a
        let grad_a_contig = a.grad().unwrap().contiguous().unwrap();
        let expected_grad_a: Vec<f32> = grad_output_data.iter().zip(b_data.iter()).map(|(&g, &bi)| g / bi).collect();
        // [0.1/2.0, 0.2/5.0] = [0.05, 0.04]
        check_tensor_near(&grad_a_contig, &shape, &expected_grad_a, 1e-6);

        // Check grad_b
        let grad_b_contig = b.grad().unwrap().contiguous().unwrap();
        let expected_grad_b: Vec<f32> = grad_output_data.iter().zip(a_data.iter()).zip(b_data.iter())
            .map(|((&g, &ai), &bi)| g * (-ai / (bi * bi)))
            .collect(); 
        // [0.1 * (-6 / 4), 0.2 * (-10 / 25)] = [-0.15, -0.08]
        check_tensor_near(&grad_b_contig, &shape, &expected_grad_b, 1e-6);
    }

    #[test]
    fn test_div_backward_broadcast() {
        // Test case: a [2, 1] / b [2] -> output [2, 2]
        let a_data = vec![6.0, 10.0];
        let b_data = vec![2.0, 5.0];
        let a_shape = vec![2, 1];
        let b_shape = vec![2];
        let output_shape = vec![2, 2];

        let a = Tensor::new(a_data.clone(), a_shape.clone()).unwrap();
        let b = Tensor::new(b_data.clone(), b_shape.clone()).unwrap();
        a.set_requires_grad(true).unwrap();
        b.set_requires_grad(true).unwrap();

        let output = div_op(&a, &b).unwrap();
        
        let grad_output_data = vec![0.1, 0.2, 0.3, 0.4]; // Shape [2, 2]
        let grad_output = Tensor::new(grad_output_data.clone(), output_shape.clone()).unwrap();

        output.backward(Some(grad_output)).expect("Backward pass failed");

        // Check grad_a (needs reduction from [2, 2] to [2, 1])
        let grad_a_contig = a.grad().unwrap().contiguous().unwrap();
        // grad_a_unreduced = grad_output / b (broadcasted) = [[0.1/2, 0.2/5], [0.3/2, 0.4/5]] = [[0.05, 0.04], [0.15, 0.08]]
        let expected_grad_a = vec![0.09, 0.23]; // Sum columns: [0.05 + 0.04, 0.15 + 0.08]
        check_tensor_near(&grad_a_contig, &a_shape, &expected_grad_a, 1e-6);

        // Check grad_b (needs reduction from [2, 2] to [2])
        let grad_b_contig = b.grad().unwrap().contiguous().unwrap();
        // grad_b_unreduced = grad_output * (-a / b^2) (broadcasted)
        // -a / b^2 = [[-6/4, -6/25], [-10/4, -10/25]] = [[-1.5, -0.24], [-2.5, -0.4]]
        // mult = [[0.1*-1.5, 0.2*-0.24], [0.3*-2.5, 0.4*-0.4]] = [[-0.15, -0.048], [-0.75, -0.16]]
        // Sum rows: [-0.15 + -0.75, -0.048 + -0.16]
        let expected_grad_b = vec![-0.9, -0.208]; 
        check_tensor_near(&grad_b_contig, &b_shape, &expected_grad_b, 1e-6);
    }

    // Test division by zero in backward is tricky, grad check likely fails numerically anyway
    // #[test]
    // fn test_div_backward_with_zero_divisor() {
    //     println!("Skipping test_div_backward_with_zero_divisor for now.");
    // }
}
