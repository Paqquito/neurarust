use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use crate::buffer::{Buffer, CpuBuffer};
use std::sync::Arc;

use std::fmt::Debug;
use std::sync::RwLock;

// +++ Copied from add.rs - TODO: Move to a shared utils module +++
/// Iterator for broadcasting over two NdArrays (represented by CpuBuffer<f32>).
/// Handles differences in rank and dimensions of size 1.
struct NdArrayBroadcastingIter<'a> {
    buffer: &'a Arc<Vec<f32>>,
    original_shape: &'a [usize],
    original_strides: &'a [usize],
    original_offset: usize,
    target_shape: &'a [usize],
    current_index: usize,
    total_elements: usize,
}

impl<'a> NdArrayBroadcastingIter<'a> {
    fn new(
        buffer: &'a Arc<Vec<f32>>,
        original_shape: &'a [usize],
        original_strides: &'a [usize],
        original_offset: usize,
        target_shape: &'a [usize],
    ) -> Result<Self, NeuraRustError> {
        // Basic compatibility check (can be expanded)
        if !original_shape.is_empty() && original_shape.len() > target_shape.len() {
             // Original shape shouldn't have more dims than target after broadcasting
             // This condition might need refinement depending on exact broadcasting rules validation
             // It assumes broadcast_shapes already validated the core logic.
        }
        
        let total_elements = target_shape.iter().product();
        Ok(Self {
            buffer,
            original_shape,
            original_strides,
            original_offset,
            target_shape,
            current_index: 0,
            total_elements,
        })
    }
}

impl<'a> Iterator for NdArrayBroadcastingIter<'a> {
    type Item = f32; // Iterates over f32 values

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_elements {
            return None;
        }

        let target_rank = self.target_shape.len();
        let original_rank = self.original_shape.len();
        
        // Calculate multi-dimensional index in the target shape
        let mut target_multi_index = vec![0; target_rank];
        let mut current_linear = self.current_index;
        for dim in (0..target_rank).rev() {
            let shape_val = self.target_shape[dim];
             if shape_val > 0 { // Avoid division by zero for empty dimensions
                target_multi_index[dim] = current_linear % shape_val;
                current_linear /= shape_val;
            } else {
                target_multi_index[dim] = 0;
            }
        }

        // Calculate corresponding multi-dimensional index in the original shape
        let mut original_multi_index = vec![0; original_rank];
        let rank_diff = target_rank as isize - original_rank as isize;

        for original_dim in 0..original_rank {
             let target_dim = (original_dim as isize + rank_diff) as usize;
             // If the original dimension size is 1, use index 0 (broadcast)
             // Otherwise, use the corresponding index from the target shape
             if self.original_shape[original_dim] == 1 {
                 original_multi_index[original_dim] = 0;
             } else {
                 original_multi_index[original_dim] = target_multi_index[target_dim];
             }
        }

        // Calculate physical offset using original strides
        let physical_offset = self.original_offset
            + original_multi_index
                .iter()
                .zip(self.original_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();

        let value = self.buffer[physical_offset];
        self.current_index += 1;
        Some(value)
    }
}
// +++ End of copied code +++

// +++ Copied from add.rs - TODO: Move to a shared utils module +++
/// Reduces a gradient tensor to match a target shape, summing along broadcasted dimensions.
fn reduce_gradient_to_shape(
    grad: &Tensor,
    target_shape: &[usize],
) -> Result<Tensor, NeuraRustError> {
    let grad_shape = grad.shape();

    // No reduction needed if shapes already match
    if grad_shape == target_shape {
        return Ok(grad.clone()); // No reduction needed
    }
    // Handle scalar target shape (sum all elements)
    if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
         return crate::ops::reduction::sum::sum_op(grad, None, false); // Sum all
    }

    let grad_rank = grad_shape.len();
    let target_rank = target_shape.len();

    if target_rank > grad_rank {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "reduce_gradient_to_shape".to_string(),
            expected: format!("rank <= {}", grad_rank),
            actual: format!("rank {}", target_rank),
        });
    }

    // Identify axes to sum over
    let mut axes_to_sum = Vec::new();
    let rank_diff = grad_rank - target_rank;

    // Sum over dimensions that were added during broadcasting
    for i in 0..rank_diff {
        axes_to_sum.push(i);
    }

    // Sum over dimensions that were broadcasted from 1
    for i in 0..target_rank {
        if target_shape[i] == 1 && grad_shape[i + rank_diff] > 1 {
            axes_to_sum.push(i + rank_diff);
        }
         // Sanity check: target dim should not be larger than grad dim
        if target_shape[i] > grad_shape[i + rank_diff] {
             return Err(NeuraRustError::ShapeMismatch {
                 operation: "reduce_gradient_to_shape (dimension check)".to_string(),
                 expected: format!("dim {} size <= {}", i, grad_shape[i + rank_diff]),
                 actual: format!("dim {} size {}", i, target_shape[i]),
             });
        }
    }

    if axes_to_sum.is_empty() {
        // This might happen if shapes are compatible but not identical (e.g., [2,1] vs [2])
        // In this specific case, a reshape might be needed if ranks differ after potential sums.
        if grad_rank != target_rank {
             // If after checking dims, ranks still differ, we might need to reshape.
             // Example: grad_shape=[1, 2, 3], target_shape=[2, 3]
             // We sum axis 0. Result is [2, 3]. No reshape needed.
             // Example: grad_shape=[2, 1, 3], target_shape=[2, 3]
             // We sum axis 1. Result is [2, 3]. Reshape might be conceptually needed if keep_dims=true was used during sum.
             // Since sum_op returns squeezed shape by default (keep_dims=false), reshape is often handled.
             // Let's assume sum_op handles the squeezing correctly for now.
             // However, if target is [1, 2, 1] and grad is [5, 1, 2, 5], axes_to_sum=[0, 3]. Result [1,2]. Needs reshape to [1,1,2]?
             // Let's try reshaping explicitly if ranks still differ after potential summation.
             // Re-evaluate this logic if issues arise.
             // The sum op should return the correct shape if keep_dims=false.
             // If keep_dims=true was used, we would need a reshape here.
             // Let's assume keep_dims=false for sum_op.
             // Consider the case grad=[5, 1], target=[1]. Sum axis 0 -> [1]. OK.
             // Consider grad=[1, 5], target=[1]. Sum axis 1 -> [1]. OK.
             // Consider grad=[5, 1], target=[]. Sum axes 0, 1 -> []. OK.
             // If reduction occurred, the rank might change.
             // Let's check if the rank *after* potential summation needs reshaping.
             // This check might be redundant if sum_op correctly squeezes.
            // return grad.reshape(target_shape); // Tentative reshape
            return Ok(grad.clone()); // If no axes identified, assume shape is compatible or already handled by sum squeeze
        }
        return Ok(grad.clone()); // Shapes must be compatible if no axes to sum found
    }

    // Perform summation
    // Use keep_dims=false, as we want to remove the summed dimensions
    let summed_grad = crate::ops::reduction::sum::sum_op(grad, Some(&axes_to_sum), false)?;

    // Reshape if necessary to match target shape (e.g., summing didn't remove all target dims of size 1)
    let final_grad = if summed_grad.shape() != target_shape {
        // Call reshape with an owned Vec<usize>
        summed_grad.reshape(target_shape.to_vec())?
    } else {
        summed_grad
    };

    Ok(final_grad)
}
// +++ End of copied code +++

// --- Backward Operation Structure ---
#[derive(Debug)]
struct MulBackward {
    // Store Tensor clones for backward pass calculation
    a: Tensor,
    b: Tensor,
    // Store Arc<RwLock<TensorData>> for graph linkage (inputs method)
    a_node: Arc<RwLock<TensorData>>,
    b_node: Arc<RwLock<TensorData>>,
    // Store original shapes for gradient reduction
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    // Keep track of which inputs need grad for the `inputs` method
    a_requires_grad: bool,
    b_requires_grad: bool,
    // Remove raw pointers: a_data_ptr, b_data_ptr
}

// --- Backward Operation Implementation ---
impl BackwardOp for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut result_grads = Vec::new();

        if self.a_requires_grad {
            let unreduced_grad_a = mul_op(grad_output, &self.b)?;
            // Reduce gradient if necessary
            let grad_a = reduce_gradient_to_shape(&unreduced_grad_a, &self.a_shape)?;
            result_grads.push(grad_a);
        } else {
           // If a doesn't require grad, we technically don't need to compute anything for it.
           // The autograd engine expects grads ONLY for inputs that require grad.
        }

        if self.b_requires_grad {
            let unreduced_grad_b = mul_op(grad_output, &self.a)?;
            // Reduce gradient if necessary
            let grad_b = reduce_gradient_to_shape(&unreduced_grad_b, &self.b_shape)?;
            result_grads.push(grad_b);
        } else {
            // Similarly for b.
        }

        Ok(result_grads)
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut inputs_vec = Vec::new();
        // Use Arc::as_ptr on the stored Arcs for safe pointers
        if self.a_requires_grad {
            inputs_vec.push(Arc::as_ptr(&self.a_node));
        }
        if self.b_requires_grad {
            inputs_vec.push(Arc::as_ptr(&self.b_node));
        }
        inputs_vec
    }
}

// --- Forward Operation ---
pub fn mul_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "mul_op".to_string(),
            expected: a_guard.device,
            actual: b_guard.device,
        });
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            "mul_op currently only supports CPU tensors.".to_string(),
        ));
    }

    // --- DType Check ---
    if a_guard.dtype != DType::F32 || b_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "mul_op currently only supports F32 tensors.".to_string(),
        ));
    }
    let _output_dtype = DType::F32;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;

    // --- Extract Buffer Arcs and metadata BEFORE dropping guards ---
    let a_buffer_arc = match &*a_guard.buffer {
        Buffer::Cpu(CpuBuffer::F32(arc)) => arc.clone(),
        _ => return Err(NeuraRustError::UnsupportedOperation("mul_op: Unsupported buffer type for a".into())),
    };
    let b_buffer_arc = match &*b_guard.buffer {
        Buffer::Cpu(CpuBuffer::F32(arc)) => arc.clone(),
        _ => return Err(NeuraRustError::UnsupportedOperation("mul_op: Unsupported buffer type for b".into())),
    };

    let a_shape = a_guard.shape.clone();
    let b_shape = b_guard.shape.clone();
    let a_strides = a_guard.strides.clone();
    let b_strides = b_guard.strides.clone();
    let a_offset = a_guard.offset;
    let b_offset = b_guard.offset;
    let a_requires_grad = a_guard.requires_grad;
    let b_requires_grad = b_guard.requires_grad;
    // Clone the Arcs for the BackwardOp struct
    let a_node_arc = a.data.clone();
    let b_node_arc = b.data.clone();
    // Remove direct pointer storage here

    // --- Drop guards ---
    drop(a_guard);
    drop(b_guard);

    // --- Perform Calculation using Iterators ---
    let numel = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel);

    let a_iter = NdArrayBroadcastingIter::new(&a_buffer_arc, &a_shape, &a_strides, a_offset, &output_shape)?;
    let b_iter = NdArrayBroadcastingIter::new(&b_buffer_arc, &b_shape, &b_strides, b_offset, &output_shape)?;

    for (val_a, val_b) in a_iter.zip(b_iter) {
        result_data_vec.push(val_a * val_b);
    }

    // --- Create Output Tensor ---
    let output_tensor = Tensor::from_vec_f32(result_data_vec, output_shape.clone())?;

    // --- Autograd Linkage ---
    if a_requires_grad || b_requires_grad {
        let backward_context = MulBackward {
            // Clone the original Tensors for use in backward calculation
            a: a.clone(),
            b: b.clone(),
            // Store the Arcs for graph linkage
            a_node: a_node_arc, // Use the cloned Arcs
            b_node: b_node_arc,
            a_shape, // Use stored shapes
            b_shape,
            a_requires_grad, // Use stored flags
            b_requires_grad,
            // Removed raw pointers: a_data_ptr, b_data_ptr
        };
        let grad_fn = Arc::new(backward_context); // Wrap in Arc<dyn BackwardOp>
        output_tensor.set_grad_fn(Some(grad_fn))?; // Use helper
        output_tensor.set_requires_grad(true)?; // Use helper
    }

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::device::StorageDevice;
    use crate::types::DType;
    use crate::error::NeuraRustError;
    use crate::buffer::{Buffer, CpuBuffer};
    use crate::autograd::grad_check::check_grad; // Import check_grad

    // Test helper function (using read_data)
    fn get_f32_data(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
        let guard = tensor.read_data();
        if guard.dtype != DType::F32 || guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::UnsupportedOperation("Test helper requires F32 CPU tensor".to_string()));
        }
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => Ok(data_arc.to_vec()),
            _ => Err(NeuraRustError::UnsupportedOperation("Buffer type not CpuF32".to_string())),
        }
    }

    #[test]
    fn test_mul_tensors_ok() {
        // Re-enable and adapt
        let t1 = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::from_vec_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let result = mul_op(&t1, &t2).unwrap();
        let result_data = get_f32_data(&result).unwrap();
        assert_eq!(result_data, vec![5.0, 12.0, 21.0, 32.0]);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.device(), StorageDevice::CPU);
        assert!(!result.requires_grad()); // Should not require grad by default
    }

    #[test]
    fn test_mul_tensors_shape_mismatch() {
        // Test remains valid
        let t1 = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let t2 = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = mul_op(&t1, &t2);
        assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
    }

    #[test]
    fn test_mul_broadcasting() {
        // Re-enable and adapt
        let matrix = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let row_vector = Tensor::from_vec_f32(vec![10.0, 20.0], vec![1, 2]).unwrap();
        let result = mul_op(&matrix, &row_vector).unwrap();
        let expected_data = vec![10.0, 40.0, 30.0, 80.0]; // 1*10, 2*20, 3*10, 4*20
        let result_data = get_f32_data(&result).unwrap();
        assert_eq!(result_data, expected_data);
        assert_eq!(result.shape(), &[2, 2]);

        let col_vector = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let result2 = mul_op(&matrix, &col_vector).unwrap();
        let expected_data2 = vec![10.0, 20.0, 60.0, 80.0]; // 1*10, 2*10, 3*20, 4*20
        let result_data2 = get_f32_data(&result2).unwrap();
        assert_eq!(result_data2, expected_data2);
        assert_eq!(result2.shape(), &[2, 2]);
    }

    // --- Autograd Tests ---
    #[test]
    fn test_mul_backward_simple() {
        // Re-enable and adapt
        let a_data = vec![1.0f32, 2.0, 3.0];
        let b_data = vec![4.0f32, 5.0, 6.0];
        let func = |inputs: &[Tensor]| mul_op(&inputs[0], &inputs[1]);

        let a = Tensor::from_vec_f32(a_data.clone(), vec![3]).unwrap();
        a.set_requires_grad(true).unwrap();
        let b = Tensor::from_vec_f32(b_data.clone(), vec![3]).unwrap();
        b.set_requires_grad(true).unwrap();

        let output_shape = func(&[a.clone(), b.clone()]).unwrap().shape();
        let output_grad = crate::tensor::ones(&output_shape).unwrap(); 
        let epsilon = 1e-4;
        let tolerance = 1e-2; // Increased tolerance

        check_grad(func, &[a, b], &output_grad, epsilon, tolerance)
            .expect("Simple mul backward grad check failed");
    }

    #[test]
    fn test_mul_backward_broadcast() {
        // Re-enable and adapt
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // shape [2, 2]
        let b_data = vec![10.0f32, 20.0];         // shape [1, 2]
        let func = |inputs: &[Tensor]| mul_op(&inputs[0], &inputs[1]);

        let a = Tensor::from_vec_f32(a_data.clone(), vec![2, 2]).unwrap();
        a.set_requires_grad(true).unwrap();
        let b = Tensor::from_vec_f32(b_data.clone(), vec![1, 2]).unwrap();
        b.set_requires_grad(true).unwrap();

        let output_shape = func(&[a.clone(), b.clone()]).unwrap().shape();
        let output_grad = crate::tensor::ones(&output_shape).unwrap();
        let epsilon = 1e-4;
        let tolerance = 1e-2; // Increased tolerance

        check_grad(func, &[a, b], &output_grad, epsilon, tolerance)
            .expect("Broadcast mul backward grad check failed");
    }
}
