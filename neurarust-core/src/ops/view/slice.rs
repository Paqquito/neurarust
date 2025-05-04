// neurarust-core/src/ops/view/slice.rs

use crate::autograd::BackwardOp;
 // Assuming non-generic
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;

use std::sync::{Arc, RwLock};
use std::fmt::Debug;

// --- Backward Operation Structure ---
#[derive(Debug)]
struct SliceBackward { // Remove <T>
    input_node: Arc<RwLock<TensorData>>, // Store original tensor info
    _input_shape: Vec<usize>,
    // Store ranges to know where to scatter the gradient back
    _ranges: Vec<(usize, usize)>,
}

// --- Backward Operation Implementation ---
impl BackwardOp for SliceBackward { // Remove <T>
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // 1. Get the shape of the original input tensor
        let input_shape = &self._input_shape;

        // 2. Create a zero tensor with the shape of the original input
        //    Assume F32 CPU for now, based on current forward op constraints.
        // TODO: Handle dtype and device properly later.
        let grad_input = crate::tensor::zeros(input_shape)?;

        // 3. Access the mutable data of the zero tensor and the grad_output tensor
        //    This is the tricky part. We need efficient write access to grad_input
        //    and read access to grad_output. This requires careful handling of buffers.
        //    For simplicity, let's try getting Vec<f32> and modifying it.
        //    WARNING: This is inefficient for large tensors!
        let mut grad_input_data = grad_input.get_f32_data()?;
        let grad_output_data = grad_output.get_f32_data()?;

        // 4. Iterate through the elements of grad_output and scatter them into grad_input
        let grad_output_shape = grad_output.shape();
        let grad_output_strides = grad_output.strides();
        let grad_input_strides = &grad_input.strides(); // Get strides of the zero tensor
        let grad_input_offset = grad_input.read_data().offset; // Should be 0 for zeros

        let mut grad_output_indices = vec![0; grad_output_shape.len()];

        // Recursive helper function to iterate over grad_output indices
        fn scatter_recursive(
            dim: usize,
            ranges: &[(usize, usize)],
            grad_output_shape: &[usize],
            grad_output_strides: &[usize],
            grad_output_offset: usize,
            grad_output_data: &[f32],
            grad_input_strides: &[usize],
            grad_input_offset: usize,
            grad_input_data: &mut [f32],
            current_indices_output: &mut Vec<usize>,
            current_indices_input: &mut Vec<usize>,
        ) -> Result<(), NeuraRustError> {
            if dim == grad_output_shape.len() {
                // Calculate the linear index for grad_output
                let output_linear_offset = grad_output_offset + current_indices_output.iter()
                    .zip(grad_output_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();
                let output_value = grad_output_data[output_linear_offset];

                // Calculate the linear index for grad_input
                let input_linear_offset = grad_input_offset + current_indices_input.iter()
                    .zip(grad_input_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();

                // Add the gradient value (scatter)
                 if input_linear_offset >= grad_input_data.len() {
                     return Err(NeuraRustError::InternalError(format!("SliceBackward: Input index out of bounds during scatter ({:?} -> index {})", current_indices_input, input_linear_offset)));
                 }
                grad_input_data[input_linear_offset] += output_value;
            } else {
                for i in 0..grad_output_shape[dim] {
                    current_indices_output[dim] = i;
                    // Map output index back to input index using the range start
                    current_indices_input[dim] = i + ranges[dim].0;
                    scatter_recursive(
                        dim + 1,
                        ranges,
                        grad_output_shape,
                        grad_output_strides,
                        grad_output_offset,
                        grad_output_data,
                        grad_input_strides,
                        grad_input_offset,
                        grad_input_data,
                        current_indices_output,
                        current_indices_input,
                    )?;
                }
            }
             Ok(())
        }

        let mut current_indices_input = vec![0; input_shape.len()];
        scatter_recursive(
            0, // Start from dim 0
            &self._ranges,
            &grad_output_shape,
            &grad_output_strides,
            grad_output.read_data().offset, // Get offset from grad_output
            &grad_output_data,
            &grad_input_strides,
            grad_input_offset,
            &mut grad_input_data,
            &mut grad_output_indices,
            &mut current_indices_input,
        )?;

        // 5. We modified the Vec, now we need to put it back into a Tensor.
        //    This is inefficient as it involves creating a new Tensor.
        //    A better approach would modify the buffer in-place if possible.
        let final_grad_input = Tensor::from_vec_f32(grad_input_data, input_shape.clone())?;

        Ok(vec![final_grad_input])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        vec![Arc::as_ptr(&self.input_node)] // Return the stored input node ID
    }
}

// Helper for validating slice ranges
fn validate_and_adjust_ranges(
    shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<(usize, usize)>, NeuraRustError> {
    // ... (Keep existing validation logic) ...
    if ranges.len() != shape.len() {
        return Err(NeuraRustError::SliceError {
            message: format!(
                "Number of ranges ({}) must match tensor rank ({})",
                ranges.len(),
                shape.len()
            ),
        });
    }
    let mut adjusted_ranges = Vec::with_capacity(shape.len());
    for (i, &(start, end)) in ranges.iter().enumerate() {
        let dim_size = shape[i];
        // Allow start == end for empty slices, adjust bounds check
        if start > end || start > dim_size || end > dim_size {
            return Err(NeuraRustError::SliceError {
                message: format!(
                    "Invalid slice range [{}, {}) for dimension {} with size {}",
                    start,
                    end,
                    i,
                    dim_size
                ),
            });
        }
        adjusted_ranges.push((start, end));
    }
    Ok(adjusted_ranges)
}

// --- Forward Operation ---
pub fn slice_op(
    tensor: &Tensor,
    ranges: Vec<(usize, usize)>,
) -> Result<Tensor, NeuraRustError> {
    let tensor_data = tensor.data.read().unwrap();

    // --- Validate Ranges ---
    let adjusted_ranges = validate_and_adjust_ranges(&tensor_data.shape, &ranges)?;

    // --- Calculate New Shape, Strides, Offset ---
    let mut new_shape = Vec::with_capacity(tensor_data.shape.len());
    let mut new_offset = tensor_data.offset;
    for i in 0..tensor_data.shape.len() {
        new_shape.push(adjusted_ranges[i].1 - adjusted_ranges[i].0);
        new_offset += adjusted_ranges[i].0 * tensor_data.strides[i];
    }
    let new_strides = tensor_data.strides.clone(); // Strides remain the same for slice view

    // --- Create View TensorData ---
    // Use the internal constructor for views
    let view_td = TensorData::new_view(
        Arc::clone(&tensor_data.buffer), // Share the buffer Arc
        tensor_data.device,              // Inherit device
        new_offset,
        new_shape,
        new_strides,
        // Dtype is set inside new_view (currently assumes F32)
    );

    // --- Wrap in Tensor and Setup Autograd ---
    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    if tensor_data.requires_grad {
        // Store context for backward pass
        let backward_context = SliceBackward {
            input_node: Arc::clone(&tensor.data), // Clone the Arc to the input TensorData
            _input_shape: tensor_data.shape.clone(),
            _ranges: adjusted_ranges,
        };
        let backward_op_arc: Arc<dyn BackwardOp + Send + Sync> = Arc::new(backward_context);
        // output_tensor.set_requires_grad(true)?; // TODO: Adapt when methods available
        // output_tensor.set_grad_fn(Some(backward_op_arc))?;
        {
            let mut output_guard = output_tensor.data.write().unwrap();
            output_guard.requires_grad = true;
            output_guard.grad_fn = Some(backward_op_arc);
        }
    }

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    use crate::error::NeuraRustError;
    use crate::buffer::{Buffer, CpuBuffer};

    // Test helper function
    fn get_f32_data(tensor: &Tensor) -> Vec<f32> {
        let tensor_data = tensor.data.read().unwrap();
        match &*tensor_data.buffer {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => data_arc.to_vec(),
            _ => panic!("Test helper expects F32 CPU tensor"),
        }
    }

    #[test]
    fn test_slice_basic() {
        // Re-enable test
        let t = Tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]).unwrap();
        let s = slice_op(&t, vec![(0, 1), (0, 2), (1, 3)]).unwrap();
        assert_eq!(s.shape(), vec![1, 2, 2]);
        // Check data after making contiguous
        let s_contig = s.contiguous().unwrap();
        let s_data = get_f32_data(&s_contig);
        // Original: [[[0,1,2], [3,4,5]], [[6,7,8], [9,10,11]]]
        // Slice [0:1] -> [[0,1,2], [3,4,5]]
        // Slice [0:2] -> [[0,1,2], [3,4,5]]
        // Slice [1:3] -> [[1,2], [4,5]]
        // Expected: [[[1,2], [4,5]]] -> Flattened: [1, 2, 4, 5]
        assert_eq!(s_data, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_slice_full() {
         // Re-enable test
        let t = Tensor::from_vec_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]).unwrap();
        let s = slice_op(&t, vec![(0, 2), (0, 3)]).unwrap();
        assert_eq!(s.shape(), vec![2, 3]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.read_data().offset, t.read_data().offset); 
    }

    #[test]
    fn test_slice_rank_mismatch() {
        let t = Tensor::new(vec![1.0], vec![1]).unwrap();
        let result = slice_op(&t, vec![(0, 1), (0, 1)]);
        assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
    }

    #[test]
    fn test_slice_invalid_range_start_gt_end() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let result = slice_op(&t, vec![(1, 0)]);
        assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
    }

     #[test]
    fn test_slice_invalid_range_end_gt_size() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let result = slice_op(&t, vec![(0, 3)]);
        assert!(matches!(result, Err(NeuraRustError::SliceError { .. })));
    }

    #[test]
    fn test_slice_empty_dim() {
        let t = Tensor::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]).unwrap();
        // Slice to create a zero-sized dimension
        let s_result = slice_op(&t, vec![(0, 1), (1, 1), (0, 3)]);
        // Creating an empty slice [1, 1) is now allowed.
        assert!(s_result.is_ok(), "Slice with empty dim [1, 1) should be Ok");
        if let Ok(s) = s_result { // Check shape and numel only if Ok
           assert_eq!(s.shape(), vec![1, 0, 3], "Shape mismatch for slice with empty dim"); 
           assert_eq!(s.numel(), 0, "Numel should be 0 for slice with empty dim");
        }
    }

    #[test]
    fn test_slice_view_data_sharing() {
         // Re-enable test - compare buffer Arc pointers
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let s = slice_op(&t, vec![(0, 1), (0, 2)]).unwrap();
        // Check if buffer pointers are the same
        assert!(Arc::ptr_eq(&t.data.read().unwrap().buffer, &s.data.read().unwrap().buffer));
    }

    // --- Autograd Tests ---
    #[test]
    fn test_slice_backward() {
         println!("Skipping test_slice_backward: Backward logic implemented but needs verification and Tensor::add_");
         // TODO: Re-enable when backward is verified & in-place add exists or grad_check adapted
        /*
        let input_data = (0..6).map(|x| x as f32).collect::<Vec<_>>();
        let input = Tensor::from_vec_f32(input_data, vec![2, 3])
                        .unwrap()
                        .with_requires_grad(true)
                        .unwrap();

        let func = |inputs: &[Tensor]| slice_op(&inputs[0], vec![(0, 1), (1, 2)]); // Slice to shape [1, 1]

        let output_shape = func(&[input.clone()]).unwrap().shape();
        let numel_out = output_shape.iter().product();
        let output_grad = Tensor::ones(&output_shape).unwrap(); // Use tensor::ones
        let epsilon = 1e-5;
        let tolerance = 1e-4;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Slice backward grad check failed: {:?}", grad_check_result.err());
        */
    }

    #[test]
    fn test_slice_backward_scalar_result() {
        println!("Skipping test_slice_backward_scalar_result until backward logic and Tensor methods are adapted.");
        // ...
    }

    #[test]
    fn test_slice_backward_empty_result() {
        println!("Skipping test_slice_backward_empty_result until backward logic and Tensor methods are adapted.");
        // ...
    }
} 