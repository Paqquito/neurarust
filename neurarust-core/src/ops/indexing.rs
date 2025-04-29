use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::calculate_strides;
use std::ops::{Range, AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::Zero;
use std::collections::HashMap;

/// Represents a slice for a single dimension.
/// Used within the `Tensor::slice` method.
#[derive(Debug, Clone)]
pub enum TensorSlice {
    /// Select a single index along the dimension. Results in a tensor with one fewer dimension.
    Index(usize),
    /// Select a range of indices (exclusive end). Keeps the dimension.
    Range(Range<usize>),
    /// Select the entire dimension (equivalent to `Range(0..dim_size)`). Keeps the dimension.
    Full,
    // TODO: Potentially add Step(Range<usize>, usize), NewAxis, Ellipsis later?
}

/// Backward operation for the `slice_op` function.
/// Stores information needed to scatter the gradient back to the original tensor.
#[derive(Debug)]
struct SliceBackward<T> {
    /// The shape of the original input tensor.
    original_shape: Vec<usize>,
    /// The slices that were used in the forward pass.
    slices: Vec<TensorSlice>,
    /// Weak reference to the original input tensor data.
    input_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SliceBackward<T>
where
    // Keep bounds minimal: AddAssign + Clone for accumulation,
    // Default + Zero for init, Debug + 'static standard.
    T: AddAssign + Clone + Default + Debug + 'static + Zero,
{
    /// Performs the backward pass for the slicing operation.
    /// Scatters the `upstream_grad` (gradient of the sliced tensor) back into 
    /// a zero tensor shaped like the original input tensor, according to the 
    /// slices used in the forward pass.
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            let input_ptr = Rc::as_ptr(&input_rc);

            // --- 1. Create zero data for scattered gradient --- 
            // Initialize a data vector of zeros with the shape of the *original* input tensor.
            let original_numel = self.original_shape.iter().product::<usize>();
            let mut scattered_data = vec![T::zero(); original_numel];
            // Calculate strides for the original shape to map coordinates to the flat index.
            let original_strides = calculate_strides(&self.original_shape);
            
            // --- 2. Iterate through upstream_grad and scatter values --- 
            let upstream_shape = upstream_grad.shape();
            let upstream_numel = upstream_grad.numel();
            let upstream_strides = calculate_strides(&upstream_shape);
            let upstream_data = upstream_grad.data();

            // Iterate through each element of the upstream gradient (the gradient of the output slice).
            for upstream_linear_idx in 0..upstream_numel {
                // Convert the linear index of the upstream gradient to its multi-dimensional coordinates
                // relative to the *output slice's* shape.
                let output_coords = crate::tensor::utils::index_to_coord(upstream_linear_idx, &upstream_strides, &upstream_shape);
                
                // Construct the corresponding multi-dimensional coordinates in the *original* input tensor
                // based on the slices used in the forward pass.
                let mut original_coords = vec![0; self.original_shape.len()];
                let mut current_output_coord_idx = 0; // Track which output dimension we are using
                for original_dim_idx in 0..self.original_shape.len() {
                    match &self.slices[original_dim_idx] {
                        TensorSlice::Index(fixed_idx) => {
                            // If the original dimension was indexed, use that fixed index.
                            original_coords[original_dim_idx] = *fixed_idx;
                            // This dimension does not exist in the output, so don't increment current_output_coord_idx.
                        },
                        TensorSlice::Range(range) => {
                            // If the original dimension was sliced with a Range, map the output coordinate back.
                            let output_coord = output_coords[current_output_coord_idx];
                            original_coords[original_dim_idx] = range.start + output_coord; // Add the range start offset.
                            current_output_coord_idx += 1;
                        },
                        TensorSlice::Full => {
                            // If the original dimension used Full slice, map the output coordinate directly.
                            let output_coord = output_coords[current_output_coord_idx];
                            original_coords[original_dim_idx] = output_coord; // No offset for Full (start=0)
                            current_output_coord_idx += 1;
                        },
                    }
                }

                // Convert the calculated original multi-dimensional coordinates to a flat index 
                // within the `scattered_data` vector.
                let mut original_linear_idx = 0;
                for dim_idx in 0..self.original_shape.len() {
                    original_linear_idx += original_coords[dim_idx] * original_strides[dim_idx];
                }

                // --- 3. Add the upstream gradient value to the scattered gradient tensor --- 
                // Clone the gradient value (needed if T is not Copy).
                let grad_val = upstream_data[upstream_linear_idx].clone(); 
                // Add the gradient value to the appropriate location in the zero tensor.
                scattered_data[original_linear_idx] += grad_val; // Requires T: AddAssign
            }
            
            // Create the final scattered gradient Tensor using the populated data and original shape.
            let scattered_grad_tensor = Tensor::new(scattered_data, self.original_shape.clone());

            // --- 4. Accumulate manually --- 
            // Add the calculated scattered gradient to the gradient map for the original input tensor.
            gradients.entry(input_ptr)
                .and_modify(|existing_grad| {
                    // Manual element-wise addition instead of Tensor += &Tensor
                    assert_eq!(existing_grad.shape(), scattered_grad_tensor.shape(),
                               "Gradient shape mismatch during slice accumulation");
                    let mut existing_data = existing_grad.data_mut();
                    let scattered_data_ref = scattered_grad_tensor.data(); // Borrow data
                    existing_data.iter_mut()
                        .zip(scattered_data_ref.iter())
                        // Use clone here as AddAssign might consume or elements aren't Copy
                        .for_each(|(e, s)| *e += s.clone()); // Requires T: AddAssign + Clone
                })
                .or_insert(scattered_grad_tensor);

        } else {
            // This should ideally not happen if the graph is managed correctly.
            eprintln!("SliceBackward::backward: Input tensor weak reference expired.");
        }
    }

    /// Returns a weak reference to the input tensor.
    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}

/// Performs the forward slicing operation.
/// Creates a new tensor containing elements selected by the slices.
/// (Internal function called by `Tensor::slice`)
pub fn slice_op<T>(input: &Tensor<T>, slices: &[TensorSlice]) -> Tensor<T>
where
    T: Clone + Debug + Default + Zero + AddAssign + 'static,
{
    let input_shape = input.shape();
    // Ensure the number of slice specifiers matches the input tensor dimensions.
    assert_eq!(input_shape.len(), slices.len(), 
        "Number of slices ({}) must match tensor dimensions ({})", 
        slices.len(), input_shape.len());

    // --- 1. Calculate output shape and gather slice details --- 
    let mut output_shape = Vec::with_capacity(input_shape.len());
    // Store (start_index, end_index, is_indexed_dimension) for each original dimension.
    let mut slice_details = Vec::with_capacity(input_shape.len()); 
    // Track which original dim corresponds to which output dim (unused for now, kept for potential future use)
    // let mut output_dim_mapping = Vec::new(); 
    // let mut current_output_dim = 0;

    for (dim_idx, slice) in slices.iter().enumerate() {
        let dim_size = input_shape[dim_idx];
        match slice {
            TensorSlice::Index(idx) => {
                // Ensure index is within bounds.
                assert!(*idx < dim_size, "Index {} out of bounds for dimension {} with size {}", idx, dim_idx, dim_size);
                // This dimension is removed in the output. Store the fixed index as start/end.
                slice_details.push((*idx, *idx + 1, true)); 
            },
            TensorSlice::Range(range) => {
                // Handle range slice.
                let start = range.start;
                let end = range.end.min(dim_size); // Clamp end to dimension size.
                assert!(start <= end, "Slice range start ({}) must be <= end ({}) for dimension {}", start, end, dim_idx);
                assert!(end <= dim_size, "Slice range end ({}) out of bounds for dimension {} with size {}", end, dim_idx, dim_size);
                let len = end - start;
                // Add the length of the slice to the output shape.
                output_shape.push(len);
                slice_details.push((start, end, false));
                // output_dim_mapping.push(dim_idx);
                // current_output_dim += 1;
            },
            TensorSlice::Full => {
                // Handle full slice (equivalent to 0..dim_size).
                let start = 0;
                let end = dim_size;
                // Add the full dimension size to the output shape.
                output_shape.push(dim_size);
                slice_details.push((start, end, false));
                // output_dim_mapping.push(dim_idx);
                // current_output_dim += 1;
            },
        }
    }

    // --- 2. Calculate output data --- 
    let output_numel = output_shape.iter().product::<usize>();
    let mut output_data = Vec::with_capacity(output_numel);
    
    let input_td = input.borrow_tensor_data();
    let input_strides = calculate_strides(&input_shape);
    // Calculate strides for the output shape to map linear output index to coordinates.
    let output_strides = calculate_strides(&output_shape);

    // Iterate through each element of the output tensor.
    for output_linear_idx in 0..output_numel {
        // Convert the linear index of the output to its multi-dimensional coordinates.
        let output_coords = crate::tensor::utils::index_to_coord(output_linear_idx, &output_strides, &output_shape);
        
        // Construct the corresponding multi-dimensional coordinates in the *input* tensor.
        let mut input_coords = vec![0; input_shape.len()];
        let mut current_output_coord_idx = 0; // Tracks the current dimension index in the output coordinates.
        for input_dim_idx in 0..input_shape.len() {
            let (start, _end, is_indexed) = slice_details[input_dim_idx];
            if is_indexed {
                // If the input dimension was fixed by Index, use the stored index.
                input_coords[input_dim_idx] = start; 
            } else {
                // If the input dimension was sliced by Range or Full, get the corresponding output coordinate.
                let output_coord = output_coords[current_output_coord_idx];
                // Map the output coordinate back to the input coordinate space by adding the start offset.
                input_coords[input_dim_idx] = start + output_coord; 
                current_output_coord_idx += 1; // Move to the next dimension in the output coordinates.
            }
        }

        // Convert the calculated input multi-dimensional coordinates to a flat index 
        // using the input tensor's strides.
        let mut input_linear_idx = 0;
        for dim_idx in 0..input_shape.len() {
            input_linear_idx += input_coords[dim_idx] * input_strides[dim_idx];
        }
        
        // Copy the data element from the input tensor to the output data vector.
        // Clone is necessary as T might not be Copy.
        output_data.push(input_td.data[input_linear_idx].clone());
    }

    drop(input_td); // Release borrow on input data.

    // --- 3. Create output tensor and set up autograd --- 
    let result = Tensor::new(output_data, output_shape);
    // If the original input tensor requires gradients, set up the backward pass.
    if input.requires_grad() {
        result.set_requires_grad(true);
        // Create the backward operation struct, storing necessary context.
        let grad_fn = SliceBackward {
            original_shape: input_shape.clone(),
            slices: slices.to_vec(), // Clone slices needed for backward.
            input_ref: input.get_weak_ref(),
            _phantom: PhantomData,
        };
        // Store the backward function in the result tensor.
        result.borrow_tensor_data_mut().grad_fn = Some(Rc::new(grad_fn));
    }
    result
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use std::ops::AddAssign;
    use num_traits::Zero;
    use std::fmt::Debug;
    use std::collections::HashMap;
    use std::rc::Rc;

    // Helper to check tensor data and shape
    fn check_tensor<T: PartialEq + Debug + Clone>(tensor: &Tensor<T>, expected_data: &[T], expected_shape: &[usize]) {
        assert_eq!(tensor.data().to_vec(), expected_data, "Tensor data mismatch");
        assert_eq!(tensor.shape(), expected_shape, "Tensor shape mismatch");
    }

    // --- Forward Tests --- 

    #[test]
    fn test_slice_index_1d() {
        let t = Tensor::new(vec![10, 20, 30], vec![3]);
        let result = t.slice(&[TensorSlice::Index(1)]);
        check_tensor(&result, &[20], &[]); // Indexing reduces dimension
    }

    #[test]
    fn test_slice_range_1d() {
        let t = Tensor::new(vec![10, 20, 30, 40, 50], vec![5]);
        let result = t.slice(&[TensorSlice::Range(1..4)]);
        check_tensor(&result, &[20, 30, 40], &[3]);
    }

    #[test]
    fn test_slice_full_1d() {
        let t = Tensor::new(vec![10, 20, 30], vec![3]);
        let result = t.slice(&[TensorSlice::Full]);
        check_tensor(&result, &[10, 20, 30], &[3]);
    }

    #[test]
    fn test_slice_2d_row() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = t.slice(&[TensorSlice::Index(1), TensorSlice::Full]);
        check_tensor(&result, &[4, 5, 6], &[3]);
    }

    #[test]
    fn test_slice_2d_col() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = t.slice(&[TensorSlice::Full, TensorSlice::Index(1)]);
        check_tensor(&result, &[2, 5], &[2]);
    }

    #[test]
    fn test_slice_2d_element() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = t.slice(&[TensorSlice::Index(1), TensorSlice::Index(2)]);
        check_tensor(&result, &[6], &[]); // Scalar result
    }

    #[test]
    fn test_slice_2d_submatrix() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]);
        // Get [[5, 6], [8, 9]]
        let result = t.slice(&[TensorSlice::Range(1..3), TensorSlice::Range(1..3)]);
        check_tensor(&result, &[5, 6, 8, 9], &[2, 2]);
    }

    #[test]
    #[should_panic(expected = "Index 3 out of bounds")]
    fn test_slice_index_out_of_bounds() {
        let t = Tensor::new(vec![10, 20, 30], vec![3]);
        t.slice(&[TensorSlice::Index(3)]);
    }

    #[test]
    #[should_panic(expected = "Number of slices (1) must match tensor dimensions (2)")]
    fn test_slice_wrong_ndim() {
        let t = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        t.slice(&[TensorSlice::Full]);
    }
    
    // --- Backward Tests --- 
    
    // Helper to create tensor requiring grad
    fn create_grad_tensor<T: Clone + Debug + Default + Zero + AddAssign + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let t = Tensor::new(data, shape);
        t.set_requires_grad(true);
        t
    }

    #[test]
    fn test_slice_backward_1d_range() {
        let t = create_grad_tensor::<f32>(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]);
        let sliced = t.slice(&[TensorSlice::Range(1..4)]); // Selects [20.0, 30.0, 40.0]
        
        assert!(sliced.requires_grad());
        let grad_fn = sliced.borrow_tensor_data().grad_fn.clone().unwrap();
        
        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]); // Grad for the sliced part
        grad_fn.backward(&upstream_grad, &mut gradients);
        
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Gradient for original tensor missing");
        // Expected: Gradient scattered back, zeros elsewhere
        let expected_grad_data = vec![0.0, 0.1, 0.2, 0.3, 0.0];
        check_tensor(grad_t, &expected_grad_data, &[5]);
    }
    
    #[test]
    fn test_slice_backward_1d_index() {
        let t = create_grad_tensor::<f32>(vec![10.0, 20.0, 30.0], vec![3]);
        let sliced = t.slice(&[TensorSlice::Index(1)]); // Selects 20.0 (scalar)
        
        let grad_fn = sliced.borrow_tensor_data().grad_fn.clone().unwrap();
        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![5.0], vec![]); // Grad for the scalar
        grad_fn.backward(&upstream_grad, &mut gradients);
        
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Gradient missing");
        let expected_grad_data = vec![0.0, 5.0, 0.0];
        check_tensor(grad_t, &expected_grad_data, &[3]);
    }
    
    #[test]
    fn test_slice_backward_2d_submatrix() {
        let t = create_grad_tensor::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        // Slice: [[5, 6], [8, 9]] -> shape [2, 2]
        let sliced = t.slice(&[TensorSlice::Range(1..3), TensorSlice::Range(1..3)]); 
        
        let grad_fn = sliced.borrow_tensor_data().grad_fn.clone().unwrap();
        let mut gradients = HashMap::new();
        // Upstream grad shape matches sliced shape [2, 2]
        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]); 
        grad_fn.backward(&upstream_grad, &mut gradients);
        
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Gradient missing");
        let expected_grad_data = vec![
            0.0, 0.0, 0.0, // Row 0
            0.0, 0.1, 0.2, // Row 1 -> elements corresponding to 5, 6
            0.0, 0.3, 0.4, // Row 2 -> elements corresponding to 8, 9
        ];
        check_tensor(grad_t, &expected_grad_data, &[3, 3]);
    }
    
    #[test]
    fn test_slice_backward_2d_row() {
        let t = create_grad_tensor::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let sliced = t.slice(&[TensorSlice::Index(1), TensorSlice::Full]); // Selects [4.0, 5.0, 6.0] -> shape [3]
        
        let grad_fn = sliced.borrow_tensor_data().grad_fn.clone().unwrap();
        let mut gradients = HashMap::new();
        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]); // Grad for the row
        grad_fn.backward(&upstream_grad, &mut gradients);
        
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Gradient missing");
        let expected_grad_data = vec![
            0.0, 0.0, 0.0, // Row 0
            0.1, 0.2, 0.3, // Row 1
        ];
        check_tensor(grad_t, &expected_grad_data, &[2, 3]);
    }

    #[test]
    fn test_slice_backward_propagate_requires_grad() {
        let t_no_grad = Tensor::new(vec![1.0], vec![1]);
        let t_grad = create_grad_tensor::<f32>(vec![10.0, 20.0], vec![2]);
        
        // Slice from tensor without grad -> no grad
        let sliced1 = t_no_grad.slice(&[TensorSlice::Index(0)]);
        assert!(!sliced1.requires_grad());
        assert!(sliced1.borrow_tensor_data().grad_fn.is_none());
        
        // Slice from tensor with grad -> grad
        let sliced2 = t_grad.slice(&[TensorSlice::Range(0..1)]);
        assert!(sliced2.requires_grad());
        assert!(sliced2.borrow_tensor_data().grad_fn.is_some());
    }

}
