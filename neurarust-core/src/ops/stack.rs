use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use crate::tensor::utils::calculate_strides;
use std::ops::{AddAssign};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;
use std::fmt::Debug;
use num_traits::Zero;
use std::collections::HashMap;
use num_traits::One;
use crate::error::NeuraRustError;

/// Backward operation for the `stack_op` function.
#[derive(Debug)]
struct StackBackward<T> {
    /// Weak references to the original input tensor data.
    input_refs: Vec<Weak<RefCell<TensorData<T>>>>,
    /// Store shapes for unstacking in backward pass
    input_shapes: Vec<Vec<usize>>,
    /// The dimension along which stacking occurred.
    dim: usize,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for StackBackward<T>
where
    // Add Zero/One for Tensor::stack/slice + Default/Debug/Static
    T: Clone + Debug + Default + Zero + One + AddAssign + 'static + Copy,
{
    /// Performs the backward pass for the stacking operation.
    /// Splits the `upstream_grad` along the stacking dimension and accumulates
    /// the resulting chunks into the gradients for the original input tensors.
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        
        let upstream_shape = upstream_grad.shape();
        let num_inputs = self.input_refs.len();
        
        // Validate that the stacked dimension size matches the number of inputs
        if upstream_shape.get(self.dim).copied() != Some(num_inputs) {
            eprintln!(
                "StackBackward Error: Upstream grad shape {:?} at dim {} does not match number of inputs {}",
                upstream_shape, self.dim, num_inputs
            );
            // Avoid panic, maybe log or handle error differently?
            // For now, just return without propagating gradients.
            return;
        }

        // Iterate through each original input tensor
        for i in 0..num_inputs {
            if let Some(input_rc) = self.input_refs[i].upgrade() {
                let input_ptr = Rc::as_ptr(&input_rc);

                // --- Slice the upstream_grad to get the chunk for this input --- 
                // Construct the slice configuration
                let mut slices = Vec::with_capacity(upstream_shape.len());
                for d in 0..upstream_shape.len() {
                    if d == self.dim {
                        slices.push(crate::ops::indexing::TensorSlice::Index(i));
                    } else {
                        slices.push(crate::ops::indexing::TensorSlice::Full);
                    }
                }
                
                // Perform the slice operation
                // Note: .slice() automatically handles requires_grad=false for the result
                let grad_chunk = upstream_grad.slice(&slices);
                // The shape of grad_chunk should now match the shape of the original input tensor.

                // --- Accumulate the gradient chunk --- 
                gradients.entry(input_ptr)
                    .and_modify(|existing_grad| { 
                        // Manual accumulation
                         assert_eq!(existing_grad.shape(), grad_chunk.shape(),
                                   "Gradient shape mismatch during stack backward accumulation (Input {}, Dim {}, Upstream {:?})",
                                   i, self.dim, upstream_shape);
                        let mut existing_data = existing_grad.data_mut();
                        let chunk_data_ref = grad_chunk.data(); 
                        existing_data.iter_mut()
                            .zip(chunk_data_ref.iter())
                            .for_each(|(e, s)| *e += s.clone());
                     })
                    .or_insert(grad_chunk);
            } else {
                 eprintln!("StackBackward::backward: Input tensor weak reference expired at index {}.", i);
            }
        }
        // Remove placeholder warning
        // eprintln!("WARNING: StackBackward::backward not implemented yet!");
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        self.input_refs.clone()
    }
}

/// Stacks a sequence of tensors along a new dimension.
/// 
/// All input tensors must have the same shape.
/// 
/// # Arguments
/// * `tensors` - A slice of tensors to stack.
/// * `dim` - The dimension along which to stack. The new dimension will be inserted here.
///
/// # Returns
/// A `Result` containing the stacked tensor or an error message.
pub fn stack_op<T>(tensors: &[Tensor<T>], dim: usize) -> Result<Tensor<T>, NeuraRustError>
where
    // Add T: One, needed for the BackwardOp trait object conversion
    T: Clone + Debug + Default + Zero + One + AddAssign + 'static + Copy, 
{
    if tensors.is_empty() {
        return Err(NeuraRustError::UnsupportedOperation("Cannot stack empty tensor list".to_string()));
    }

    // --- 1. Validate input shapes and calculate output shape --- 
    let first_shape = tensors[0].shape();
    let num_tensors = tensors.len();
    let rank = first_shape.len();

    // Validate dimension
    if dim > rank {
        return Err(NeuraRustError::IndexOutOfBounds { 
            index: vec![dim], // Simplified index for error
            shape: vec![rank + 1] // Indicate target rank
        });
    }
    
    // Validate shapes are consistent
    for (i, tensor) in tensors.iter().enumerate().skip(1) {
        if tensor.shape() != first_shape {
            return Err(NeuraRustError::IncompatibleShapes { 
                shape1: first_shape.clone(), 
                shape2: tensor.shape(),
                // Add detail about which tensor mismatched
                // detail: Some(format!("Tensor at index {} has mismatching shape", i))
            });
        }
    }

    // Calculate output shape
    let mut output_shape = first_shape.clone();
    output_shape.insert(dim, num_tensors);

    // --- 2. Allocate output data and copy input data --- 
    let output_numel = output_shape.iter().product::<usize>();
    let mut output_data = vec![T::default(); output_numel]; // Allocate and default-initialize

    let output_strides = calculate_strides(&output_shape);
    let input_numel = first_shape.iter().product::<usize>();
    let input_strides = calculate_strides(&first_shape);

    // Iterate through each input tensor to copy its data
    for (i, input_tensor) in tensors.iter().enumerate() {
        let input_td = input_tensor.borrow_tensor_data();
        
        // Iterate through each element of the current input tensor
        for input_linear_idx in 0..input_numel {
            // Get the value from the input tensor
            let input_val = input_td.data[input_linear_idx].clone();
            
            // Calculate the multi-dimensional coordinates in the *input* tensor shape
            let input_coords = crate::tensor::utils::index_to_coord(input_linear_idx, &input_strides, &first_shape);
            let input_coords_clone = input_coords.clone(); 
            let mut output_coords = input_coords_clone;
            output_coords.insert(dim, i);
            
            // Calculate the linear index in the output tensor manually
            let output_linear_idx = output_coords.iter().zip(&output_strides)
                .map(|(&coord, &stride)| coord * stride)
                .sum::<usize>();
            
            if output_linear_idx >= output_numel {
                return Err(NeuraRustError::InternalError(format!(
                    "Internal error: Calculated output index {} out of bounds ({}). Input tensor {}, input idx {}, input coords {:?}, output coords {:?}",
                    output_linear_idx, 
                    output_numel, 
                    i, 
                    input_linear_idx, 
                    input_coords, 
                    output_coords
                )));
            }
            output_data[output_linear_idx] = input_val;
        }
        // Release borrow for the current input tensor
        drop(input_td);
    }

    // Remove placeholder warning
    // eprintln!("WARNING: stack_op data copying not implemented yet!");

    // --- 3. Create output tensor and set up autograd --- 
    let result = Tensor::new(output_data, output_shape)?;
    let requires_grad = tensors.iter().any(|t| t.requires_grad());

    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = StackBackward {
            input_refs: tensors.iter().map(|t| t.get_weak_ref()).collect(),
            input_shapes: tensors.iter().map(|t| t.shape()).collect(),
            dim,
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }

    Ok(result)
} 

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*; 
    use crate::Tensor;
    use num_traits::{Zero, One};
    use crate::error::NeuraRustError;

    // Helper functions
    fn create_tensor<T: Clone + Debug + Default + Zero + One + AddAssign + 'static + Copy>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
     fn create_grad_tensor<T: Clone + Debug + Default + Zero + One + AddAssign + 'static + Copy>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }

    #[test]
    fn test_stack_basic() {
        let t1 = create_tensor(vec![1, 2], vec![2]);
        let t2 = create_tensor(vec![3, 4], vec![2]);

        // Stack along new axis 0
        let result0 = stack_op(&[t1.clone(), t2.clone()], 0);
        assert!(result0.is_ok());
        let res0 = result0.unwrap();
        assert_eq!(res0.shape(), vec![2, 2]);
        assert_eq!(res0.data().to_vec(), vec![1, 2, 3, 4]);

        // Stack along new axis 1
        let result1 = stack_op(&[t1.clone(), t2.clone()], 1);
        assert!(result1.is_ok());
        let res1 = result1.unwrap();
        assert_eq!(res1.shape(), vec![2, 2]);
        assert_eq!(res1.data().to_vec(), vec![1, 3, 2, 4]);
    }

    #[test]
    fn test_stack_multiple_tensors() {
        let t1 = create_tensor(vec![1], vec![1]);
        let t2 = create_tensor(vec![2], vec![1]);
        let t3 = create_tensor(vec![3], vec![1]);

        let result = stack_op(&[t1, t2, t3], 0);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.shape(), vec![3, 1]);
        assert_eq!(res.data().to_vec(), vec![1, 2, 3]);
    }
    
     #[test]
    fn test_stack_2d_tensors() {
        let t1 = create_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let t2 = create_tensor(vec![5, 6, 7, 8], vec![2, 2]);

        // Stack along axis 0
        let result0 = stack_op(&[t1.clone(), t2.clone()], 0);
         assert!(result0.is_ok());
        let res0 = result0.unwrap();
        assert_eq!(res0.shape(), vec![2, 2, 2]);
        assert_eq!(res0.data().to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        // Stack along axis 1
        let result1 = stack_op(&[t1.clone(), t2.clone()], 1);
        assert!(result1.is_ok());
        let res1 = result1.unwrap();
        assert_eq!(res1.shape(), vec![2, 2, 2]);
        // Expected: [[ [1,2], [5,6] ], [ [3,4], [7,8] ] ] -> Flattened
        assert_eq!(res1.data().to_vec(), vec![1, 2, 5, 6, 3, 4, 7, 8]);

        // Stack along axis 2
        let result2 = stack_op(&[t1.clone(), t2.clone()], 2);
         assert!(result2.is_ok());
        let res2 = result2.unwrap();
        assert_eq!(res2.shape(), vec![2, 2, 2]);
         // Expected: [[ [1,5], [2,6] ], [ [3,7], [4,8] ] ] -> Flattened
        assert_eq!(res2.data().to_vec(), vec![1, 5, 2, 6, 3, 7, 4, 8]);
    }

    #[test]
    fn test_stack_errors() {
        let t1 = create_tensor(vec![1, 2], vec![2]);
        let t2_diff_shape = create_tensor(vec![3, 4, 5], vec![3]);

        // Empty list
        let result_empty = stack_op::<i32>(&[], 0);
        assert!(result_empty.is_err());
        assert!(matches!(result_empty.err().unwrap(), NeuraRustError::UnsupportedOperation(_)));

        // Axis out of bounds
        let result_axis = stack_op(&[t1.clone()], 2); // rank=1, axis=2
        assert!(result_axis.is_err());
         assert!(matches!(result_axis.err().unwrap(), NeuraRustError::IndexOutOfBounds { .. }));

        // Shape mismatch
        let result_shape = stack_op(&[t1.clone(), t2_diff_shape], 0);
        assert!(result_shape.is_err());
         assert!(matches!(result_shape.err().unwrap(), NeuraRustError::IncompatibleShapes { .. }));
    }

    // --- Backward Tests --- 
    // TODO: Add backward tests for stack

} 