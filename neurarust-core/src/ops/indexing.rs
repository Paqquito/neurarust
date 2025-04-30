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
use crate::error::NeuraRustError;
use std::cell::RefMut;

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

/// Helper for Backward
fn accumulate_gradient_scatter<T>(
    gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>,
    input_weak_ref: &Weak<RefCell<TensorData<T>>>, 
    scattered_grad_result: Result<Tensor<T>, NeuraRustError>,
    original_shape: &[usize],
    slices: &[TensorSlice],
)
where
    T: AddAssign + Clone + Debug + Zero + Copy + Default + 'static,
{
    if let Some(input_rc) = input_weak_ref.upgrade() {
        let input_ptr = Rc::as_ptr(&input_rc);
        
        let scattered_grad_tensor = match scattered_grad_result {
            Ok(t) => t,
            Err(e) => panic!("Failed to create scattered gradient tensor: {:?}", e),
        };
        
        gradients.entry(input_ptr)
            .and_modify(|existing_grad| {
                if existing_grad.shape() != original_shape {
                     panic!("Shape mismatch during gradient accumulation: existing {:?} != original {:?}", existing_grad.shape(), original_shape);
                }
                let mut existing_data_mut = existing_grad.borrow_tensor_data_mut();
                
                scatter_add(&mut existing_data_mut, &scattered_grad_tensor, slices);
            })
            .or_insert_with(|| { 
                let numel = original_shape.iter().product();
                let zero_data = vec![T::zero(); numel];
                let zero_grad = Tensor::new(zero_data, original_shape.to_vec())
                    .expect("Failed to create zero gradient tensor in accumulate_gradient_scatter");
                scatter_add(&mut zero_grad.borrow_tensor_data_mut(), &scattered_grad_tensor, slices);
                zero_grad
            });
    }
}

/// Helper to scatter-add the `incoming_grad` into the `target_data` based on `slices`.
/// Modifies `target_data` in place.
fn scatter_add<T>(target_data: &mut RefMut<TensorData<T>>, incoming_grad: &Tensor<T>, slices: &[TensorSlice]) 
where
 T: AddAssign + Copy + Clone + Debug + Default + Zero + 'static,
 {
    let target_shape = target_data.shape.clone();
    let target_strides = target_data.strides.clone();
    let incoming_shape = incoming_grad.shape();
    let incoming_data = incoming_grad.borrow_tensor_data();
    let incoming_strides = &incoming_data.strides;

    let num_incoming = incoming_grad.numel();
    let rank = target_shape.len();

    let mut current_target_indices = vec![0; rank];

    for i in 0..num_incoming {
        let incoming_coords = crate::tensor::utils::index_to_coord(i, incoming_strides, &incoming_shape);
        
        let mut target_coord_offset = 0;
        for dim in 0..rank {
            match &slices[dim] {
                TensorSlice::Index(idx) => {
                    current_target_indices[dim] = *idx;
                }
                TensorSlice::Range(range) => {
                    current_target_indices[dim] = range.start + incoming_coords[target_coord_offset];
                    target_coord_offset += 1;
                }
                TensorSlice::Full => {
                    current_target_indices[dim] = incoming_coords[target_coord_offset];
                    target_coord_offset += 1;
                }
            }
        }
        
        let target_offset = target_data.get_offset(&current_target_indices);
        
        target_data.data[target_offset] += incoming_data.data[i];
    }
}

/// Backward Implementation
impl<T> BackwardOp<T> for SliceBackward<T> 
where
    T: AddAssign + Clone + Debug + Zero + Copy + Default + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) { 
        accumulate_gradient_scatter(
            gradients, 
            &self.input_ref, 
            Ok(upstream_grad.clone()),
            &self.original_shape, 
            &self.slices
        );
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}

/// Performs the forward slicing operation.
/// Creates a new tensor containing elements selected by the slices.
/// (Internal function called by `Tensor::slice`)
pub fn slice_op<T>(input: &Tensor<T>, slices: &[TensorSlice]) -> Result<Tensor<T>, NeuraRustError>
where
    T: Clone + Debug + Default + Zero + AddAssign + Copy + 'static,
{
    let input_td = input.borrow_tensor_data();
    let input_shape = &input_td.shape;
    let rank = input_shape.len();

    if slices.len() != rank {
        return Err(NeuraRustError::SliceError {
            message: format!("Number of slices ({}) does not match tensor rank ({})", slices.len(), rank)
        });
    }

    let mut output_shape = Vec::new();
    let mut output_offsets = Vec::new();
    let mut current_offset = 0;
    let mut sliced_strides = Vec::new();

    for (dim, slice) in slices.iter().enumerate() {
        let dim_size = input_shape[dim];
        match slice {
            TensorSlice::Index(idx) => {
                if *idx >= dim_size {
                    return Err(NeuraRustError::IndexOutOfBounds { 
                        index: vec![*idx],
                        shape: input_shape.clone()
                    });
                }
                current_offset += idx * input_td.strides[dim];
            }
            TensorSlice::Range(range) => {
                if range.start >= dim_size || range.end > dim_size || range.start >= range.end {
                    return Err(NeuraRustError::SliceError {
                       message: format!("Invalid range {:?} for dimension {} with size {}", range, dim, dim_size)
                    });
                }
                let len = range.end - range.start;
                output_shape.push(len);
                output_offsets.push(range.start * input_td.strides[dim]);
                 sliced_strides.push(input_td.strides[dim]);
            }
            TensorSlice::Full => {
                output_shape.push(dim_size);
                output_offsets.push(0);
                sliced_strides.push(input_td.strides[dim]);
            }
        }
    }

    let numel_output = output_shape.iter().product();
    let mut output_data = Vec::with_capacity(numel_output);
    unsafe { output_data.set_len(numel_output); }

    let output_strides_contiguous = calculate_strides(&output_shape);
    for i in 0..numel_output {
        let output_coords = crate::tensor::utils::index_to_coord(i, &output_strides_contiguous, &output_shape);
        
        let mut input_index_offset = current_offset;
        let mut output_coord_idx = 0;
        for (dim, slice) in slices.iter().enumerate() {
            match slice {
                TensorSlice::Index(_) => { /* Already handled by current_offset */ }
                TensorSlice::Range(_) | TensorSlice::Full => {
                    input_index_offset += output_coords[output_coord_idx] * input_td.strides[dim];
                    input_index_offset += output_offsets[output_coord_idx]; 
                    output_coord_idx += 1;
                }
            }
        }
        output_data[i] = input_td.data[input_index_offset];
    }
    
    let original_shape = input_shape.clone();
    let owned_slices = slices.to_vec();
    let input_ref = input.get_weak_ref();
    drop(input_td);

    let result = Tensor::new(output_data, output_shape)?;

    let requires_grad = input.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = SliceBackward {
            original_shape,
            slices: owned_slices,
            input_ref,
            _phantom: PhantomData,
        };
        result.borrow_tensor_data_mut().grad_fn = Some(Rc::new(grad_fn));
    }

    Ok(result)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use num_traits::Zero;
    use crate::error::NeuraRustError;

    fn create_tensor<T: Clone + Debug + Default + Zero + AddAssign + Copy + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    fn create_grad_tensor<T: Clone + Debug + Default + Zero + AddAssign + Copy + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }

    #[test]
    fn test_slice_index() {
        let t = create_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = slice_op(&t, &[TensorSlice::Index(1), TensorSlice::Full]);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        assert_eq!(res_tensor.shape(), vec![3]);
        assert_eq!(res_tensor.data().to_vec(), vec![4, 5, 6]);

        let result2 = slice_op(&t, &[TensorSlice::Index(0), TensorSlice::Index(1)]);
         assert!(result2.is_ok());
        let res_tensor2 = result2.unwrap();
        assert_eq!(res_tensor2.shape(), Vec::<usize>::new());
        assert_eq!(res_tensor2.data().to_vec(), vec![2]);
    }

     #[test]
    fn test_slice_range() {
        let t = create_tensor((0..24).collect::<Vec<i32>>(), vec![2, 3, 4]);
        let result = slice_op(&t, &[TensorSlice::Full, TensorSlice::Range(1..3), TensorSlice::Full]);
         assert!(result.is_ok());
        let res_tensor = result.unwrap();
        assert_eq!(res_tensor.shape(), vec![2, 2, 4]);
        assert_eq!(res_tensor.data().to_vec(), vec![4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23]);
    }
    
    #[test]
    fn test_slice_full() {
         let t = create_tensor(vec![1, 2, 3, 4], vec![2, 2]);
         let result = slice_op(&t, &[TensorSlice::Full, TensorSlice::Full]);
         assert!(result.is_ok());
         let res_tensor = result.unwrap();
         assert_eq!(res_tensor.shape(), vec![2, 2]);
         assert_eq!(res_tensor.data().to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_slice_mixed() {
        let t = create_tensor((0..24).collect::<Vec<i32>>(), vec![2, 3, 4]);
        let result = slice_op(&t, &[TensorSlice::Index(1), TensorSlice::Range(0..2), TensorSlice::Index(3)]);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        assert_eq!(res_tensor.shape(), vec![2]); 
        assert_eq!(res_tensor.data().to_vec(), vec![15, 19]);
    }
    
     #[test]
    fn test_slice_errors() {
        let t = create_tensor(vec![1, 2, 3, 4], vec![2, 2]);

        let result1 = slice_op(&t, &[TensorSlice::Index(0)]);
        assert!(result1.is_err());
        assert!(matches!(result1.err().unwrap(), NeuraRustError::SliceError { .. }));

        let result2 = slice_op(&t, &[TensorSlice::Index(2), TensorSlice::Full]);
        assert!(result2.is_err());
         assert!(matches!(result2.err().unwrap(), NeuraRustError::IndexOutOfBounds { .. }));

        let result3 = slice_op(&t, &[TensorSlice::Range(2..3), TensorSlice::Full]);
        assert!(result3.is_err());
        assert!(matches!(result3.err().unwrap(), NeuraRustError::SliceError { .. }));

         let result4 = slice_op(&t, &[TensorSlice::Range(0..3), TensorSlice::Full]);
         assert!(result4.is_err());
         assert!(matches!(result4.err().unwrap(), NeuraRustError::SliceError { .. }));

         let result5 = slice_op(&t, &[TensorSlice::Range(1..1), TensorSlice::Full]);
         assert!(result5.is_err());
         assert!(matches!(result5.err().unwrap(), NeuraRustError::SliceError { .. }));
    }

    #[test]
    fn test_slice_backward_index() {
        let t = create_grad_tensor::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let sliced = slice_op(&t, &[TensorSlice::Index(1), TensorSlice::Full]).unwrap();
        let loss = sliced.sum();
        loss.backward(None);
        
        let grad_t = t.grad().expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2, 3]);
        assert_eq!(grad_t.data().to_vec(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }
    
    #[test]
    fn test_slice_backward_range() {
        let t = create_grad_tensor::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let sliced = slice_op(&t, &[TensorSlice::Full, TensorSlice::Range(1..3)]).unwrap();
        let loss = sliced.sum();
        loss.backward(None);

        let grad_t = t.grad().expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2, 3]);
        assert_eq!(grad_t.data().to_vec(), vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_slice_backward_mixed() {
        let t = create_grad_tensor::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        let sliced = slice_op(&t, &[TensorSlice::Index(0), TensorSlice::Range(1..3)]).unwrap();
        let loss = sliced.sum();
        loss.backward(None);

        let grad_t = t.grad().expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2, 4]);
        assert_eq!(grad_t.data().to_vec(), vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    
     #[test]
    fn test_slice_propagate_grad() {
        let t_grad = create_grad_tensor::<f32>(vec![1.0; 6], vec![2, 3]);
        let t_no_grad = create_tensor::<f32>(vec![1.0; 6], vec![2, 3]);
        
        let sliced_grad = slice_op(&t_grad, &[TensorSlice::Index(0), TensorSlice::Full]).unwrap();
        assert!(sliced_grad.requires_grad());
        assert!(sliced_grad.grad_fn().is_some());
        
        let sliced_no_grad = slice_op(&t_no_grad, &[TensorSlice::Index(0), TensorSlice::Full]).unwrap();
        assert!(!sliced_no_grad.requires_grad());
        assert!(sliced_no_grad.grad_fn().is_none());
    }
}
