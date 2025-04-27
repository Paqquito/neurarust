use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{AddAssign};
use num_traits::{One}; // For AddAssign+Clone in BackwardOp
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::cell::RefCell;

// --- Forward Operation --- 

impl<T> Tensor<T> {
    /// Transposes the tensor by swapping the last two dimensions.
    /// For a 2D tensor, this is the standard matrix transpose.
    /// For tensors with more dimensions, it transposes the inner matrices.
    pub fn transpose(&self) -> Tensor<T>
    where
        T: Copy + Clone + 'static + AddAssign + One, // Bounds needed for potential backward pass
    {
        let input_td = self.borrow_tensor_data();
        let shape = &input_td.shape;
        let rank = shape.len();
        assert!(rank >= 2, "Transpose requires at least 2 dimensions.");

        let dim1 = rank - 2;
        let dim2 = rank - 1;
        let mut new_shape = shape.clone();
        new_shape.swap(dim1, dim2);

        let numel = input_td.numel();
        let mut result_data = Vec::with_capacity(numel);
        // Safety: We are filling the vector completely.
        unsafe { result_data.set_len(numel); }

        let n = shape[dim1];
        let m = shape[dim2];
        let batch_size: usize = shape[..dim1].iter().product(); // Product of dimensions before the last two

        // Iterate through batches (if any) and transpose inner matrices
        for batch_idx in 0..batch_size {
            let input_batch_offset = batch_idx * n * m;
            let output_batch_offset = batch_idx * m * n;
            for i in 0..n { // Original rows
                for j in 0..m { // Original columns
                    let input_idx = input_batch_offset + i * m + j;
                    let output_idx = output_batch_offset + j * n + i; // Swapped indices
                    result_data[output_idx] = input_td.data[input_idx];
                }
            }
        }
        
        let original_shape = shape.clone(); // Needed for backward
        drop(input_td);

        let requires_grad = self.requires_grad();
        let result = Tensor::new(result_data, new_shape);
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = TransposeBackward {
                input_ref: self.get_weak_ref(),
                original_shape: original_shape,
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }
}

// --- Backward Operation --- 

struct TransposeBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    original_shape: Vec<usize>, // Store original shape to transpose gradient back
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for TransposeBackward<T>
where
    T: Copy + Clone + 'static + AddAssign + One,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        let grad_clone = upstream_grad.clone(); // Clone upstream_grad to avoid borrow issues
        grad_clone.set_requires_grad(false); // Ensure the clone doesn't require grad

        if let Some(input_rc) = self.input_ref.upgrade() { 
            let mut input_td = input_rc.borrow_mut();
            if input_td.requires_grad {
                // The gradient of transpose is the transpose of the gradient.
                // Use the clone in calculations
                let local_grad = grad_clone.transpose(); 
                local_grad.set_requires_grad(false); // Ensure gradient tensor does not require grad
                
                // Sanity check: shape should match original input
                assert_eq!(local_grad.shape(), self.original_shape, "Gradient shape mismatch in TransposeBackward");

                // Accumulate gradient
                if let Some(existing_grad) = input_td.grad.as_mut() {
                    *existing_grad += &local_grad; // Use AddAssign
                } else {
                    input_td.grad = Some(local_grad); // Assign the new grad (requires_grad=false)
                }
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use num_traits::{Zero, One}; // For test helpers
    use std::ops::AddAssign;    // For test helpers

    // Basic helper
    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where T: Copy + Clone + 'static + AddAssign + One + Zero + std::fmt::Debug + PartialEq
    {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_transpose_2d() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let t = Tensor::new(data, shape);
        let result = t.transpose();
        
        let expected_data = vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let expected_shape = vec![3, 2];
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
        assert!(!result.requires_grad());
    }
    
    #[test]
    fn test_transpose_3d() {
        // Shape [2, 2, 3] -> transpose last two dims -> [2, 3, 2]
        let data = vec![
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0, // Batch 0
            7.0, 8.0, 9.0, 10.0,11.0,12.0  // Batch 1
        ];
        let shape = vec![2, 2, 3];
        let t = Tensor::new(data, shape);
        let result = t.transpose();
        
        let expected_data = vec![
            1.0, 4.0,  2.0, 5.0,  3.0, 6.0, // Batch 0 transposed
            7.0, 10.0, 8.0, 11.0, 9.0, 12.0  // Batch 1 transposed
        ];
        let expected_shape = vec![2, 3, 2];
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }

    #[test]
    #[should_panic]
    fn test_transpose_1d() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![3];
        let t = Tensor::new(data, shape);
        let _ = t.transpose(); // Should panic
    }

    #[test]
    fn test_transpose_propagate_requires_grad() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t1.transpose();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let t2 = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result2 = t2.transpose();
        assert!(!result2.requires_grad());
        assert!(result2.grad_fn().is_none());
    }

    #[test]
    fn test_transpose_backward() {
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        // t1 = [[1, 2], [3, 4]]
        let result = t1.transpose(); 
        // result = [[1, 3], [2, 4]]
        
        // Create a dummy gradient for the result (same shape as result)
        let upstream_grad_data = vec![10.0, 20.0, 30.0, 40.0]; // [[10, 20], [30, 40]] -> shape [2, 2] ?? No, shape [3,2]? Check t1 shape.
        // result shape is [2,2]. So upstream grad shape is [2,2].
        let upstream_grad = Tensor::new(upstream_grad_data, vec![2, 2]); 

        // Manually call backward on the TransposeBackward op
        let grad_fn = result.grad_fn().clone().unwrap();
        grad_fn.backward(&upstream_grad);

        let grad_t1 = t1.grad();
        assert!(grad_t1.is_some());
        let grad_t1_tensor = grad_t1.unwrap();
        
        // Expected grad is transpose of upstream_grad: [[10, 30], [20, 40]]
        let expected_grad_data = vec![10.0_f32, 30.0, 20.0, 40.0];
        assert_eq!(grad_t1_tensor.data(), expected_grad_data);
        assert_eq!(grad_t1_tensor.shape(), vec![2, 2]); // Shape of original t1
    }
} 