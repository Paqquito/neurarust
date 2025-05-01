use crate::tensor::Tensor;
use crate::autograd::{BackwardOp, accumulate_gradient};
use crate::tensor_data::TensorData;
use crate::error::NeuraRustError;

use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::fmt::Debug;
use std::collections::HashMap;
use num_traits::{Zero, One};
use std::ops::{AddAssign}; // Needed for accumulate_gradient
use std::iter::Sum;


// --- Forward Operation ---

/// Reshapes a tensor to a new shape without changing its data.
///
/// Currently, this operation performs a data copy. True view behavior (no copy)
/// will be implemented later, potentially requiring architectural changes.
///
/// # Arguments
/// * `input` - The tensor to reshape.
/// * `new_shape` - The desired new shape. The total number of elements must remain the same.
///
/// # Returns
/// A `Result` containing the reshaped tensor or a `NeuraRustError` if shapes are incompatible.
pub fn reshape<T>(input: &Tensor<T>, new_shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError>
where
    T: Clone + Debug + Default + Zero + AddAssign + Copy + One + Sum + 'static,
{
    let input_td = input.borrow_tensor_data();
    let input_shape = input_td.shape.clone();
    let input_numel = input_shape.iter().product::<usize>();
    let new_numel = new_shape.iter().product::<usize>();

    // 1. Validate number of elements
    if input_numel != new_numel {
        return Err(NeuraRustError::ShapeMismatch {
            expected: input_shape, // Or perhaps format a message indicating numel mismatch
            actual: new_shape,
        });
    }

    // 2. Perform data copy (TEMPORARY - should be a view)
    // TODO: Implement true view behavior (no data copy) when architecture allows.
    // This might involve checking for contiguity and manipulating strides/offset.
    let data_copy = input_td.data.clone();

    drop(input_td); // Release borrow before creating new tensor

    // 3. Create the new tensor
    let result = Tensor::new(data_copy, new_shape.clone())?;

    // 4. Set up autograd if needed
    if input.requires_grad() {
        result.set_requires_grad(true);
        let grad_fn = ReshapeBackward {
            input_shape, // Store original shape
            input_ref: input.get_weak_ref(),
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }

    Ok(result)
}

// --- Backward Operation ---

#[derive(Debug)]
struct ReshapeBackward<T: 'static> {
    input_shape: Vec<usize>, // Shape of the original input tensor
    input_ref: Weak<RefCell<TensorData<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for ReshapeBackward<T>
where
    T: Clone + Debug + Default + Zero + AddAssign + Copy + One + Sum + 'static,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                // The gradient w.r.t. the input is the upstream gradient reshaped
                // back to the input's original shape.
                match reshape(upstream_grad, self.input_shape.clone()) {
                    Ok(local_gradient) => {
                        // Ensure the computed gradient does not itself require grad
                        local_gradient.set_requires_grad(false);
                        accumulate_gradient(gradients, &self.input_ref, local_gradient);
                    }
                    Err(e) => {
                        // This should ideally not happen if forward reshape succeeded
                        // Maybe log an error or panic in debug mode
                        eprintln!("Error during reshape backward: {:?}", e);
                    }
                }
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}


// --- Tensor Method ---

impl<T> Tensor<T> {
    /// Returns a tensor with the same data and number of elements as `self`
    /// but with the specified shape.
    ///
    /// Currently, this operation performs a data copy. True view behavior (no copy)
    /// will be implemented later.
    ///
    /// # Arguments
    /// * `new_shape` - The desired new shape.
    ///
    /// # Panics
    /// Panics if the total number of elements in `new_shape` is different from `self`.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor<T>
    where
        T: Clone + Debug + Default + Zero + AddAssign + Copy + One + Sum + 'static,
    {
        reshape(self, new_shape)
            .unwrap_or_else(|e| panic!("Tensor reshape failed: {:?}", e))
    }
}


// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    // Helper to create tensors for tests
    fn create_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
        T: Clone + Debug + Default + Zero + AddAssign + PartialEq + 'static,
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    fn create_grad_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
        T: Clone + Debug + Default + Zero + AddAssign + PartialEq + 'static,
    {
        let t = Tensor::new(data, shape).expect("Test grad tensor creation failed");
        t.set_requires_grad(true);
        t
    }

    #[test]
    fn test_reshape_forward_ok() {
        let t1 = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let new_shape = vec![3, 2];
        let result = reshape(&t1, new_shape.clone());

        assert!(result.is_ok());
        let res_tensor = result.unwrap();

        assert_eq!(res_tensor.shape(), new_shape);
        assert_eq!(res_tensor.data().to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // Data is copied
        assert!(!res_tensor.requires_grad());
    }

    #[test]
    fn test_reshape_forward_numel_mismatch() {
        let t1 = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let new_shape = vec![3, 3]; // 9 elements vs 6
        let result = reshape(&t1, new_shape.clone());

        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(actual, new_shape);
            }
            _ => panic!("Incorrect error type"),
        }
    }

    #[test]
    fn test_reshape_propagates_grad() {
        let t1 = create_grad_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = reshape(&t1, vec![4, 1]).unwrap();

        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());
        // Check name? Requires Debug impl for dyn BackwardOp
        // println!("{:?}", result.grad_fn().unwrap());
    }

    #[test]
    fn test_reshape_backward() {
        let t1 = create_grad_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        t1.zero_grad(); // Ensure grad is initially None

        let reshaped = reshape(&t1, vec![6]).unwrap();

        // Simulate backward pass
        let upstream_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![6])
            .expect("Failed to create upstream grad");
        reshaped.backward(Some(&upstream_grad));

        let grad_t1 = t1.grad();
        assert!(grad_t1.is_some());
        let grad_t1 = grad_t1.unwrap();

        // Gradient should be reshaped back to the original shape of t1
        assert_eq!(grad_t1.shape(), vec![2, 3]);
        assert_eq!(grad_t1.data().to_vec(), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

     #[test]
     fn test_reshape_method() {
         let t1 = create_tensor(vec![1, 2, 3, 4], vec![2, 2]);
         let res = t1.reshape(vec![1, 4]);
         assert_eq!(res.shape(), vec![1, 4]);
         assert_eq!(res.data().to_vec(), vec![1, 2, 3, 4]);
     }

     #[test]
     #[should_panic]
     fn test_reshape_method_panic() {
         let t1 = create_tensor(vec![1, 2, 3, 4], vec![2, 2]);
         t1.reshape(vec![1, 5]); // Should panic due to numel mismatch
     }
} 