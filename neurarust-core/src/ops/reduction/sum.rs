use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Add, AddAssign};
use num_traits::{Zero, One}; // Need Zero for sum init, One for grad init, AddAssign for grad accum
use std::rc::{Rc, Weak};
use std::marker::PhantomData; // Add PhantomData import
use std::cell::RefCell;
use std::collections::HashSet; // Added for shape calculation helper
use crate::tensor::utils::calculate_strides; // Mise à jour de l'import
use std::fmt::Debug;
use std::collections::HashMap;
use crate::error::NeuraRustError;
use crate::autograd::accumulate_gradient; // Import from autograd
use std::iter::Sum; // MOVED IMPORT HERE

// --- Structures (BackwardOp defined before use) ---

/// Backward operation for sum reduction.
#[derive(Debug)]
struct SumAxesBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    input_shape: Vec<usize>,
    axes: Vec<usize>, // Axes that were reduced
    keep_dims: bool,   // Whether dims were kept
    _phantom: PhantomData<T>,
}

// --- Forward Operation --- 

// Helper function to calculate the output shape after reduction
fn calculate_reduced_shape(input_shape: &[usize], axes: &[usize], keep_dims: bool) -> Vec<usize> {
    let rank = input_shape.len();
    let axes_set: HashSet<_> = if axes.is_empty() {
        // If axes is empty, we are summing over all axes.
        (0..rank).collect()
    } else {
        axes.iter().cloned().collect()
    };

    if axes_set.len() == rank { // Summing over all dimensions
        return if keep_dims { vec![1; rank] } else { vec![] }; // Return empty shape for scalar
    }

    let mut reduced_shape = Vec::new();
    for (i, &dim) in input_shape.iter().enumerate() {
        if axes_set.contains(&i) {
            if keep_dims {
                reduced_shape.push(1);
            } // else: omit the dimension
        } else {
            reduced_shape.push(dim);
        }
    }
    
    // If the reduced shape is empty AND the input was not already scalar, 
    // it implies a reduction to scalar without keep_dims.
    // This case is now handled by the `axes_set.len() == rank` check above.
    // The only way reduced_shape can be empty here is if input_shape was empty.
    if reduced_shape.is_empty() && !input_shape.is_empty() {
         // This branch should ideally not be reached if the logic above is correct.
         // Let's return [] as a scalar shape.
         vec![] 
    } else {
        reduced_shape
    }
}

// Define the public, fallible function

/// Calculates the sum of tensor elements over given dimensions.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn sum_axes<T>(
    input: &Tensor<T>, 
    axes: &[usize], 
    keep_dims: bool
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Add<Output = T> + Zero + Copy + Clone + 'static + AddAssign + Debug + Default + One + Sum<T>,
{
    let input_td = input.borrow_tensor_data();
    let input_shape = input_td.shape.clone();
    let input_data = &input_td.data;
    let rank = input_shape.len();
    
    // Normalize and validate axes
    let axes_set: HashSet<_> = if axes.is_empty() {
        (0..rank).collect()
    } else {
        let set: HashSet<_> = axes.iter().cloned().collect();
        for &axis in &set {
            if axis >= rank {
                // Return specific error for out-of-bounds axis
                return Err(NeuraRustError::IndexOutOfBounds { 
                    index: vec![axis], // Represent axis as index
                    shape: vec![rank], // Represent rank as shape bounds
                });
            }
        }
        set
    };
    
    let output_shape = calculate_reduced_shape(&input_shape, axes, keep_dims);
    let output_numel: usize = output_shape.iter().product();
    let mut output_data = vec![T::zero(); output_numel];
    
    let output_strides = calculate_strides(&output_shape);

    // --- Reduction Logic ---
    let mut current_input_coords = vec![0; rank];
    for input_linear_idx in 0..input_td.numel() {
        let mut current_output_coords = Vec::with_capacity(output_shape.len());
        for input_axis in 0..rank {
            if !axes_set.contains(&input_axis) {
                current_output_coords.push(current_input_coords[input_axis]);
            } else if keep_dims {
                current_output_coords.push(0);
            }
        }

        let output_linear_idx = if output_shape.is_empty() {
            0 
        } else {
            current_output_coords.iter().zip(&output_strides).map(|(&coord, &stride)| coord * stride).sum::<usize>()
        };
        
        if output_linear_idx < output_numel {
             // Use checked add or assume AddAssign handles overflow if necessary
             output_data[output_linear_idx] = output_data[output_linear_idx] + input_data[input_linear_idx];
        } else {
            // This path indicates a logic error, return InternalError
            return Err(NeuraRustError::InternalError(
                "Output index out of bounds during sum reduction logic.".to_string()
            ));
        }

        // Increment input coordinates
        for i in (0..rank).rev() {
            current_input_coords[i] += 1;
            if current_input_coords[i] < input_shape[i] { break; }
            current_input_coords[i] = 0;
        }
    }
    
    let input_shape_clone = input_shape; // Keep clone for grad_fn
    let input_weak_ref = input.get_weak_ref(); // Get weak ref before dropping borrow
    drop(input_td);

    let requires_grad = input.requires_grad();
    // Use ? for Tensor::new error
    let result = Tensor::new(output_data, output_shape.clone())?; 

    if requires_grad {
        result.set_requires_grad(true);
        let grad_fn = SumAxesBackward {
            input_ref: input_weak_ref,
            input_shape: input_shape_clone,
            axes: axes_set.into_iter().collect(),
            keep_dims,           
            _phantom: PhantomData,
        };
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }
    Ok(result)
}

// --- Tensor Methods (call fallible function) ---
impl<T: Debug> Tensor<T> {
    pub fn sum_axes(&self, axes: &[usize], keep_dims: bool) -> Tensor<T>
    where
        T: Add<Output = T> + Zero + Copy + Clone + 'static + AddAssign + Debug + Default + One + Sum<T>,
    {
        sum_axes(self, axes, keep_dims)
            .unwrap_or_else(|e| panic!("Tensor sum_axes failed: {:?}", e))
    }

    /// Computes the sum of all elements in the tensor.
    pub fn sum(&self) -> Tensor<T>
    where
         // Also add Sum constraint to the public method
        T: Add<Output = T> + Zero + Copy + Clone + 'static + AddAssign + One + Debug + Default + Sum<T>,
    {
        let rank = self.shape().len(); // Use shape().len() instead of rank()
        let axes: Vec<usize> = (0..rank).collect();
        // Call the fallible sum_axes and unwrap
        sum_axes(self, &axes, false)
            .unwrap_or_else(|e| panic!("Tensor sum failed: {:?}", e))
    }
}

// --- Backward Operation (SumAxes) ---

impl<T> BackwardOp<T> for SumAxesBackward<T>
where
    T: Clone + Debug + AddAssign + Zero + Copy + 'static + Default + One + Sum<T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                // --- Restauration de la logique d'expansion --- 
                let input_shape = &self.input_shape;
                let input_rank = input_shape.len();
                let input_numel = input_shape.iter().product();
                let axes_set: HashSet<_> = self.axes.iter().cloned().collect();
                
                let mut local_grad_data = vec![T::zero(); input_numel];
                let upstream_data = upstream_grad.data();
                let upstream_shape = upstream_grad.shape();
                let upstream_strides = calculate_strides(&upstream_shape);

                let mut current_input_coords = vec![0; input_rank];
                for input_linear_idx in 0..input_numel {
                    let mut current_upstream_coords = Vec::with_capacity(upstream_shape.len());
                    let mut _upstream_coord_idx = 0; // Prefix with underscore
                    for input_axis in 0..input_rank {
                        if !axes_set.contains(&input_axis) {
                            current_upstream_coords.push(current_input_coords[input_axis]);
                            _upstream_coord_idx += 1;
                        } else if self.keep_dims {
                            current_upstream_coords.push(0);
                            _upstream_coord_idx += 1;
                        }
                    }

                    let upstream_linear_idx = if upstream_shape.is_empty() { // Check for empty shape
                        0 // Scalar upstream grad always comes from index 0
                    } else {
                         current_upstream_coords.iter().zip(&upstream_strides).map(|(&coord, &stride)| coord * stride).sum::<usize>()
                    };

                    if upstream_linear_idx < upstream_data.len() { 
                         local_grad_data[input_linear_idx] = upstream_data[upstream_linear_idx];
                    } else {
                         panic!("Upstream index out of bounds during sum backward broadcasting.");
                    }

                    for i in (0..input_rank).rev() {
                        current_input_coords[i] += 1;
                        if current_input_coords[i] < input_shape[i] {
                            break; 
                        }
                        current_input_coords[i] = 0; 
                    }
                }
                // --- Fin de la logique d'expansion ---
                
                let grad_expanded = Tensor::new(local_grad_data, self.input_shape.clone())
                    .expect("Internal error: Failed to create expanded gradient in sum backward");
                accumulate_gradient(gradients, &self.input_ref, grad_expanded); 
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
    use super::*;
    use crate::Tensor;
    
    
    use std::rc::Rc; 
    use std::collections::HashMap;
    use num_traits::{Zero, One};
    use std::ops::{AddAssign, Add};
    use std::fmt::Debug;
    use crate::error::NeuraRustError;
    use std::iter::Sum; // Keep Sum import here as well if tests use it

    // --- Helpers --- (Ajouter bounds manquants si nécessaire)
    fn create_test_tensor<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T>
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + 'static
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
     fn create_test_tensor_with_grad<T>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T>
    where 
        T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + 'static
    {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }

    // --- Tests --- 
    #[test]
    fn test_calculate_reduced_shape() {
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0], false), vec![3, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1], false), vec![2, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[2], false), vec![2, 3]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1], false), vec![4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 2], false), vec![3]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1, 2], false), vec![2]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1, 2], false), vec![]); // Sum all

        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0], true), vec![1, 3, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1], true), vec![2, 1, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 2], true), vec![1, 3, 1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1, 2], true), vec![1, 1, 1]);
    }

    #[test]
    fn test_sum_axes_forward() {
        let t = create_test_tensor::<i32>(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Sum all axes, no keep_dims (scalar)
        let res_all = sum_axes(&t, &[], false);
        assert!(res_all.is_ok());
        let res_all_t = res_all.unwrap();
        assert_eq!(res_all_t.shape(), Vec::<usize>::new());
        assert_eq!(res_all_t.data().to_vec(), vec![21]);
        assert!(!res_all_t.requires_grad());

        // Sum axis 0, keep_dims
        let res0_keep = sum_axes(&t, &[0], true);
        assert!(res0_keep.is_ok());
        let res0_keep_t = res0_keep.unwrap();
        assert_eq!(res0_keep_t.shape(), vec![1, 3]);
        assert_eq!(res0_keep_t.data().to_vec(), vec![5, 7, 9]);

        // Sum axis 1, no keep_dims
        let res1_nokeep = sum_axes(&t, &[1], false);
        assert!(res1_nokeep.is_ok());
        let res1_nokeep_t = res1_nokeep.unwrap();
        assert_eq!(res1_nokeep_t.shape(), vec![2]);
        assert_eq!(res1_nokeep_t.data().to_vec(), vec![6, 15]);
        
        // Test Tensor method too
        let res_method = t.sum_axes(&[1], false);
        assert_eq!(res_method.shape(), vec![2]);
        assert_eq!(res_method.data().to_vec(), vec![6, 15]);
    }

    #[test]
    fn test_sum_forward() { // Tests the .sum() convenience method
        let t = create_test_tensor::<i32>(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let res = t.sum(); // This now calls the fallible sum_axes and unwraps
        assert_eq!(res.shape(), Vec::<usize>::new());
        assert_eq!(res.data().to_vec(), vec![21]);
    }

    #[test]
    fn test_sum_axes_invalid_axis() {
        let t = create_test_tensor::<i32>(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let result = sum_axes(&t, &[0, 2], false); // Axis 2 is out of bounds
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::IndexOutOfBounds { .. }));

        // // Test that Tensor method panics - REMOVED due to UnwindSafe issues
        // let panic_result = std::panic::catch_unwind(|| t.sum_axes(&[0, 2], false));
        // assert!(panic_result.is_err());
    }

    #[test]
    fn test_sum_grad_propagation() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum_axes(&t, &[], false).expect("Sum failed in test"); // Expect Ok
        assert!(s.requires_grad());
        assert!(s.grad_fn().is_some());

        // Test propagation over axis 0
        let t2 = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s2 = sum_axes(&t2, &[0], true).expect("Sum axes 0 failed in test"); // Expect Ok
        assert!(s2.requires_grad());
        assert!(s2.grad_fn().is_some());

        // Test with no grad
        let t_no_grad = create_test_tensor::<f32>(vec![1.0, 2.0], vec![2]);
        let s_no_grad = sum_axes(&t_no_grad, &[], false).expect("Sum no grad failed"); // Expect Ok
        assert!(!s_no_grad.requires_grad());
        assert!(s_no_grad.grad_fn().is_none());
    }

    #[test]
    fn test_sum_backward_multiple_inputs_simplified() {
        let t1 = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        let t2 = create_test_tensor_with_grad::<f32>(vec![3.0, 4.0], vec![2]);
        
        // s1 = t1.sum()
        let s1 = sum_axes(&t1, &[], false).expect("s1 sum failed"); // Expect Ok
        
        // s2 = t2.sum()
        let s2 = sum_axes(&t2, &[], false).expect("s2 sum failed"); // Expect Ok

        // Simulate an operation that uses both s1 and s2, e.g., result = s1 + s2
        // For simplicity, we'll just backpropagate a gradient of 1.0 through both s1 and s2
        let final_result_mock_grad = Tensor::new(vec![1.0f32], vec![]).unwrap();

        let mut gradients = HashMap::new();
        // Backpropagate through s1
        if let Some(grad_fn1) = s1.grad_fn() {
            grad_fn1.backward(&final_result_mock_grad, &mut gradients);
        }
        // Backpropagate through s2
        if let Some(grad_fn2) = s2.grad_fn() {
             grad_fn2.backward(&final_result_mock_grad, &mut gradients);
        }

        // Check gradients for t1 and t2
        let grad_t1 = gradients.get(&Rc::as_ptr(&t1.data)).expect("Grad t1 missing");
        assert_eq!(grad_t1.shape(), vec![2]);
        assert_eq!(grad_t1.data().to_vec(), vec![1.0, 1.0]);

        let grad_t2 = gradients.get(&Rc::as_ptr(&t2.data)).expect("Grad t2 missing");
        assert_eq!(grad_t2.shape(), vec![2]);
        assert_eq!(grad_t2.data().to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward_specific_axes_and_keepdim() -> Result<(), NeuraRustError> {
        // Input tensor: 2x2x3
        let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
        let input = Tensor::new_with_grad(data, vec![2, 2, 3])?;
        // Sum along axes 0 and 2, keeping dimensions
        let axes = vec![0, 2];
        let keep_dim = true;

        let loss = sum_axes(&input, &axes, keep_dim)?; // Pass input by reference

        // Check output shape
        assert_eq!(loss.shape(), vec![1, 2, 1]);

        // Check output values
        // Sum over axis 0: [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] -> [[6, 8, 10], [12, 14, 16]]
        // Sum over axis 2: [[6+8+10], [12+14+16]] -> [[24], [42]] -> shape [1, 2, 1]
        assert_eq!(loss.data().to_vec(), vec![24.0, 42.0]);

        // Perform backward pass
        {
             // Create an upstream gradient with the same shape as loss, filled with ones.
             let upstream_grad = Tensor::new(vec![1.0f32; loss.numel()], loss.shape())?;
             loss.backward(Some(&upstream_grad)); // Pass the explicit upstream grad
        } 

        // Access gradient *after* the backward scope
        let grad = input.grad().unwrap().clone();

        // Check the gradient
        // Gradient should be broadcasted back to the original shape. Since sum is element-wise 1, grad is 1 everywhere.
        let expected_grad_data = vec![1.0; 12];
        assert_eq!(grad.data().to_vec(), expected_grad_data);
        assert_eq!(grad.shape(), vec![2, 2, 3]);

        Ok(())
    }
    
    #[test]
    fn test_sum_backward_simple() { 
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
        let s = sum_axes(&t, &[], false).expect("Sum failed in test setup"); // Sum all
        
        s.backward(None); // Call backward on scalar

        let grad_t = t.grad().expect("Grad missing");
        assert_eq!(grad_t.shape(), vec![3]);
        assert_eq!(grad_t.data().to_vec(), vec![1.0, 1.0, 1.0]); // Gradient should be ones
    }

    #[test]
    fn test_sum_backward_keepdim() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum_axes(&t, &[], true).expect("Sum keepdim failed"); // Sum all, keep dims
        assert_eq!(s.shape(), vec![1, 1]);
        
        s.backward(None); // Default upstream grad is 1.0

        let grad_t = t.grad().expect("Grad missing");
        assert_eq!(grad_t.shape(), vec![2, 2]);
        assert_eq!(grad_t.data().to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward_axis() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = sum_axes(&t, &[1], false).expect("Sum axis 1 failed"); // Sum along axis 1
        assert_eq!(s.shape(), vec![2]);
        
        let upstream_grad = Tensor::new(vec![0.5, 2.0], vec![2]).unwrap();
        s.backward(Some(&upstream_grad));

        let grad_t = t.grad().expect("Grad missing");
        assert_eq!(grad_t.shape(), vec![2, 3]);
        assert_eq!(grad_t.data().to_vec(), vec![0.5, 0.5, 0.5, 2.0, 2.0, 2.0]);
    }

} 