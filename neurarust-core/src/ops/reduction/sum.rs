use crate::tensor::Tensor;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::ops::{Add, AddAssign};
use std::iter::Sum as IterSum; // Keep IterSum
use num_traits::{Zero, One}; // Need Zero for sum init, One for grad init, AddAssign for grad accum
use std::rc::{Rc, Weak};
use std::marker::PhantomData; // Add PhantomData import
use std::cell::RefCell;
use std::collections::HashSet; // Added for shape calculation helper

// --- Structures (BackwardOp defined before use) ---

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
    if axes.is_empty() { // Sum over all axes
        return if keep_dims { vec![1; input_shape.len()] } else { vec![1] };
    }
    let mut reduced_shape = Vec::new();
    let axes_set: HashSet<_> = axes.iter().cloned().collect(); // For quick lookup
    for (i, &dim) in input_shape.iter().enumerate() {
        if axes_set.contains(&i) {
            if keep_dims {
                reduced_shape.push(1);
            }
        } else {
            reduced_shape.push(dim);
        }
    }
    // If the result is empty (e.g., input was scalar, summed over axis 0),
    // return a scalar shape [1].
    if reduced_shape.is_empty() {
        vec![1]
    } else {
        reduced_shape
    }
}

// Helper function to calculate strides for a given shape
// TODO: Consider moving this to a more general tensor utils module
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![0; rank];
    if rank > 0 {
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

impl<T> Tensor<T> {
    /// Calculates the sum of tensor elements over given dimensions.
    ///
    /// # Arguments
    /// * `axes` - The dimensions to reduce. If empty, all dimensions are reduced.
    /// * `keep_dims` - Whether the output tensor has `dim` retained or not.
    pub fn sum_axes(&self, axes: &[usize], keep_dims: bool) -> Tensor<T>
    where
        T: Add<Output = T> + Zero + Copy + Clone + 'static + AddAssign + One + IterSum,
    {
        let input_td = self.borrow_tensor_data();
        let input_shape = input_td.shape.clone();
        let input_data = &input_td.data; // Borrow data instead of cloning
        let rank = input_shape.len();
        
        // Normalize and validate axes
        let axes_set: HashSet<_> = if axes.is_empty() {
            (0..rank).collect()
        } else {
            let set: HashSet<_> = axes.iter().cloned().collect();
            for &axis in &set {
                 assert!(axis < rank, "Axis {} out of bounds for tensor of rank {}", axis, rank);
            }
            set
        };
        
        let output_shape = calculate_reduced_shape(&input_shape, axes, keep_dims); // Pass original axes
        let output_numel: usize = output_shape.iter().product();
        let mut output_data = vec![T::zero(); output_numel];
        
        // Calculate strides for input and output shapes
        let input_strides = calculate_strides(&input_shape);
        let output_strides = calculate_strides(&output_shape);

        // --- Reduction Logic ---
        let mut current_input_coords = vec![0; rank];
        for input_linear_idx in 0..input_td.numel() {
            // Calculate output coordinates based on input coordinates
            let mut current_output_coords = Vec::with_capacity(output_shape.len());
            let mut output_coord_idx = 0;
            for input_axis in 0..rank {
                if !axes_set.contains(&input_axis) {
                    // Dimension not reduced, copy coordinate
                    current_output_coords.push(current_input_coords[input_axis]);
                    output_coord_idx += 1;
                } else if keep_dims {
                    // Dimension reduced, but kept, coordinate is 0
                    current_output_coords.push(0);
                    output_coord_idx += 1;
                }
                // If dimension is reduced and not kept, skip it
            }

            // Calculate output linear index (handle scalar case)
            let output_linear_idx = if output_shape == [1] {
                0
            } else {
                current_output_coords.iter().zip(&output_strides).map(|(&coord, &stride)| coord * stride).sum::<usize>()
            };
            
            // Add input value to output
            if output_linear_idx < output_numel { // Bounds check
                 output_data[output_linear_idx] = output_data[output_linear_idx] + input_data[input_linear_idx];
            } else {
                // This should ideally not happen if shapes/strides are correct
                panic!("Output index out of bounds during sum reduction.");
            }

            // --- Increment input coordinates (like nested loops) ---
            for i in (0..rank).rev() {
                current_input_coords[i] += 1;
                if current_input_coords[i] < input_shape[i] {
                    break; // No carry-over needed
                }
                current_input_coords[i] = 0; // Carry-over
            }
        }
        // --- End Reduction Logic ---
        
        let input_shape_clone = input_shape; // Already cloned
        drop(input_td);

        let requires_grad = self.requires_grad();
        let result = Tensor::new(output_data, output_shape); // Use calculated output_data
        if requires_grad {
            result.set_requires_grad(true);
            let grad_fn = SumAxesBackward {
                input_ref: self.get_weak_ref(),
                input_shape: input_shape_clone,
                axes: axes_set.into_iter().collect(), // Store the axes used as Vec
                keep_dims,           
                _phantom: PhantomData,
            };
            result.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }
        result
    }

    /// Calculates the sum of all elements in the tensor, returning a scalar tensor.
    /// Convenience wrapper around `sum_axes`.
    pub fn sum(&self) -> Tensor<T>
    where
        T: Add<Output = T> + Zero + Copy + Clone + 'static + AddAssign + One + IterSum, // Added IterSum back
    {
        self.sum_axes(&[], false) // Sum all axes, don't keep dims
    }
}

// --- Backward Operation (SumAxes) ---

impl<T> BackwardOp<T> for SumAxesBackward<T>
where
    T: Clone + Copy + One + AddAssign + 'static + Zero + Add<Output=T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        let grad_clone = upstream_grad.clone();
        grad_clone.set_requires_grad(false);

        if let Some(input_rc) = self.input_ref.upgrade() {
            let mut input_td = input_rc.borrow_mut();
            if input_td.requires_grad {
                // Gradient of sum is the upstream gradient broadcasted/
                // expanded back to the original input shape along the summed axes.

                let input_shape = &self.input_shape;
                let input_rank = input_shape.len();
                let input_numel = input_td.numel();
                let axes_set: HashSet<_> = self.axes.iter().cloned().collect();
                
                let mut local_grad_data = vec![T::zero(); input_numel];
                let upstream_data = grad_clone.data();
                let upstream_shape = grad_clone.shape();
                let upstream_strides = calculate_strides(&upstream_shape);

                // --- Broadcasting Logic ---
                let mut current_input_coords = vec![0; input_rank];
                for input_linear_idx in 0..input_numel {
                    // Calculate corresponding upstream coordinates
                    let mut current_upstream_coords = Vec::with_capacity(upstream_shape.len());
                    let mut upstream_coord_idx = 0;
                    for input_axis in 0..input_rank {
                        if !axes_set.contains(&input_axis) {
                            // Dimension was not reduced, copy coordinate
                            current_upstream_coords.push(current_input_coords[input_axis]);
                            upstream_coord_idx += 1;
                        } else if self.keep_dims {
                            // Dimension was reduced but kept (size 1)
                            // The upstream coordinate for this dimension is always 0
                            current_upstream_coords.push(0);
                             upstream_coord_idx += 1;
                        }
                        // If dimension reduced and not kept, skip (no corresponding upstream coord)
                    }

                    // Calculate upstream linear index (handle scalar case)
                    let upstream_linear_idx = if upstream_shape == [1] {
                        0
                    } else {
                         current_upstream_coords.iter().zip(&upstream_strides).map(|(&coord, &stride)| coord * stride).sum::<usize>()
                    };

                    // Assign upstream grad value to the current input position
                    if upstream_linear_idx < upstream_data.len() { // Bounds check
                         local_grad_data[input_linear_idx] = upstream_data[upstream_linear_idx];
                    } else {
                         panic!("Upstream index out of bounds during sum backward broadcasting.");
                    }

                    // --- Increment input coordinates ---
                    for i in (0..input_rank).rev() {
                        current_input_coords[i] += 1;
                        if current_input_coords[i] < input_shape[i] {
                            break; 
                        }
                        current_input_coords[i] = 0; 
                    }
                }
                // --- End Broadcasting Logic ---
                
                let local_grad = Tensor::new(local_grad_data, self.input_shape.clone());
                local_grad.set_requires_grad(false);

                // Accumulate gradient
                if let Some(existing_grad) = input_td.grad.as_mut() {
                    *existing_grad += &local_grad;
                } else {
                    input_td.grad = Some(local_grad);
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
    use num_traits::{Zero, One};
    use std::ops::{AddAssign, Add};
     
    use super::calculate_reduced_shape; 
     // Added import for test
    use crate::autograd::BackwardOp; // Import the trait

    // Helpers adjusted slightly if needed
    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq + Zero + AddAssign + One + Copy + Add<Output=T> + std::iter::Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero + AddAssign + One + Copy + Add<Output=T> + std::iter::Sum>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_calculate_reduced_shape() {
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[], false), vec![1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[], true), vec![1, 1, 1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0], false), vec![3, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0], true), vec![1, 3, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1], false), vec![2, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1], true), vec![2, 1, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[2], false), vec![2, 3]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[2], true), vec![2, 3, 1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1], false), vec![4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1], true), vec![1, 1, 4]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 2], false), vec![3]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 2], true), vec![1, 3, 1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1, 2], false), vec![2]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[1, 2], true), vec![2, 1, 1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1, 2], false), vec![1]);
        assert_eq!(calculate_reduced_shape(&[2, 3, 4], &[0, 1, 2], true), vec![1, 1, 1]);
        assert_eq!(calculate_reduced_shape(&[5], &[0], false), vec![1]); // Reduce scalar
        assert_eq!(calculate_reduced_shape(&[5], &[0], true), vec![1]); // Reduce scalar keep_dims
    }

    #[test]
    fn test_sum_all_forward() { // Renamed from test_sum_forward
        let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Test both ways
        let result_sum = t1.sum();
        let result_axes = t1.sum_axes(&[], false);
        assert_eq!(result_sum.data(), vec![21.0_f32]);
        assert_eq!(result_sum.shape(), vec![1]);
        assert_eq!(result_axes.data(), vec![21.0_f32]);
        assert_eq!(result_axes.shape(), vec![1]);
        assert!(!result_sum.requires_grad());
    }
    
     #[test]
    fn test_sum_scalar_forward() { // Renamed from test_sum_forward_scalar
        let t1 = create_test_tensor(vec![-5.0_f32], vec![1]);
        let result = t1.sum();
        assert_eq!(result.data(), vec![-5.0_f32]);
        assert_eq!(result.shape(), vec![1]);
        assert!(!result.requires_grad());
        let result_axes = t1.sum_axes(&[0], false);
        assert_eq!(result_axes.data(), vec![-5.0_f32]);
        assert_eq!(result_axes.shape(), vec![1]);
    }

    #[test]
    fn test_sum_all_propagate_requires_grad() { // Renamed
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0], vec![2]);
        let result = t1.sum();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let t2 = create_test_tensor(vec![3.0_f32], vec![1]);
        let result2 = t2.sum();
        assert!(!result2.requires_grad());
        assert!(result2.grad_fn().is_none());
    }

    #[test]
    fn test_sum_all_backward() { // Renamed
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t1.sum(); // result = 10.0

        assert!(t1.grad().is_none());
        result.backward(); // Upstream grad is implicitly 1.0

        let grad_t1 = t1.grad();
        assert!(grad_t1.is_some());
        let grad_t1_tensor = grad_t1.unwrap();
        // Expected grad is 1.0 distributed across the original shape
        assert_eq!(grad_t1_tensor.data(), vec![1.0_f32, 1.0, 1.0, 1.0]);
        assert_eq!(grad_t1_tensor.shape(), vec![2, 2]);
    }
    
    #[test]
    fn test_sum_all_backward_accumulation() { // Renamed
        let t1 = create_test_tensor_with_grad(vec![1.0_f32, 2.0], vec![2]);
        let t2 = create_test_tensor_with_grad(vec![3.0_f32, 4.0], vec![2]);
        
        let sum1 = t1.sum(); // 3.0
        let sum2 = t2.sum(); // 7.0
        let final_sum = &sum1 + &sum2; // 10.0

        final_sum.backward();

        // grad(t1) = grad(final_sum) * d(final_sum)/d(sum1) * d(sum1)/dt1 = 1.0 * 1.0 * [1.0, 1.0]
        assert_eq!(t1.grad().unwrap().data(), vec![1.0_f32, 1.0]);
        // grad(t2) = grad(final_sum) * d(final_sum)/d(sum2) * d(sum2)/dt2 = 1.0 * 1.0 * [1.0, 1.0]
        assert_eq!(t2.grad().unwrap().data(), vec![1.0_f32, 1.0]);
    }

    // --- Tests for sum_axes ---
    #[test]
    fn test_sum_axes_forward() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Sum axis 0, keep_dims=false -> Shape [3]
        let r0 = t.sum_axes(&[0], false);
        assert_eq!(r0.data(), vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]
        assert_eq!(r0.shape(), vec![3]);
        assert!(!r0.requires_grad());

        // Sum axis 0, keep_dims=true -> Shape [1, 3]
        let r0k = t.sum_axes(&[0], true);
        assert_eq!(r0k.data(), vec![5.0, 7.0, 9.0]);
        assert_eq!(r0k.shape(), vec![1, 3]);
        assert!(!r0k.requires_grad());

        // Sum axis 1, keep_dims=false -> Shape [2]
        let r1 = t.sum_axes(&[1], false);
        assert_eq!(r1.data(), vec![6.0, 15.0]); // [1+2+3, 4+5+6]
        assert_eq!(r1.shape(), vec![2]);
        assert!(!r1.requires_grad());

        // Sum axis 1, keep_dims=true -> Shape [2, 1]
        let r1k = t.sum_axes(&[1], true);
        assert_eq!(r1k.data(), vec![6.0, 15.0]);
        assert_eq!(r1k.shape(), vec![2, 1]);
        assert!(!r1k.requires_grad());

        // Sum axes 0 and 1 (all axes) -> Shape [1]
        let r01 = t.sum_axes(&[0, 1], false);
        assert_eq!(r01.data(), vec![21.0]); 
        assert_eq!(r01.shape(), vec![1]);
        assert!(!r01.requires_grad());

        // Sum axes 0 and 1 (all axes), keep_dims=true -> Shape [1, 1]
        let r01k = t.sum_axes(&[0, 1], true);
        assert_eq!(r01k.data(), vec![21.0]);
        assert_eq!(r01k.shape(), vec![1, 1]);
        assert!(!r01k.requires_grad());
    }
    
    #[test]
    fn test_sum_axes_propagate_requires_grad() {
        let t_grad = create_test_tensor_with_grad::<f32>(vec![1., 2., 3., 4.], vec![2, 2]);
        let t_no_grad = create_test_tensor::<f32>(vec![5., 6., 7., 8.], vec![2, 2]);

        // If any input requires grad, output requires grad
        let r0 = t_grad.sum_axes(&[0], false);
        assert!(r0.requires_grad());
        assert!(r0.grad_fn().is_some());
        
        let r1 = t_no_grad.sum_axes(&[0], false);
        assert!(!r1.requires_grad());
        assert!(r1.grad_fn().is_none());
    }
    
    #[test]
    fn test_sum_axes_backward() {
        let t = create_test_tensor_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // --- Test sum axis 0, keep_dims=false --- 
        let r0 = t.sum_axes(&[0], false); // Shape [3], Data [5, 7, 9]
        let upstream0 = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]); // Grad for shape [3]
        
        // Manually call backward
        let grad_fn0 = r0.grad_fn().clone().unwrap();
        grad_fn0.backward(&upstream0);
        
        let grad0 = t.grad().expect("Grad missing axis 0");
        // Expected grad: upstream broadcasted back to shape [2, 3]
        // [[10, 20, 30], [10, 20, 30]]
        assert_eq!(grad0.data(), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        assert_eq!(grad0.shape(), vec![2, 3]);
        t.borrow_tensor_data_mut().grad = None; // Clear grad for next test

        // --- Test sum axis 1, keep_dims=true --- 
        let r1k = t.sum_axes(&[1], true); // Shape [2, 1], Data [6, 15]
        let upstream1k = Tensor::new(vec![100.0, 200.0], vec![2, 1]); // Grad for shape [2, 1]
        
        let grad_fn1k = r1k.grad_fn().clone().unwrap();
        grad_fn1k.backward(&upstream1k);
        
        let grad1k = t.grad().expect("Grad missing axis 1 keep_dims");
        // Expected grad: upstream broadcasted back to shape [2, 3]
        // [[100, 100, 100], [200, 200, 200]]
        assert_eq!(grad1k.data(), vec![100.0, 100.0, 100.0, 200.0, 200.0, 200.0]);
        assert_eq!(grad1k.shape(), vec![2, 3]);
        t.borrow_tensor_data_mut().grad = None; // Clear grad
        
        // --- Test sum all axes, keep_dims=true --- 
        let r_all_k = t.sum_axes(&[0, 1], true); // Shape [1, 1], Data [21]
        let upstream_all_k = Tensor::new(vec![5.0], vec![1, 1]); // Grad for shape [1, 1]
        
        let grad_fn_all_k = r_all_k.grad_fn().clone().unwrap();
        grad_fn_all_k.backward(&upstream_all_k);
        
        let grad_all_k = t.grad().expect("Grad missing all axes keep_dims");
        // Expected grad: upstream broadcasted back to shape [2, 3]
        // [[5, 5, 5], [5, 5, 5]]
        assert_eq!(grad_all_k.data(), vec![5.0; 6]);
        assert_eq!(grad_all_k.shape(), vec![2, 3]);
    }

    // TODO: Add more tests for sum_axes forward/backward on higher rank tensors
} 