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
use crate::tensor::utils::calculate_strides; // Mise à jour de l'import
use std::fmt::Debug;
use std::collections::HashMap;

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

impl<T: Debug> Tensor<T> {
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
        let _input_strides = calculate_strides(&input_shape);
        let output_strides = calculate_strides(&output_shape);

        // --- Reduction Logic ---
        let mut current_input_coords = vec![0; rank];
        for input_linear_idx in 0..input_td.numel() {
            // Calculate output coordinates based on input coordinates
            let mut current_output_coords = Vec::with_capacity(output_shape.len());
            let mut _output_coord_idx = 0;
            for input_axis in 0..rank {
                if !axes_set.contains(&input_axis) {
                    // Dimension not reduced, copy coordinate
                    current_output_coords.push(current_input_coords[input_axis]);
                    _output_coord_idx += 1;
                } else if keep_dims {
                    // Dimension reduced, but kept, coordinate is 0
                    current_output_coords.push(0);
                    _output_coord_idx += 1;
                }
                // If dimension is reduced and not kept, skip it
            }

            // Calculate output linear index (handle scalar case)
            let output_linear_idx = if output_shape.is_empty() { // Check for empty shape
                0 // Scalar output always maps to index 0
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
            result.data.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
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
    T: Clone + Debug + AddAssign + Zero + Copy + 'static,
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
                
                let grad_expanded = Tensor::new(local_grad_data, self.input_shape.clone());
                accumulate_gradient(gradients, &self.input_ref, grad_expanded); 
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}

// --- Helper Function (copied from add.rs) ---
fn accumulate_gradient<T>(
    gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>,
    input_weak_ref: &Weak<RefCell<TensorData<T>>>,
    local_gradient: Tensor<T>,
)
where
    T: AddAssign + Clone + Debug + Zero + Copy + 'static,
{
    if let Some(input_rc) = input_weak_ref.upgrade() {
        let input_ptr = Rc::as_ptr(&input_rc);
        gradients.entry(input_ptr)
            .and_modify(|existing_grad| { *existing_grad += &local_gradient; })
            .or_insert(local_gradient);
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
    use std::iter::Sum;

    // --- Helpers --- (Ajouter bounds manquants si nécessaire)
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Sum + Default>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
     fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Sum + Default + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
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
    fn test_sum_forward() {
         let t1 = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
         // Utiliser sum_axes
         let res_no_axes_no_keep = t1.sum_axes(&[], false); // Appelait t1.sum(None, false)
         assert_eq!(res_no_axes_no_keep.data().to_vec(), vec![21.0]);
         assert_eq!(res_no_axes_no_keep.shape(), vec![]); 

         let res_axes_0_no_keep = t1.sum_axes(&[0], false); // Appelait t1.sum(Some(vec![0]), false)
         assert_eq!(res_axes_0_no_keep.data().to_vec(), vec![5.0, 7.0, 9.0]);
         assert_eq!(res_axes_0_no_keep.shape(), vec![3]);

         let res_axes_1_keep = t1.sum_axes(&[1], true); // Appelait t1.sum(Some(vec![1]), true)
         assert_eq!(res_axes_1_keep.data().to_vec(), vec![6.0, 15.0]);
         assert_eq!(res_axes_1_keep.shape(), vec![2, 1]);
    }

    #[test]
    fn test_sum_grad_propagation() {
        let t_grad = create_test_tensor_with_grad::<f32>(vec![1., 2., 3., 4.], vec![2, 2]);
        let t_no_grad = create_test_tensor::<f32>(vec![5., 6., 7., 8.], vec![2, 2]);

        // Utiliser sum_axes
        let r0 = t_grad.sum_axes(&[], false); // Appelait t_grad.sum(None, false)
        assert!(r0.requires_grad());
        assert!(r0.grad_fn().is_some());
        
        let r1 = t_no_grad.sum_axes(&[], false); // Appelait t_no_grad.sum(None, false)
        assert!(!r1.requires_grad());
        assert!(r1.grad_fn().is_none());
    }

    #[test]
    fn test_sum_backward_multiple_inputs_simplified() { 
        let t1 = Tensor::new_with_grad(vec![1.0_f32, 2.0], vec![2]);
        t1.zero_grad(); 
                
        // Utiliser sum_axes
        let sum1 = t1.sum_axes(&[], false); // Appelait t1.sum(None, false)
        let sum1_clone = t1.sum_axes(&[], false); // Appelait t1.sum(None, false)
        
        let final_sum = &sum1 + &sum1_clone; 
        final_sum.backward(None); 

        let grad_t1 = t1.grad().expect("Grad t1 missing");
        assert_eq!(grad_t1.data().to_vec(), vec![2.0, 2.0]);
        assert_eq!(grad_t1.shape(), vec![2]);
    }

    #[test]
    fn test_sum_backward_specific_axes_and_keepdim() {
        let t = Tensor::new_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Utiliser sum_axes
        let r0 = t.sum_axes(&[0], false); 
        let upstream0 = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]); 
        let grad_fn0 = r0.grad_fn().clone().unwrap();
        let mut gradients0 = HashMap::new();
        gradients0.insert(Rc::as_ptr(&r0.data), upstream0.clone()); 
        grad_fn0.backward(&upstream0, &mut gradients0);
        let grad0 = gradients0.get(&Rc::as_ptr(&t.data)).expect("Grad missing axis 0"); 
        assert_eq!(grad0.data().to_vec(), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        assert_eq!(grad0.shape(), vec![2, 3]);

        let r1k = t.sum_axes(&[1], true); 
        let upstream1k = Tensor::new(vec![100.0, 200.0], vec![2, 1]); 
        let grad_fn1k = r1k.grad_fn().clone().unwrap();
        let mut gradients1k = HashMap::new();
        gradients1k.insert(Rc::as_ptr(&r1k.data), upstream1k.clone());
        grad_fn1k.backward(&upstream1k, &mut gradients1k);
        let grad1k = gradients1k.get(&Rc::as_ptr(&t.data)).expect("Grad missing axis 1 keep_dims");
        assert_eq!(grad1k.data().to_vec(), vec![100.0, 100.0, 100.0, 200.0, 200.0, 200.0]);
        assert_eq!(grad1k.shape(), vec![2, 3]);
        
        let r_all_k = t.sum_axes(&[0, 1], true); 
        let upstream_all_k = Tensor::new(vec![5.0], vec![1, 1]); 
        let grad_fn_all_k = r_all_k.grad_fn().clone().unwrap();
        let mut gradients_all_k = HashMap::new();
        gradients_all_k.insert(Rc::as_ptr(&r_all_k.data), upstream_all_k.clone());
        grad_fn_all_k.backward(&upstream_all_k, &mut gradients_all_k);
        let grad_all_k = gradients_all_k.get(&Rc::as_ptr(&t.data)).expect("Grad missing all axes keep_dims");
        assert_eq!(grad_all_k.data().to_vec(), vec![5.0; 6]);
        assert_eq!(grad_all_k.shape(), vec![2, 3]);
    }
    
    #[test]
    fn test_sum_backward_simple() {
        let t = Tensor::new_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        // Utiliser sum_axes
        let result = t.sum_axes(&[], false); // Appelait t.sum(None, false)
        let grad_fn = result.grad_fn().clone().unwrap();
        let upstream_grad = Tensor::new(vec![1.0_f32], vec![]); 
        let mut gradients = HashMap::new();
        gradients.insert(Rc::as_ptr(&result.data), upstream_grad.clone());
        grad_fn.backward(&upstream_grad, &mut gradients); 
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2, 2]);
        assert_eq!(grad_t.data().to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward_keepdim() {
        let t = Tensor::new_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        // Utiliser sum_axes
        let result = t.sum_axes(&[1], true); // Appelait t.sum(Some(vec![1]), true)
        let grad_fn = result.grad_fn().clone().unwrap();
        let upstream_grad = Tensor::new(vec![10.0_f32, 20.0], vec![2, 1]);
        let mut gradients = HashMap::new();
        gradients.insert(Rc::as_ptr(&result.data), upstream_grad.clone());
        grad_fn.backward(&upstream_grad, &mut gradients); 
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2, 2]);
        assert_eq!(grad_t.data().to_vec(), vec![10.0, 10.0, 20.0, 20.0]);
    }

    #[test]
    fn test_sum_backward_axis() {
        let t = Tensor::new_with_grad(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        // Utiliser sum_axes
        let result = t.sum_axes(&[0], false); // Appelait t.sum(Some(vec![0]), false)
        let grad_fn = result.grad_fn().clone().unwrap();
        let upstream_grad = Tensor::new(vec![10.0_f32, 20.0], vec![2]);
        let mut gradients = HashMap::new();
        gradients.insert(Rc::as_ptr(&result.data), upstream_grad.clone());
        grad_fn.backward(&upstream_grad, &mut gradients); 
        let grad_t = gradients.get(&Rc::as_ptr(&t.data)).expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2, 2]);
        assert_eq!(grad_t.data().to_vec(), vec![10.0, 20.0, 10.0, 20.0]);
    }

} 