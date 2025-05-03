use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
// use crate::buffer::Buffer; // Removed unused import
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::One;
use num_traits::Zero;
use std::cmp::PartialEq;
use std::cmp::PartialOrd; // Added for create_test_tensor
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use std::sync::RwLockReadGuard;

/// Backward operation context for `sum_axes`.
/// Stores information needed to compute the gradient for the input tensor.
#[derive(Debug)]
struct SumAxesBackward<T: Debug + Copy + Send + Sync + 'static> {
    input: Tensor<T>,
    input_shape: Vec<usize>, // Original shape of the input tensor
    axes: Vec<usize>,     // Axes along which summation was performed
    keep_dims: bool,      // Whether dims were kept in the output
}

// --- BackwardOp Implementation for SumAxesBackward ---

impl<T> BackwardOp<T> for SumAxesBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Zero
        + One
        + AddAssign
        + Default
        + PartialEq
        + Sum
        + Add<Output = T>
        + PartialOrd,
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let target_shape = self.input_shape.clone();
        let target_rank = target_shape.len();

        // 1. Prepare the grad_output: Reshape if keep_dims was false to match rank.
        let grad_output_reshaped = if !self.keep_dims && !self.axes.is_empty() {
            let mut shape_with_kept_dims = Vec::with_capacity(target_rank);
            let mut current_grad_dim = 0;
            for i in 0..target_rank {
                if self.axes.contains(&i) {
                    shape_with_kept_dims.push(1);
                } else {
                    // Ensure grad_output shape matches non-reduced dims
                    if current_grad_dim >= grad_output.shape().len()
                        || grad_output.shape()[current_grad_dim] != target_shape[i]
                    {
                        return Err(NeuraRustError::ShapeMismatch {
                            expected: target_shape.clone(), // Or a more specific expected shape
                            actual: grad_output.shape(),
                            operation: "sum_axes_backward (reshape)".to_string(),
                        });
                    }
                    shape_with_kept_dims.push(target_shape[i]);
                    current_grad_dim += 1;
                }
            }
            crate::ops::view::reshape_op(grad_output, shape_with_kept_dims)?
        } else if self.axes.is_empty() && !self.keep_dims {
            // Special case: Summing all elements without keep_dims -> scalar grad_output.
            // Reshape the scalar grad_output to have the rank of the input, with all dimensions as 1.
            let shape_all_ones = vec![1; target_rank];
            crate::ops::view::reshape_op(grad_output, shape_all_ones)?
        } else {
            // If keep_dims was true or axes was empty (sum all with keep_dims), grad_output already has the correct rank.
            grad_output.clone()
        };

        // 2. Utiliser directement le gradient reshaped pour l'expansion
        let grad_output_for_expand = grad_output_reshaped;

        // 3. Expand the grad_output_for_expand to the target shape.
        if grad_output_for_expand.shape() == target_shape {
            return Ok(vec![grad_output_for_expand]);
        }

        // --- Manual Expand Implementation (devrait fonctionner avec non-contigu) ---
        let grad_output_guard = grad_output_for_expand.read_data();
        let grad_output_buffer = grad_output_guard.data.cpu_data()?.clone();
        let grad_output_shape = grad_output_guard.shape.clone();
        let grad_output_strides = grad_output_guard.strides.clone();
        let grad_output_offset = grad_output_guard.offset;
        let grad_rank = grad_output_shape.len(); // Rank of the source gradient

        // Check ranks match (should after reshape)
        if target_rank != grad_rank {
             return Err(NeuraRustError::InternalError(format!(
                 "Internal Error in SumBackward: Rank mismatch before expansion. Target: {}, Grad: {}",
                 target_rank, grad_rank
             )));
        }

        let input_grad_numel = target_shape.iter().product::<usize>();
        let mut input_grad_data = vec![T::zero(); input_grad_numel];
        let target_strides = crate::tensor_data::TensorData::<T>::calculate_contiguous_strides(&target_shape);
        let mut current_target_indices = vec![0; target_rank];

        for _target_linear_idx in 0..input_grad_numel {
            // Calculate source offset based on target indices and broadcasting rules
            let mut grad_output_relative_offset = 0;
            for dim_idx in 0..target_rank { // Iterate dimensions
                let source_index_for_dim = if grad_output_shape[dim_idx] == 1 && target_shape[dim_idx] > 1 {
                    0 // Dimension was broadcasted, use index 0
                } else {
                    current_target_indices[dim_idx] // Dimension matched, use target index
                };
                grad_output_relative_offset += source_index_for_dim * grad_output_strides[dim_idx];
            }
            let grad_output_logical_offset = grad_output_offset + grad_output_relative_offset;
            let val = grad_output_buffer[grad_output_logical_offset];

            // Calculate target offset
            let mut input_grad_flat_idx = 0;
            for dim_idx in 0..target_rank {
                input_grad_flat_idx += current_target_indices[dim_idx] * target_strides[dim_idx];
            }
            input_grad_data[input_grad_flat_idx] = val;

            // Increment target indices
            if input_grad_numel > 0 && _target_linear_idx < input_grad_numel - 1 {
                let mut dim_to_increment = target_rank;
                while dim_to_increment > 0 {
                    dim_to_increment -= 1;
                    current_target_indices[dim_to_increment] += 1;
                    if current_target_indices[dim_to_increment] < target_shape[dim_to_increment] {
                        break;
                    }
                    current_target_indices[dim_to_increment] = 0;
                }
            }
        }

        let input_gradient = Tensor::new(input_grad_data, target_shape)?;
        Ok(vec![input_gradient])
    }

    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input.get_node_id()]
    }
}

/// Noyau de calcul privé pour la somme avec réduction d'axes.
pub(crate) fn sum_kernel<T>(
    input_guard: &RwLockReadGuard<'_, TensorData<T>>,
    input_data_slice: &[T],
    axes: &[usize], // Utiliser les axes déjà traités
    keep_dims: bool,
    output_shape: &[usize],
) -> Result<Vec<T>, NeuraRustError>
where
    T: Copy + Debug + Zero + AddAssign,
{
    let output_numel: usize = if output_shape.is_empty() { 1 } else { output_shape.iter().product::<usize>() };
    let mut result_data = vec![T::zero(); output_numel];

    let input_shape = &input_guard.shape;
    let input_rank = input_shape.len();
    let input_strides = &input_guard.strides;
    let input_offset = input_guard.offset;

    let mut current_input_indices = vec![0; input_rank];

    // Itérer sur tous les éléments de l'entrée
    for _i in 0..input_guard.numel() {
        // Calculer l'offset et obtenir la valeur
        let mut current_relative_offset = 0;
        for dim_idx in 0..input_rank {
            current_relative_offset += current_input_indices[dim_idx] * input_strides[dim_idx];
        }
        let logical_offset = input_offset + current_relative_offset;
        let val = input_data_slice[logical_offset];

        // Calculer l'index de sortie
        let mut output_indices = Vec::with_capacity(output_shape.len());
        let mut output_idx_pos = 0;
        for (dim_idx, &coord) in current_input_indices.iter().enumerate() {
            if !axes.contains(&dim_idx) {
                if output_idx_pos < output_shape.len() {
                    output_indices.push(coord);
                    output_idx_pos += 1;
                }
            } else if keep_dims {
                if output_idx_pos < output_shape.len() {
                    output_indices.push(0);
                    output_idx_pos += 1;
                }
            }
        }

        // Calculer l'index plat de sortie
        let mut output_flat_idx = 0;
        if !output_shape.is_empty() {
            let mut stride_product: usize = 1;
            // Utiliser directement output_shape.len() ici car output_indices a la bonne taille
            for j in (0..output_shape.len()).rev() {
                output_flat_idx += output_indices[j] * stride_product;
                 // Le calcul des strides de sortie se fait implicitement par cette boucle
                if j > 0 {
                    // On a besoin de la taille de la dimension j pour calculer le stride
                    // C'est output_shape[j] car on itère à l'envers
                     stride_product *= output_shape[j];
                }
            }
        }

        // Accumuler la valeur
        if output_flat_idx < result_data.len() {
            result_data[output_flat_idx] += val;
        }

        // Incrémenter les indices d'entrée
        if input_guard.numel() > 0 && _i < input_guard.numel() - 1 {
            let mut dim_to_increment = input_rank;
            while dim_to_increment > 0 {
                dim_to_increment -= 1;
                current_input_indices[dim_to_increment] += 1;
                if current_input_indices[dim_to_increment] < input_shape[dim_to_increment] {
                    break;
                }
                current_input_indices[dim_to_increment] = 0;
            }
        }
    }
    Ok(result_data)
}

/// Calculates the sum of elements along specified axes.
/// Requires the tensor to be on CPU.
/// Supports autograd.
pub fn sum_axes<T>(
    input: &Tensor<T>,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Clone
        + Zero
        + AddAssign
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Default
        + PartialEq
        + PartialOrd
        + One
        + Sum
        + Add<Output = T>,
{
    // --- Autograd Setup ---
    let requires_grad = input.requires_grad();
    let mut input_maybe_clone: Option<Tensor<T>> = None;
    let mut input_shape_clone: Option<Vec<usize>> = None;
    if requires_grad {
        input_maybe_clone = Some(input.clone());
        input_shape_clone = Some(input.shape());
    }

    // --- Acquire read lock ---
    let input_guard = input.read_data();

    // --- Device Check ---
    let device = input_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Summation is currently only supported on CPU, not {:?}",
            device
        )));
    }
    // --- Get CPU Data Buffer ---
    let input_data_arc = input_guard.data.cpu_data()?.clone();
    let input_data_slice = input_data_arc.as_slice();

    // --- Shape and Axis Validation ---
    let input_shape = input_guard.shape.clone();
    let input_rank = input_shape.len();

    // --- Handle Sum All Case ---
    if axes.is_empty() {
        // Sum all elements using the cpu data arc
        let sum_val = input_data_arc.iter().map(|x| *x).sum::<T>();
        let output_shape = if keep_dims {
            vec![1; input_rank]
        } else {
            vec![]
        };
        // Result tensor on CPU
        let result_tensor = Tensor::new(vec![sum_val], output_shape)?;

        // --- Autograd Integration ---
        if requires_grad {
            let backward_op = SumAxesBackward {
                input: input_maybe_clone.unwrap(),
                input_shape: input_shape_clone.unwrap(),
                axes: axes.to_vec(),
                keep_dims,
            };
            result_tensor.set_grad_fn(Some(Arc::new(backward_op)))?;
            result_tensor.set_requires_grad(true)?;
        }

        return Ok(result_tensor);
    }

    // --- Validate Axes ---
    let mut processed_axes = Vec::with_capacity(axes.len());
    for &axis in axes {
        if axis >= input_rank {
            drop(input_guard);
            return Err(NeuraRustError::IndexOutOfBounds {
                index: vec![axis],
                shape: input_shape,
            });
        }
        processed_axes.push(axis);
    }
    processed_axes.sort_unstable();
    processed_axes.dedup();

    // --- Calculate Output Shape ---
    let output_shape = {
        let mut shape = Vec::new();
        for (dim, &size) in input_shape.iter().enumerate() {
            if !processed_axes.contains(&dim) {
                shape.push(size);
            } else if keep_dims {
                shape.push(1);
            }
        }
        if shape.is_empty() && !processed_axes.is_empty() && !keep_dims {
            shape = vec![];
        } else if shape.is_empty() && keep_dims && input_rank > 0 {
            shape = vec![1; input_rank];
        }
        shape
    };

    // --- Perform Summation (Appel au Kernel) ---
    let result_data = sum_kernel(
        &input_guard,
        input_data_slice,
        &processed_axes,
        keep_dims,
        &output_shape,
    )?;

    // Drop lock
    drop(input_guard);

    // Create result tensor
    let result_tensor = Tensor::new(result_data, output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        let backward_op = SumAxesBackward {
            input: input_maybe_clone.unwrap(),
            input_shape: input_shape_clone.unwrap(),
            axes: processed_axes,
            keep_dims,
        };
        result_tensor.set_grad_fn(Some(Arc::new(backward_op)))?;
        result_tensor.set_requires_grad(true)?;
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;
    use approx::assert_relative_eq;
    use crate::utils::testing::{create_test_tensor};

    #[test]
    fn test_sum_all() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[], false).unwrap();
        assert_eq!(result.shape(), vec![]); // Scalar shape
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_relative_eq!(res_cpu_data[0], 21.0);
    }

    #[test]
    fn test_sum_axis_0() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[0], false).unwrap();
        assert_eq!(result.shape(), vec![3]);
        let expected_data = vec![5.0, 7.0, 9.0];
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_axis_1() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[1], false).unwrap();
        assert_eq!(result.shape(), vec![2]);
        let expected_data = vec![6.0, 15.0];
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_axes_multiple() {
        let t = create_test_tensor(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        );
        // Sum over axes 0 and 2
        let result = sum_axes(&t, &[0, 2], false).unwrap();
        assert_eq!(result.shape(), vec![2]);
        let expected_data = vec![14.0, 22.0];
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_keep_dims() {
        let t = create_test_tensor(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = sum_axes(&t, &[0], true).unwrap();
        assert_eq!(result.shape(), vec![1, 2]);
        let expected_data = vec![4.0, 6.0];
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        let result_all = sum_axes(&t, &[], true).unwrap();
        assert_eq!(result_all.shape(), vec![1, 1]);
        let res_all_buffer_arc = result_all.borrow_data_buffer();
        let res_all_cpu_data = res_all_buffer_arc
            .cpu_data()
            .expect("Result all not on CPU");
        assert_relative_eq!(res_all_cpu_data[0], 10.0);
    }

    #[test]
    fn test_sum_invalid_axis() {
        let t = create_test_tensor(vec![1.0_f32, 2.0], vec![2]);
        let result = sum_axes(&t, &[1], false);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::IndexOutOfBounds { index, shape } => {
                assert_eq!(index, vec![1]);
                assert_eq!(shape, vec![2]);
            }
            _ => panic!("Expected IndexOutOfBounds error"),
        }
    }
}

#[cfg(test)]
mod autograd_tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::utils::testing::{create_test_tensor_with_grad};

    #[test]
    fn test_sum_axes_backward_simple_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[0], true)
        };

        let output_shape = vec![1, 3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

     #[test]
    fn test_sum_axes_backward_simple_no_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[1], false)
        };

        let output_shape = vec![2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

    #[test]
    fn test_sum_all_backward_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f32>]| -> Result<Tensor<f32>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[], true)
        };

        let output_shape = vec![1, 1];
        let output_grad = Tensor::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 5e-2;

        let grad_check_result = check_grad(func, &[input.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_sum_all_backward_no_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f32>]| -> Result<Tensor<f32>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[], false)
        };

        let output_shape = vec![];
        let output_grad = Tensor::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 5e-2;

        let grad_check_result = check_grad(func, &[input.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_sum_multiple_axes_backward() {
        let input = create_test_tensor_with_grad((1..=24).map(|x| x as f64).collect::<Vec<_>>(), vec![2, 3, 4]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[0, 2], false)
        };

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

      #[test]
     fn test_sum_no_reduction_backward() {
         let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
         input.set_requires_grad(true).unwrap();

         let epsilon = 1e-5;
         let tolerance = 5e-3;

         let func1 = |inputs: &[Tensor<f32>]| {
             assert_eq!(inputs.len(), 1);
             sum_axes(&inputs[0], &[], false)
        };
         let output1_shape = vec![];
         let output1_grad = Tensor::ones(output1_shape).unwrap();
         let check1 = check_grad(func1, &[input.clone()], &output1_grad, epsilon, tolerance);
         assert!(check1.is_ok(), "Sum all (scalar) grad check failed: {:?}", check1.err());
         input.clear_grad();

         let func2 = |inputs: &[Tensor<f32>]| {
             assert_eq!(inputs.len(), 1);
             sum_axes(&inputs[0], &[], true)
         };
         let output2_shape = vec![1];
         let output2_grad = Tensor::ones(output2_shape).unwrap();
         let check2 = check_grad(func2, &[input.clone()], &output2_grad, epsilon, tolerance);
          assert!(check2.is_ok(), "Sum all (keep_dims) grad check failed: {:?}", check2.err());
     }
}
