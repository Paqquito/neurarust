use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
// use crate::device::StorageDevice; // Removed unused import
use crate::error::NeuraRustError;
use crate::ops::reduction::sum::sum_axes;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{FromPrimitive, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Neg};
use std::sync::{Arc, RwLock};

// --- MeanBackward Definition ---

/// Backward operation context for `mean_axes`.
#[derive(Debug)]
struct MeanBackward<T: Debug + Copy + Send + Sync + 'static> {
    input_node: Arc<RwLock<TensorData<T>>>,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    keep_dims: bool,
    n: T, // Number of elements averaged (as type T)
}

// --- BackwardOp Implementation for MeanBackward ---

impl<T> BackwardOp<T> for MeanBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Zero
        + One
        + Add<Output = T>
        + AddAssign
        + Div<Output = T>
        + Neg<Output = T>
        + Default
        + PartialEq
        + std::iter::Sum
        + PartialOrd,
{
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // Calculate gradient scaled by 1/N
        let one_over_n = T::one() / self.n;
        let one_over_n_tensor = Tensor::full(grad_output.shape(), one_over_n)?;
        let scaled_grad = crate::ops::arithmetic::mul::mul_op(
            grad_output,
            &one_over_n_tensor,
        )?;

        // Reuse the broadcasting logic similar to SumBackward
        let target_shape = self.input_shape.clone();
        let target_rank = target_shape.len();

        // 1. Reshape scaled_grad if keep_dims was false
        let grad_reshaped = if !self.keep_dims && !self.axes.is_empty() {
            let mut shape_with_kept_dims = Vec::with_capacity(target_rank);
            let mut current_grad_dim = 0;
            for i in 0..target_rank {
                if self.axes.contains(&i) {
                    shape_with_kept_dims.push(1);
                } else {
                    if current_grad_dim >= scaled_grad.shape().len()
                        || scaled_grad.shape()[current_grad_dim] != target_shape[i]
                    {
                        return Err(NeuraRustError::ShapeMismatch {
                            expected: target_shape.clone(),
                            actual: scaled_grad.shape(),
                            operation: "mean_axes_backward (reshape)".to_string(),
                        });
                    }
                    shape_with_kept_dims.push(target_shape[i]);
                    current_grad_dim += 1;
                }
            }
            crate::ops::view::reshape_op(&scaled_grad, shape_with_kept_dims)?
        } else if self.axes.is_empty() && !self.keep_dims {
            let shape_all_ones = vec![1; target_rank];
            crate::ops::view::reshape_op(&scaled_grad, shape_all_ones)?
        } else {
            scaled_grad.clone()
        };

        // 2. Ensure Contiguous Gradient for Expansion (Manual Copy)
        let grad_for_expand = {
            let guard = grad_reshaped.read_data();
            if guard.is_contiguous() {
                grad_reshaped.clone()
            } else {
                // Manual contiguous copy logic (same as in SumBackward)
                let shape = guard.shape.clone();
                let numel = shape.iter().product::<usize>();
                let mut new_data = Vec::with_capacity(numel);
                let buffer_arc = guard.data.cpu_data()?.clone();
                let data_slice = buffer_arc.as_slice();
                let mut current_indices = vec![0; shape.len()];
                for _ in 0..numel {
                    let offset = guard.get_offset(&current_indices);
                    new_data.push(data_slice[offset]);
                    if numel > 0 {
                        let mut dim_to_increment = shape.len();
                        while dim_to_increment > 0 {
                            dim_to_increment -= 1;
                            current_indices[dim_to_increment] += 1;
                            if current_indices[dim_to_increment] < shape[dim_to_increment] {
                                break;
                            }
                            current_indices[dim_to_increment] = 0;
                        }
                    }
                }
                Tensor::new(new_data, shape)?
            }
        };

        // 3. Expand the contiguous grad_for_expand to the target shape (Manual Expand)
        if grad_for_expand.shape() == target_shape {
            return Ok(vec![grad_for_expand]);
        }

        let grad_guard = grad_for_expand.read_data();
        let grad_buffer = grad_guard.data.cpu_data()?.clone();
        let grad_shape = grad_guard.shape.clone();
        let grad_strides = grad_guard.strides.clone();
        let grad_offset = grad_guard.offset;
        let grad_rank = grad_shape.len();

        if target_rank != grad_rank {
            return Err(NeuraRustError::InternalError(
                format!("Mean backward rank mismatch: target {}, grad {}", target_rank, grad_rank)
            ));
        }

        let input_grad_numel = target_shape.iter().product::<usize>();
        let mut input_grad_data = vec![T::zero(); input_grad_numel];
        let target_strides = TensorData::<T>::calculate_contiguous_strides(&target_shape);
        let mut current_target_indices = vec![0; target_rank];

        for _ in 0..input_grad_numel {
            let mut grad_relative_offset = 0;
            for dim_idx in 0..target_rank {
                let source_index = if grad_shape[dim_idx] == 1 && target_shape[dim_idx] > 1 {
                    0
                } else {
                    current_target_indices[dim_idx]
                };
                grad_relative_offset += source_index * grad_strides[dim_idx];
            }
            let grad_logical_offset = grad_offset + grad_relative_offset;
            let val = grad_buffer[grad_logical_offset];

            let mut target_flat_idx = 0;
            for dim_idx in 0..target_rank {
                target_flat_idx += current_target_indices[dim_idx] * target_strides[dim_idx];
            }
            input_grad_data[target_flat_idx] = val;

            if input_grad_numel > 0 {
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
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- mean_op Implementation ---

/// Calculates the mean of elements along specified axes.
/// Currently supports CPU only.
pub fn mean_op<T>(
    input: &Tensor<T>,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Clone
        + Zero
        + One // For division in backward
        + AddAssign
        + Div<Output = T> // For division by N
        + FromPrimitive // To convert usize N to T
        + Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Default
        + PartialEq
        + PartialOrd
        + std::iter::Sum
        + Add<Output = T>
        + Neg<Output = T>,
{
    // Calculate sum using sum_axes
    let sum_result = sum_axes(input, axes, keep_dims)?;

    // Calculate N (number of elements summed over)
    let n = {
        let input_shape = input.shape();
        let input_rank = input_shape.len();
        if axes.is_empty() {
            input.numel()
        } else {
            let mut count = 1;
            for &axis in axes {
                if axis < input_rank {
                    count *= input_shape[axis];
                }
                // Ignore invalid axes for count, sum_axes would have errored already
            }
            count
        }
    };

    // Convert N to type T
    let n_t = T::from_usize(n).ok_or_else(|| {
        NeuraRustError::InternalError("Failed to convert element count N to tensor type T".to_string())
    })?;

    // Divide sum by N
    let n_tensor = Tensor::full(sum_result.shape(), n_t)?;
    let mean_result = crate::ops::arithmetic::div::div_op(&sum_result, &n_tensor)?;

    // --- Autograd Integration ---
    if input.requires_grad() {
        let grad_fn = MeanBackward {
            input_node: input.data.clone(),
            input_shape: input.shape(),
            axes: axes.to_vec(),
            keep_dims,
            n: n_t,
        };
        mean_result.set_grad_fn(Some(Arc::new(grad_fn)))?;
         mean_result.set_requires_grad(true)?;
    }

    Ok(mean_result)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use approx::assert_relative_eq;

    fn create_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn test_mean_all() {
        let t = create_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = mean_op(&t, &[], false).unwrap();
        assert_eq!(result.shape(), vec![]);
        assert_relative_eq!(result.get(&[]).unwrap(), 3.5);
    }

    #[test]
    fn test_mean_axis_0() {
        let t = create_tensor_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = mean_op(&t, &[0], false).unwrap();
        assert_eq!(result.shape(), vec![3]);
        let expected_data = vec![2.5, 3.5, 4.5]; // (1+4)/2, (2+5)/2, (3+6)/2
        let res_data = result.read_data().data.cpu_data().unwrap().clone();
        res_data
            .iter()
            .zip(expected_data.iter())
            .for_each(|(r, e)| assert_relative_eq!(*r, *e));
    }

    // TODO: Add tests for keep_dims

    // --- Autograd Tests ---
    use crate::autograd::grad_check::check_grad;

    fn create_tensor_f64_with_grad(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let t = Tensor::new(data, shape).unwrap();
        t.set_requires_grad(true).unwrap();
        t
    }

    #[test]
    fn test_mean_all_backward() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor<f64>]| mean_op(&inputs[0], &[], false);

        let output_shape = vec![];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Mean all backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_mean_axis_0_backward() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor<f64>]| mean_op(&inputs[0], &[0], false);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean axis 0 backward grad check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_mean_axis_1_keep_dims_backward() {
        let input = create_tensor_f64_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let func = |inputs: &[Tensor<f64>]| mean_op(&inputs[0], &[1], true);

        let output_shape = vec![2, 1];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean axis 1 keep_dims backward grad check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_mean_multiple_axes_backward() {
        let input = create_tensor_f64_with_grad((1..=24).map(|x| x as f64).collect(), vec![2, 3, 4]);
        let func = |inputs: &[Tensor<f64>]| mean_op(&inputs[0], &[0, 2], false);

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Mean multiple axes backward grad check failed: {:?}", grad_check_result.err());
    }
} 