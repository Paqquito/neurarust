use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::device::StorageDevice;
use crate::ops::reduction::sum::sum_kernel;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{FromPrimitive, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Neg, Sub, Mul};
use std::sync::{Arc, RwLockReadGuard};
use std::iter::Sum;
use crate::tensor::broadcast_utils::expand_kernel;

// --- MeanBackward Definition ---

/// Backward operation context for `mean_axes`.
#[derive(Debug)]
struct MeanBackward<T: Debug + Copy + Send + Sync + 'static> {
    input: Tensor<T>,
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
        + Mul<Output = T>
        + Default
        + PartialEq
        + std::iter::Sum
        + PartialOrd
        + Sub<Output = T>
        + std::iter::Product,
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

        // 3. Expand the contiguous grad_for_expand to the target shape using kernel.
        if grad_for_expand.shape() == target_shape {
            return Ok(vec![grad_for_expand]);
        }

        // --- Call expand_kernel --- 
        let grad_guard = grad_for_expand.read_data();
        let grad_data_slice = grad_guard.data.cpu_data()?.clone();
        let grad_data_slice = grad_data_slice.as_slice();
        let grad_shape = &grad_guard.shape;
        let grad_strides = &grad_guard.strides;
        let grad_offset = grad_guard.offset;

        let expanded_data = expand_kernel(
            &target_shape,
            grad_data_slice,
            grad_shape,
            grad_strides,
            grad_offset,
        )?;

        // Drop guard after use
        drop(grad_guard);

        let input_gradient = Tensor::new(expanded_data, target_shape)?;
        Ok(vec![input_gradient])
    }

    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input.get_node_id()]
    }
}

// --- Kernel de Calcul ---

/// Noyau de calcul privé pour la moyenne avec réduction d'axes.
fn mean_kernel<T>(
    input_guard: &RwLockReadGuard<'_, TensorData<T>>,
    input_data_slice: &[T],
    axes: &[usize],
    keep_dims: bool,
    output_shape: &[usize], // La shape après réduction
    n: T,                   // Le nombre d'éléments moyennés (déjà calculé)
) -> Result<Vec<T>, NeuraRustError>
where
    T: Copy
        + Debug
        + Zero
        + AddAssign
        + Div<Output = T> // Ajout pour la division
        + Send // Ajoutés pour correspondre aux appels internes si nécessaire
        + Sync
        + PartialEq // Ajout pour la comparaison n == T::zero()
        + 'static,
{
    // 1. Calculer la somme en utilisant le kernel de sum
    let sum_data = sum_kernel(input_guard, input_data_slice, axes, keep_dims, output_shape)?;

    // 2. Diviser chaque élément par N
    // Vérifier si N est zéro pour éviter la division par zéro.
    // Bien que N soit généralement > 0, une bonne pratique est de vérifier.
    if n == T::zero() {
        // Que faire ? Retourner une erreur ou un vecteur de zéros/NaN ?
        // Retourner une erreur semble plus sûr.
        return Err(NeuraRustError::DivisionByZero);
    }

    let mean_data: Vec<T> = sum_data.into_iter().map(|val| val / n).collect();

    Ok(mean_data)
}

// --- mean_op Implementation ---

/// Calculates the mean of elements along specified axes.
/// Requires the tensor to be on CPU.
/// Supports autograd.
pub fn mean_axes<T>(
    input: &Tensor<T>,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + Sub<Output = T>
        + FromPrimitive
        + Add<Output = T>
        + AddAssign
        + Sum
        + PartialOrd
        + std::iter::Product
        + Send
        + Sync
        + 'static
        + Debug
        + Copy
        + Clone
        + Default
        + PartialEq
        + Zero
        + One,
{
    // --- Autograd Setup ---
    let requires_grad = input.requires_grad();
    let mut input_maybe_clone: Option<Tensor<T>> = None;
    if requires_grad {
        input_maybe_clone = Some(input.clone());
    }

    // --- Acquire read lock ---
    let input_guard = input.read_data();

    // --- Device Check ---
    let device = input_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Mean operation is currently only supported on CPU, not {:?}",
            device
        )));
    }

    // --- Get CPU Data Buffer ---
    let input_data_arc = input_guard.data.cpu_data()?.clone();
    let input_data_slice = input_data_arc.as_slice();

    // --- Shape and Axis Validation / Calculation de N ---
    let input_shape = input_guard.shape.clone();
    let input_rank = input_shape.len();

    let processed_axes = {
        let mut pa = Vec::new();
        if !axes.is_empty() {
            for &axis in axes {
                if axis >= input_rank {
                    drop(input_guard);
                    return Err(NeuraRustError::IndexOutOfBounds {
                        index: vec![axis],
                        shape: input_shape.clone(),
                    });
                }
                pa.push(axis);
            }
            pa.sort_unstable();
            pa.dedup();
        }
        pa
    };

    let n = {
        if processed_axes.is_empty() {
            input_guard.numel()
        } else {
            processed_axes.iter().map(|&axis| input_shape[axis]).product()
        }
    };
    let n_t = T::from_usize(n).ok_or_else(|| {
        NeuraRustError::InternalError("Failed to convert element count N to tensor type T".to_string())
    })?;
    if n_t == T::zero() && n > 0 {
         return Err(NeuraRustError::InternalError("Calculated N > 0 but T::from_usize(N) resulted in T::zero()".to_string()));
    }

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
            shape = vec![]; // Reduce to scalar
        } else if shape.is_empty() && keep_dims && input_rank > 0 {
            // keep_dims=true and all axes reduced or input was scalar
             shape = vec![1; input_rank];
        } else if shape.is_empty() && input_rank == 0 {
             // Input was scalar, output is scalar
             shape = vec![];
        } else if processed_axes.is_empty() && !keep_dims {
            shape = vec![]; // mean_all sans keep_dims -> scalaire
        }
        shape
    };


    // --- Perform Mean Calculation (Appel au Kernel) ---
    let result_data = mean_kernel(
        &input_guard,
        input_data_slice,
        &processed_axes,
        keep_dims,
        &output_shape,
        n_t, // Passer n_t calculé
    )?;

    // Drop lock
    drop(input_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data, output_shape)?;


    // --- Autograd Integration ---
    if requires_grad {
        // On utilise input_maybe_clone.unwrap() car on a vérifié requires_grad
        let input_clone = input_maybe_clone.unwrap();
        let grad_fn = MeanBackward {
            input: input_clone.clone(), // Passer le Tensor cloné
            input_shape: input_clone.shape(), // Obtenir la shape du clone
            axes: processed_axes,
            keep_dims,
            n: n_t,
        };
        result_tensor.set_grad_fn(Some(Arc::new(grad_fn)))?;
         result_tensor.set_requires_grad(true)?;
    }

    Ok(result_tensor)
}

#[path = "mean_test.rs"]
mod tests; 