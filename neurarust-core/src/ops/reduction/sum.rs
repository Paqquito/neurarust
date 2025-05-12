use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use std::fmt::Debug;
use std::sync::{Arc, RwLock, RwLockReadGuard};
use crate::tensor::broadcast_utils::expand_kernel;
use super::utils::process_reduction_axes;
use super::utils::calculate_reduction_output_shape;
use super::utils::calculate_grad_broadcast_shape;

// Removed simple iterator imports for now, sum needs more complex logic
// use crate::tensor::iter_utils::{NdArraySimpleIter, NdArraySimpleIterF64};

/// Backward operation context for sum reduction.
///
/// Stores information needed to compute the gradient of the sum operation:
/// - The original input tensor's data (`input_node`).
/// - The original shape of the input tensor (`input_shape`).
/// - The axes along which the summation was performed (`axes`).
/// - Whether the reduced dimensions were kept in the output (`keep_dims`).
#[derive(Debug)]
struct SumAxesBackward {
    input_node: Arc<RwLock<TensorData>>,
    input_shape: Vec<usize>, // Original shape of the input tensor
    axes: Vec<usize>,     // Axes along which summation was performed
    keep_dims: bool,      // Whether dims were kept in the output
}

// --- BackwardOp Implementation for SumAxesBackward ---

impl BackwardOp for SumAxesBackward {
    /// Computes the gradient for the sum reduction operation.
    ///
    /// The gradient of the sum operation with respect to its input is essentially the
    /// incoming gradient (`grad_output`) broadcasted back to the original input shape.
    /// If `keep_dims` was false during the forward pass, the `grad_output` is first
    /// reshaped to include the reduced dimensions (as size 1) before broadcasting.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original sum operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with a single element: the gradient
    /// with respect to the original input tensor. Returns an error if broadcasting or
    /// device operations fail.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let grad_output_guard = grad_output.read_data();
        let grad_output_shape = grad_output_guard.shape.clone(); // Obtenir shape avant drop
        let grad_dtype = grad_output_guard.dtype;

        // --- Device Check --- (peut être factorisé aussi? Pour l'instant, inchangé)
        if grad_output_guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                operation: "sum_op_backward".to_string(),
                expected: StorageDevice::CPU,
                actual: grad_output_guard.device,
            });
        }
        drop(grad_output_guard); // Drop guard avant reshape

        // --- Reshape grad_output if needed --- (Utiliser la nouvelle fonction)
        let target_broadcast_shape = calculate_grad_broadcast_shape(
            &self.input_shape,
            &grad_output_shape,
            &self.axes,
            self.keep_dims,
        );

        let grad_output_reshaped = if grad_output_shape != target_broadcast_shape {
            // S'assurer que grad_output est contigu *avant* le reshape si nécessaire ?
            // Le reshape actuel peut échouer sur non-contigu.
            // Pour l'instant, supposons que reshape gère ou que grad_output est souvent contigu.
            // TODO: Vérifier si grad_output.contiguous() est nécessaire avant reshape
            crate::ops::view::reshape_op(grad_output, target_broadcast_shape)?
        } else {
            grad_output.clone()
        };

        // --- Ensure Contiguous Gradient --- (Utiliser Tensor::contiguous)
        let contiguous_grad = grad_output_reshaped.contiguous()?;
        let contiguous_grad_guard = contiguous_grad.read_data();

        // --- Expand Gradient using Kernel --- (Logique inchangée, mais utilise le grad contigu)
        let input_gradient = match grad_dtype {
            DType::F32 => {
                let grad_data_slice = contiguous_grad_guard.buffer().try_get_cpu_f32()?.as_slice();
                let expanded_data = expand_kernel(
                    &self.input_shape, // Target shape pour l'expansion
                    grad_data_slice,
                    &contiguous_grad_guard.shape, // Shape du grad contigu (après reshape)
                    &contiguous_grad_guard.strides, // Strides du grad contigu
                    contiguous_grad_guard.offset, // Offset du grad contigu
                )?;
                drop(contiguous_grad_guard);
                Tensor::new(expanded_data, self.input_shape.clone())?
            }
            DType::F64 => {
                 let grad_data_slice = contiguous_grad_guard.buffer().try_get_cpu_f64()?.as_slice();
                 let expanded_data = expand_kernel(
                    &self.input_shape, // Target shape
                    grad_data_slice,
                    &contiguous_grad_guard.shape,
                    &contiguous_grad_guard.strides,
                    contiguous_grad_guard.offset,
                )?;
                drop(contiguous_grad_guard);
                Tensor::new_f64(expanded_data, self.input_shape.clone())?
            }
            DType::I32 | DType::I64 | DType::Bool => {
                return Err(NeuraRustError::UnsupportedOperation(
                    "sum_op n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                );
            }
        };

        Ok(vec![input_gradient])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Noyau de calcul privé pour la somme avec réduction d'axes.
/// Rendue générique sur le type numérique T.
pub(crate) fn sum_kernel<T>(
    input_guard: &RwLockReadGuard<'_, TensorData>,
    input_data_slice: &[T],
    axes: &[usize],
    keep_dims: bool,
    output_shape: &[usize],
) -> Result<Vec<T>, NeuraRustError>
where
    T: Copy + Default + std::ops::AddAssign + Debug // Traits requis pour l'opération
{
    let output_numel: usize = if output_shape.is_empty() { 1 } else { output_shape.iter().product::<usize>() };
    let mut result_data = vec![T::default(); output_numel]; // Utilise T::default()

    let input_shape = &input_guard.shape;
    let input_rank = input_shape.len();
    let input_strides = &input_guard.strides;
    let input_offset = input_guard.offset;

    let mut current_input_indices = vec![0usize; input_rank];

    // Itérer sur tous les éléments de l'entrée
    for _i in 0..input_guard.numel() {
        let mut current_relative_offset: usize = 0;
        for dim_idx in 0..input_rank {
            current_relative_offset += current_input_indices[dim_idx] * input_strides[dim_idx];
        }
        let logical_offset: usize = input_offset + current_relative_offset;

        if logical_offset >= input_data_slice.len() {
            return Err(NeuraRustError::InternalError(format!(
                "Sum kernel index out of bounds. Coords: {:?}, Offset: {}, DataLen: {}",
                 current_input_indices,
                 logical_offset,
                 input_data_slice.len()
            )));
        }
        let val: T = input_data_slice[logical_offset]; // Le type est T

        // Calculer l'index de sortie (indices multi-dimensionnels)
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

        // Calculer l'index linéaire de sortie (output_flat_idx)
        let mut output_flat_idx: usize = 0;
        if !output_shape.is_empty() {
            // Calculer les strides pour le output_shape (contigu attendu)
            let mut output_strides = vec![1; output_shape.len()];
            for i in (0..output_shape.len() - 1).rev() {
                output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            }

            // Calculer l'index linéaire en utilisant les strides et les indices
            for j in 0..output_shape.len() {
                if j < output_indices.len() { // Assurer que l'indice existe
                     output_flat_idx += output_indices[j] * output_strides[j];
                } else {
                     // Erreur si output_indices n'a pas la bonne longueur
                     return Err(NeuraRustError::InternalError(format!(
                        "Sum kernel output index calculation error: j={} out of bounds for indices len {}",
                        j, output_indices.len()
                    )));
                }
            }
        } // else: output_shape is empty, output_flat_idx reste 0 pour le scalaire

        // Accumuler la valeur (Utilise AddAssign pour T)
        if output_flat_idx < result_data.len() {
            result_data[output_flat_idx] += val;
        } else {
            // Logique d'erreur si l'index calculé est hors limites (ne devrait pas arriver si output_numel est correct)
            return Err(NeuraRustError::InternalError(format!(
                "Sum kernel output flat index {} out of bounds for result_data len {}",
                output_flat_idx, result_data.len()
            )));
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

/// Performs element-wise sum reduction along specified axes.
///
/// This is a crate-internal function, typically called via the `Tensor::sum` method.
/// It calculates the sum of elements of a tensor `t` along the given `axes`.
///
/// # Arguments
///
/// * `t` - The input tensor.
/// * `axes` - An optional slice of `usize` specifying the axes along which to reduce.
///   If `None`, the sum is calculated over all elements, resulting in a scalar tensor.
/// * `keep_dims` - A boolean indicating whether to keep the reduced dimensions in the
///   output tensor's shape (with size 1). If `false`, the reduced dimensions are removed.
///
/// # Returns
///
/// A `Result` containing the reduced `Tensor`. Returns an error if:
/// *   An axis is out of bounds.
/// *   Device operations fail.
/// *   Autograd graph operations fail.
///
/// # Example (Conceptual - Use `Tensor::sum` instead)
///
/// ```rust,ignore
/// // Assuming t is a Tensor of shape [2, 3]
/// // use crate::ops::reduction::sum::sum_op; // Assuming direct access
///
/// // Sum along axis 0
/// let sum_axis0 = sum_op(&t, Some(&[0]), false)?; // Shape [3]
///
/// // Sum along axis 1
/// let sum_axis1 = sum_op(&t, Some(&[1]), true)?; // Shape [2, 1]
///
/// // Sum all elements
/// let sum_all = sum_op(&t, None, false)?; // Shape [] (scalar)
/// ```
///
/// // Example ignored as doc-test: illustrative purpose
/// ```rust, ignore
/// use neurarust_core::{Tensor, tensor, DType};
/// use neurarust_core::ops::reduction::sum_op;
///
pub(crate) fn sum_op(
    t: &Tensor,
    axes: Option<&[usize]>,
    keep_dims: bool,
) -> Result<Tensor, NeuraRustError> {
    let t_guard = t.read_data();

    // --- Device Check --- 
    if t_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch { operation: "sum_op".to_string(), expected: StorageDevice::CPU, actual: t_guard.device });
    }

    // --- DType Check ---
    let dtype = t_guard.dtype;
    if dtype != DType::F32 && dtype != DType::F64 && dtype != DType::I32 && dtype != DType::I64 && dtype != DType::Bool {
        return Err(NeuraRustError::UnsupportedOperation(format!("Sum operation only supports F32, F64, I32, I64, and Bool, got {:?}", dtype)));
    }

    // --- Process Axes --- 
    let rank = t_guard.shape.len();
    let axes_vec = process_reduction_axes(rank, axes)?;

    // --- Calculate Output Shape --- (Utiliser la nouvelle fonction)
    let output_shape = calculate_reduction_output_shape(&t_guard.shape, &axes_vec, keep_dims);

    // --- Autograd Info ---
    let requires_grad = t_guard.requires_grad;
    let input_shape = t_guard.shape.clone(); // Toujours nécessaire pour Backward
    let input_node_arc = if requires_grad { Some(t.data.clone()) } else { None };

    // --- Dispatch based on DType --- (Appelle sum_kernel avec la nouvelle output_shape)
    let output_tensor = match dtype {
        DType::F32 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f32()?.as_slice();
            let result_data = sum_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec, 
                keep_dims,
                &output_shape,
            )?;
            drop(t_guard);
            Tensor::new(result_data, output_shape)?
        }
        DType::F64 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f64()?.as_slice();
             let result_data = sum_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec, 
                keep_dims,
                &output_shape,
            )?;
            drop(t_guard);
            Tensor::new_f64(result_data, output_shape)?
        }
        DType::I32 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_i32()?.as_slice();
            let result_data = sum_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec,
                keep_dims,
                &output_shape,
            )?;
            drop(t_guard);
            Tensor::new_i32(result_data, output_shape)?
        }
        DType::I64 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_i64()?.as_slice();
            let result_data = sum_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec,
                keep_dims,
                &output_shape,
            )?;
            drop(t_guard);
            Tensor::new_i64(result_data, output_shape)?
        }
        DType::Bool => {
            let input_data_slice = t_guard.buffer().try_get_cpu_bool()?.as_slice();
            // On convertit chaque booléen en i64 (true=1, false=0)
            let input_as_i64: Vec<i64> = input_data_slice.iter().map(|&b| if b { 1 } else { 0 }).collect();
            let result_data = sum_kernel(
                &t_guard,
                &input_as_i64,
                &axes_vec,
                keep_dims,
                &output_shape,
            )?;
            drop(t_guard);
            Tensor::new_i64(result_data, output_shape)?
        }
    };

    // --- Autograd Setup --- (inchangé)
    if requires_grad {
        if let Some(node_arc) = input_node_arc {
            let grad_fn = SumAxesBackward {
                input_node: node_arc,
                input_shape, // Garder la shape originale
                axes: axes_vec, // Utiliser les axes traités
                keep_dims,
            };
            let mut output_guard = output_tensor.write_data();
            output_guard.requires_grad = true;
            output_guard.grad_fn = Some(Arc::new(grad_fn));
        } else {
            return Err(NeuraRustError::InternalError(
                "Sum op requires grad but input Arc Node unavailable".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// Link the tests from the separate file
#[cfg(test)]
#[path = "sum_test.rs"]
mod tests;
