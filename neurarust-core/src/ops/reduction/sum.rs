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

/// Backward operation context for `sum_axes`.
#[derive(Debug)]
struct SumAxesBackward {
    input_node: Arc<RwLock<TensorData>>,
    input_shape: Vec<usize>, // Original shape of the input tensor
    axes: Vec<usize>,     // Axes along which summation was performed
    keep_dims: bool,      // Whether dims were kept in the output
}

// --- BackwardOp Implementation for SumAxesBackward ---

impl BackwardOp for SumAxesBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let target_shape = self.input_shape.clone();
        let target_rank = target_shape.len();

        // Ensure grad_output is F32 CPU
        let grad_output_guard = grad_output.read_data();
        if grad_output_guard.dtype != DType::F32 || grad_output_guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::UnsupportedOperation(
                "Sum backward currently only supports F32 CPU gradients.".to_string()
            ));
        }

        // 1. Prepare the grad_output: Reshape if keep_dims was false to match rank.
        let grad_output_reshaped = if !self.keep_dims && !self.axes.is_empty() {
            let mut shape_with_kept_dims = Vec::with_capacity(target_rank);
            let mut current_grad_dim = 0;
            for i in 0..target_rank {
                if self.axes.contains(&i) {
                    shape_with_kept_dims.push(1);
                } else {
                    if current_grad_dim >= grad_output_guard.shape.len()
                        || grad_output_guard.shape[current_grad_dim] != target_shape[i]
                    {
                        return Err(NeuraRustError::ShapeMismatch {
                            expected: format!("{:?}", target_shape),
                            actual: format!("{:?}", grad_output_guard.shape),
                            operation: "sum_axes_backward (reshape)".to_string(),
                        });
                    }
                    shape_with_kept_dims.push(target_shape[i]);
                    current_grad_dim += 1;
                }
            }
            // Release guard before reshape
            drop(grad_output_guard);
            crate::ops::view::reshape_op(grad_output, shape_with_kept_dims)?
        } else if self.axes.is_empty() && !self.keep_dims {
             // Release guard before reshape
            drop(grad_output_guard);
            let shape_all_ones = vec![1; target_rank];
            crate::ops::view::reshape_op(grad_output, shape_all_ones)?
        } else {
             // Release guard, clone original tensor
            drop(grad_output_guard);
            grad_output.clone()
        };

        // 2. Ensure gradient for expansion is contiguous (Manual Copy)
        let grad_output_for_expand = {
            let guard = grad_output_reshaped.read_data();
            if guard.is_contiguous() {
                 // Drop guard, clone tensor
                drop(guard);
                grad_output_reshaped.clone()
            } else {
                // Manual contiguous copy logic (assuming F32)
                let shape = guard.shape.clone();
                let numel = guard.numel();
                let mut new_data = vec![0.0f32; numel]; // Initialize with 0.0f32
                // Use correct buffer access
                let buffer_arc = guard.buffer().try_get_cpu_f32()?.clone();
                let data_slice = buffer_arc.as_slice();
                let mut current_indices = vec![0usize; shape.len()];
                for idx in 0..numel {
                    let offset = guard.get_offset(&current_indices);
                    if offset < data_slice.len() {
                        new_data[idx] = data_slice[offset]; // Direct assignment for f32
                    } else {
                         return Err(NeuraRustError::InternalError(
                            "Index out of bounds during contiguous copy in sum backward".to_string()
                         ));
                    }

                    // Increment indices (standard contiguous iteration logic)
                    if numel > 0 && idx < numel - 1 {
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
                // Drop guard before creating new tensor
                let tensor_shape = guard.shape.clone();
                drop(guard);
                Tensor::new(new_data, tensor_shape)?
            }
        };

        // 3. Expand the (now contiguous) grad_output_for_expand to the target shape using kernel.
        if grad_output_for_expand.shape() == target_shape {
            return Ok(vec![grad_output_for_expand]);
        }

        // --- Call expand_kernel --- 
        let grad_guard = grad_output_for_expand.read_data();
        // Ensure it's F32 CPU before getting slice
        if grad_guard.dtype != DType::F32 || grad_guard.device != StorageDevice::CPU {
             return Err(NeuraRustError::UnsupportedOperation(
                "Sum backward expand kernel requires F32 CPU gradient.".to_string()
             ));
        }
        // Use correct buffer access
        let grad_data_arc = grad_guard.buffer().try_get_cpu_f32()?.clone();
        let grad_data_slice = grad_data_arc.as_slice();
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

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Noyau de calcul privé pour la somme avec réduction d'axes.
pub(crate) fn sum_kernel(
    input_guard: &RwLockReadGuard<'_, TensorData>,
    input_data_slice: &[f32],
    axes: &[usize],
    keep_dims: bool,
    output_shape: &[usize],
) -> Result<Vec<f32>, NeuraRustError> {
    let output_numel: usize = if output_shape.is_empty() { 1 } else { output_shape.iter().product::<usize>() };
    let mut result_data = vec![0.0f32; output_numel];

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
        let val = input_data_slice[logical_offset];

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

        // Accumuler la valeur (f32 addition)
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

/// Calculates the sum of elements along specified axes.
/// Requires the tensor to be on CPU.
/// Supports autograd.
pub fn sum_axes(
    input: &Tensor,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor, NeuraRustError> {
    let input_guard = input.read_data();

    // --- Device and DType Checks ---
    if input_guard.device != StorageDevice::CPU || input_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "sum_axes currently only supports F32 CPU tensors.".to_string(),
        ));
    }

    let input_shape = &input_guard.shape;
    let input_rank = input_shape.len();

    // --- Axis Validation & Processing ---
    // Si l'utilisateur passe un slice vide et que le tenseur n'est pas scalaire, cela signifie "tous les axes".
    let axes_to_process: Vec<usize> = if axes.is_empty() && input_rank > 0 {
        (0..input_rank).collect()
    } else {
        axes.to_vec()
    };

    let mut processed_axes = Vec::with_capacity(axes_to_process.len());
    for &axis in &axes_to_process { // Utiliser axes_to_process ici
        if axis >= input_rank {
            return Err(NeuraRustError::DimensionMismatch {
                expected: input_rank, // Rank
                actual: axis,      // Invalid axis
            });
        }
        // Handle potential duplicate axes gracefully
        if !processed_axes.contains(&axis) {
            processed_axes.push(axis);
        }
    }
    processed_axes.sort_unstable(); // Sorting helps in consistent processing

    // --- Calculate Output Shape ---
    // Utiliser processed_axes qui contient maintenant les axes corrects.
    let output_shape: Vec<usize> = if processed_axes.len() == input_rank { // Tous les axes sont réduits
        if keep_dims {
            // La shape est de rang input_rank, avec que des 1
            // Sauf si le rang initial était 0 (scalaire), auquel cas on garde []
            if input_rank == 0 { vec![] } else { vec![1; input_rank] }
        } else {
            vec![] // Scalaire
        }
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| {
                if processed_axes.contains(&i) {
                    if keep_dims { Some(1) } else { None } // Keep dim as 1 or remove
                } else {
                    Some(dim) // Keep original dimension
                }
            })
            .collect()
    };

    // --- Extract Data Slice ---
    let input_data_arc = input_guard.buffer().try_get_cpu_f32()?.clone();
    let input_data_slice = input_data_arc.as_slice();

    // --- Call Kernel ---
    // Need to clone required fields before dropping guard
    let kernel_axes = processed_axes.clone(); // kernel_axes contient maintenant tous les axes si nécessaire
    let kernel_output_shape = output_shape.clone();
    let requires_grad = input.requires_grad();
    let input_node_arc = if requires_grad { Some(input.data.clone()) } else { None };
    let original_input_shape = input_guard.shape.clone(); // For backward pass

    let result_data = sum_kernel(
        &input_guard, // Pass the guard reference itself
        input_data_slice,
        &kernel_axes, // Utilise la liste d'axes correctement traitée
        keep_dims,
        &kernel_output_shape,
    )?;

    // Drop guard after kernel call
    drop(input_guard);

    // --- Create Output Tensor ---
    let output_tensor = Tensor::new(result_data, kernel_output_shape)?;

    // --- Autograd Integration ---
    if requires_grad {
        if let Some(input_arc) = input_node_arc {
            let grad_fn = SumAxesBackward {
                input_node: input_arc,
                input_shape: original_input_shape,
                axes: kernel_axes, // Utilise la liste d'axes correctement traitée
                keep_dims,
            };
            let mut output_write_guard = output_tensor.write_data();
            output_write_guard.grad_fn = Some(Arc::new(grad_fn));
            output_write_guard.requires_grad = true;
        } else {
            return Err(NeuraRustError::InternalError(
                "SumAxes requires grad but input Arc<TensorData> was not available".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::NeuraRustError,
        tensor::Tensor,
        buffer::{Buffer, CpuBuffer},
    };
    use crate::utils::testing::check_tensor_near;
    use approx::assert_relative_eq;

    // Helper function to extract Vec<f32> from Tensor for checking
    fn get_f32_data(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
        let locked_data = tensor.data.read().map_err(|e| NeuraRustError::LockError {
            lock_type: "read".to_string(),
            reason: format!("Failed to read tensor data in helper: {}", e),
        })?;
        match &*locked_data.buffer {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => Ok(data_arc.to_vec()),
            _ => Err(NeuraRustError::UnsupportedOperation("Helper requires CpuF32 buffer".to_string())),
        }
    }

    #[test]
    fn test_sum_all() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = sum_axes(&t, &[], false)?;
        let result_data = get_f32_data(&result)?;
        assert_eq!(result.shape(), &[] as &[usize], "Result shape should be scalar");
        assert_relative_eq!(result_data[0], 21.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_axis0() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = sum_axes(&t, &[0], false)?;
        let expected_data = vec![5.0, 7.0, 9.0];
        check_tensor_near(&result, &[3], &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_axis1_keepdims() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = sum_axes(&t, &[1], true)?;
        let expected_data = vec![6.0, 15.0];
        check_tensor_near(&result, &[2, 1], &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_multiple_axes() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
        let result = sum_axes(&t, &[0, 2], false)?;
        let expected_data = vec![60.0, 92.0, 124.0];
        check_tensor_near(&result, &[3], &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_invalid_axis() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?;
        let result = sum_axes(&t, &[1], false);
        assert!(matches!(result, Err(NeuraRustError::DimensionMismatch { .. })));
        Ok(())
    }

    // --- Test Non Contigu --- 
    #[test]
    fn test_sum_all_non_contiguous() -> Result<(), NeuraRustError> {
        // Créer un tenseur 2x3
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        // Le transposer pour le rendre non contigu
        let t_transposed = crate::ops::view::transpose_op(&t, 0, 1)?; 
        assert!(!t_transposed.is_contiguous(), "Transposed tensor should not be contiguous");
        assert_eq!(t_transposed.shape(), &[3, 2]);
        assert_eq!(t_transposed.strides(), &[1, 3], "Strides mismatch after transpose"); 

        // Calculer la somme globale sur le tenseur non contigu via sum_op
        let sum_result = sum_op(&t_transposed, None, false)?;
        
        // Vérifier le résultat
        let sum_data = get_f32_data(&sum_result)?;
        assert_eq!(sum_result.shape(), &[] as &[usize], "Sum result shape should be scalar");
        assert_eq!(sum_data.len(), 1, "Sum result should have 1 element");
        assert_relative_eq!(sum_data[0], 21.0, epsilon = 1e-6);

        Ok(())
    }
}

/// Computes the sum of all elements in the tensor or along specified axes.
/// This is a convenience wrapper.
pub(crate) fn sum_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let all_axes: Vec<usize> = (0..tensor.shape().len()).collect();
    let axes_to_sum = axes.unwrap_or(&all_axes);
    sum_axes(tensor, axes_to_sum, keep_dims)
}
