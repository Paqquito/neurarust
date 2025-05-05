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

        // --- DType Handling --- 
        let grad_output_guard = grad_output.read_data();
        let grad_dtype = grad_output_guard.dtype;

        // --- Device Check --- 
        if grad_output_guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                operation: "sum_axes_backward".to_string(),
                expected: StorageDevice::CPU,
                actual: grad_output_guard.device,
            });
        }

        // --- Reshape grad_output if necessary --- 
        let grad_output_reshaped = if !self.keep_dims && !self.axes.is_empty() {
            let mut shape_with_kept_dims = Vec::with_capacity(target_rank);
            let mut current_grad_dim = 0;
            for i in 0..target_rank {
                if self.axes.contains(&i) {
                    shape_with_kept_dims.push(1);
                } else {
                    if current_grad_dim >= grad_output_guard.shape.len() {
                         return Err(NeuraRustError::ShapeMismatch {
                            expected: format!("{:?}", target_shape),
                            actual: format!("{:?}", grad_output_guard.shape),
                            operation: "sum_axes_backward (reshape)".to_string(),
                        });
                    }
                    // Validate dimensions BEFORE pushing
                    if grad_output_guard.shape[current_grad_dim] != target_shape[i] {
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
            drop(grad_output_guard);
            crate::ops::view::reshape_op(grad_output, shape_with_kept_dims)?
        } else if self.axes.is_empty() && !self.keep_dims {
            let shape_all_ones = vec![1; target_rank];
            drop(grad_output_guard);
            crate::ops::view::reshape_op(grad_output, shape_all_ones)?
        } else {
            drop(grad_output_guard);
            grad_output.clone()
        };
        
        // Re-acquire guard after potential reshape
        let reshaped_grad_guard = grad_output_reshaped.read_data();

        // --- Dispatch based on DType for Contiguous Copy and Expansion --- 
        let input_gradient = match grad_dtype {
            DType::F32 => {
                // --- Ensure Contiguous F32 --- 
                let (contiguous_grad_data_arc, contiguous_grad_shape, contiguous_grad_strides, contiguous_grad_offset) = 
                    if reshaped_grad_guard.is_contiguous() {
                        (reshaped_grad_guard.buffer().try_get_cpu_f32()?.clone(), 
                         reshaped_grad_guard.shape.clone(), 
                         reshaped_grad_guard.strides.clone(), 
                         reshaped_grad_guard.offset)
                    } else {
                         // Manual contiguous copy F32
                        let shape = reshaped_grad_guard.shape.clone();
                        let numel = reshaped_grad_guard.numel();
                        let mut new_data = vec![0.0f32; numel];
                        let buffer_arc = reshaped_grad_guard.buffer().try_get_cpu_f32()?.clone();
                        let data_slice = buffer_arc.as_slice();
                        let mut current_indices = vec![0usize; shape.len()];
                        for idx in 0..numel {
                            let offset = reshaped_grad_guard.get_offset(&current_indices);
                            if offset < data_slice.len() { new_data[idx] = data_slice[offset]; } 
                            else { return Err(NeuraRustError::InternalError(
                                "Index out of bounds during contiguous copy in sum backward".to_string()
                            )); }
                            // Increment indices
                            if numel > 0 && idx < numel - 1 {
                                let mut dim_to_increment = shape.len();
                                while dim_to_increment > 0 {
                                    dim_to_increment -= 1;
                                    current_indices[dim_to_increment] += 1;
                                    if current_indices[dim_to_increment] < shape[dim_to_increment] { break; }
                                    current_indices[dim_to_increment] = 0;
                                }
                             }
                        }
                        let new_buffer = Arc::new(new_data);
                        // Contiguous tensor has standard strides and zero offset
                        let strides = TensorData::calculate_contiguous_strides(&shape);
                        (new_buffer, shape, strides, 0)
                    };
                drop(reshaped_grad_guard); // Drop guard before kernel call

                // --- Expand F32 using Generic Kernel ---
                let grad_data_slice = contiguous_grad_data_arc.as_slice();
                let expanded_data = expand_kernel(
                    &target_shape,
                    grad_data_slice,
                    &contiguous_grad_shape,
                    &contiguous_grad_strides,
                    contiguous_grad_offset,
                )?;
                Tensor::new(expanded_data, target_shape)?
            }
            DType::F64 => {
                 // --- Ensure Contiguous F64 --- 
                let (contiguous_grad_data_arc, contiguous_grad_shape, contiguous_grad_strides, contiguous_grad_offset) = 
                    if reshaped_grad_guard.is_contiguous() {
                        (reshaped_grad_guard.buffer().try_get_cpu_f64()?.clone(), 
                         reshaped_grad_guard.shape.clone(), 
                         reshaped_grad_guard.strides.clone(), 
                         reshaped_grad_guard.offset)
                    } else {
                         // Manual contiguous copy F64
                        let shape = reshaped_grad_guard.shape.clone();
                        let numel = reshaped_grad_guard.numel();
                        let mut new_data = vec![0.0f64; numel]; // Use f64
                        let buffer_arc = reshaped_grad_guard.buffer().try_get_cpu_f64()?.clone(); // Use f64
                        let data_slice = buffer_arc.as_slice();
                        let mut current_indices = vec![0usize; shape.len()];
                        for idx in 0..numel {
                            let offset = reshaped_grad_guard.get_offset(&current_indices);
                            if offset < data_slice.len() { new_data[idx] = data_slice[offset]; } // Use f64
                            else { return Err(NeuraRustError::InternalError(
                                "Index out of bounds during contiguous copy in sum backward".to_string()
                            )); }
                            // Increment indices (same logic)
                            if numel > 0 && idx < numel - 1 {
                                let mut dim_to_increment = shape.len();
                                while dim_to_increment > 0 {
                                    dim_to_increment -= 1;
                                    current_indices[dim_to_increment] += 1;
                                    if current_indices[dim_to_increment] < shape[dim_to_increment] { break; }
                                    current_indices[dim_to_increment] = 0;
                                }
                            }
                        }
                        let new_buffer = Arc::new(new_data);
                        let strides = TensorData::calculate_contiguous_strides(&shape);
                        (new_buffer, shape, strides, 0)
                    };
                drop(reshaped_grad_guard); // Drop guard before kernel call

                 // --- Expand F64 using Generic Kernel ---
                let grad_data_slice = contiguous_grad_data_arc.as_slice();
                let expanded_data = expand_kernel(
                    &target_shape,
                    grad_data_slice,
                    &contiguous_grad_shape,
                    &contiguous_grad_strides,
                    contiguous_grad_offset,
                )?;
                Tensor::new_f64(expanded_data, target_shape)?
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

/// Calcule la somme d'un tenseur le long des axes spécifiés.
pub(crate) fn sum_axes(
    input: &Tensor,
    axes: &[usize],
    keep_dims: bool,
) -> Result<Tensor, NeuraRustError> {
    let input_guard = input.read_data();

    // Device Check
    if input_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "sum_axes".to_string(),
            expected: StorageDevice::CPU,
            actual: input_guard.device,
        });
    }

    // Validate axes
    let input_rank = input_guard.shape.len();
    for &axis in axes {
        if axis >= input_rank {
            return Err(NeuraRustError::ShapeMismatch {
                operation: "sum_axes (axis validation)".to_string(),
                expected: format!("axis < {}", input_rank),
                actual: format!("axis {}", axis),
            });
        }
    }

    // Calculate output shape
    let mut output_shape = Vec::new();
    if keep_dims {
        for (i, &dim_size) in input_guard.shape.iter().enumerate() {
            if axes.contains(&i) {
                output_shape.push(1);
            } else {
                output_shape.push(dim_size);
            }
        }
    } else {
        if axes.len() == input_rank && input_rank > 0 {
            // Summing all axes, result is scalar
            output_shape.clear(); // Shape is []
        } else {
            // Keep dimensions that are *not* in axes
            for (i, &dim_size) in input_guard.shape.iter().enumerate() {
                if !axes.contains(&i) {
                    output_shape.push(dim_size);
                }
            }
            // If input was scalar, output is scalar
            if input_rank == 0 {
                 output_shape.clear();
            }
        }
    }
    
    let final_output_shape = output_shape; // Use the corrected shape

    // --- Prepare for Autograd ---
    let requires_grad = input_guard.requires_grad;
    let input_node_arc = if requires_grad { Some(Arc::clone(&input.data)) } else { None };
    let input_shape_clone = input_guard.shape.clone();
    let axes_clone = axes.to_vec(); // Clone axes needed for backward
    let keep_dims_clone = keep_dims;

    // --- DType Dispatch for Computation --- 
    let output_tensor = match input_guard.dtype {
        DType::F32 => {
            let input_buffer_arc = input_guard.buffer().try_get_cpu_f32()?;
            let input_data_slice = input_buffer_arc.as_slice();
            // Call generic kernel
            let result_data = sum_kernel(
                &input_guard,
                input_data_slice,
                &axes_clone,
                keep_dims_clone,
                &final_output_shape // Use the potentially corrected shape
            )?;
            drop(input_guard);
            Tensor::new(result_data, final_output_shape)? // Use the potentially corrected shape
        }
        DType::F64 => {
            let input_buffer_arc = input_guard.buffer().try_get_cpu_f64()?;
            let input_data_slice = input_buffer_arc.as_slice();
             // Call generic kernel
            let result_data = sum_kernel(
                &input_guard,
                input_data_slice,
                &axes_clone,
                keep_dims_clone,
                &final_output_shape // Use the potentially corrected shape
            )?;
            drop(input_guard);
            Tensor::new_f64(result_data, final_output_shape)? // Use the potentially corrected shape
        }
    };

    // --- Autograd Setup ---
    if requires_grad {
        if let Some(node_arc) = input_node_arc {
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                lock_type: "write".to_string(),
                reason: "Failed to lock output TensorData for write (autograd setup in sum_axes)".to_string(),
            })?;
            output_data_write_guard.requires_grad = true;
            let backward_op = SumAxesBackward {
                input_node: node_arc,
                input_shape: input_shape_clone,
                axes: axes_clone,
                keep_dims: keep_dims_clone,
            };
            output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in sum_axes".to_string()));
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
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let expected_sum = 21.0;
        // Appel via sum_op avec axes=None pour sommer sur tout
        let result = sum_op(&t, None, false)?;

        // Assertions
        assert_eq!(result.shape(), &[] as &[usize], "Result shape should be scalar");
        let data = get_f32_data(&result)?;
        assert_eq!(data.len(), 1, "Result data should have one element");
        assert!((data[0] - expected_sum).abs() < 1e-6, "Sum value mismatch");
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
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let axes = vec![0, 2]; // Axis 2 is invalid for rank 2
        let result = sum_op(&t, Some(&axes), false);
        // Check for the specific error (now ShapeMismatch)
        assert!(
            matches!(result, Err(NeuraRustError::ShapeMismatch { .. })),
            "Expected ShapeMismatch for invalid axis, got {:?}", result
        );
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

/// Public facing sum operation.
/// Handles optional axes argument.
/// If axes is None, sums over all axes.
pub(crate) fn sum_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let all_axes: Vec<usize>; // Variable pour stocker les axes si nécessaire
    
    let axes_slice: &[usize] = match axes {
        Some(ax) => ax, // Utilise les axes fournis
        None => {
            // Si None, génère les axes 0..rank
            let rank = tensor.shape().len(); // Utilise shape().len() pour obtenir le rang
            all_axes = (0..rank).collect(); // Stocke dans `all_axes`
            &all_axes // Passe une référence au vecteur détenu
        }
    };
    
    // Appelle sum_axes avec le slice d'axes déterminé
    sum_axes(tensor, axes_slice, keep_dims)
}
