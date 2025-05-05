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

// Removed simple iterator imports for now, sum needs more complex logic
// use crate::tensor::iter_utils::{NdArraySimpleIter, NdArraySimpleIterF64};

// Need index_to_coord and calculate_strides for the new logic
use crate::tensor::utils::{index_to_coord, calculate_strides};

/// Backward operation context for sum reduction.
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
                operation: "sum_op_backward".to_string(),
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
                            operation: "sum_op_backward (reshape)".to_string(),
                        });
                    }
                    if grad_output_guard.shape[current_grad_dim] != target_shape[i] {
                        return Err(NeuraRustError::ShapeMismatch {
                            expected: format!("{:?}", target_shape),
                            actual: format!("{:?}", grad_output_guard.shape),
                            operation: "sum_op_backward (reshape)".to_string(),
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

/// Public facing sum operation.
/// Handles optional axes argument.
/// If axes is None, sums over all axes.
pub(crate) fn sum_op(
    t: &Tensor,
    axes: Option<&[usize]>,
    keep_dims: bool,
) -> Result<Tensor, NeuraRustError> {
    let t_guard = t.read_data();

    // --- Device Check ---
    if t_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(
            "sum_op currently only supports CPU tensors.".to_string(),
        ));
    }

    // --- Metadata Extraction --- 
    let input_shape = t_guard.shape.clone();
    let input_strides = t_guard.strides.clone(); 
    let input_offset = t_guard.offset;          
    let requires_grad = t_guard.requires_grad;
    let dtype = t_guard.dtype;
    let input_node_arc = if requires_grad { Some(Arc::clone(&t.data)) } else { None };

    let rank = input_shape.len();
    let axes_to_reduce = match axes {
        Some(ax) => { 
            let mut unique_axes: Vec<usize> = ax.to_vec();
            unique_axes.sort_unstable();
            unique_axes.dedup();
            for &axis in &unique_axes {
                if axis >= rank {
                    return Err(NeuraRustError::InvalidAxis { axis, rank });
                }
            }
            unique_axes
         }
        None => (0..rank).collect(),
    };

    // --- Calculate Output Shape --- 
    let mut output_shape_vec = Vec::with_capacity(rank);
    let mut final_output_shape = Vec::new(); // Shape used for tensor creation
    if keep_dims {
        output_shape_vec = input_shape.clone();
        for &axis in &axes_to_reduce {
            output_shape_vec[axis] = 1;
        }
        final_output_shape = output_shape_vec.clone(); // Use the shape with ones
    } else {
        for i in 0..rank {
            if !axes_to_reduce.contains(&i) {
                output_shape_vec.push(input_shape[i]); // Temporarily store non-reduced dims
                 final_output_shape.push(input_shape[i]);
            } else {
                 output_shape_vec.push(1); // Placeholder for reduced dims 
            }
        }
        // Handle scalar output (reducing all dimensions results in empty final_output_shape)
        if final_output_shape.is_empty() {
             // No need to push(1) here, tensor creation handles scalar from empty vec
        }
    }
    let output_numel: usize = final_output_shape.iter().product();

    // --- Perform Summation (New Logic) --- 
    let output_tensor = match dtype {
        DType::F32 => {
            let input_buffer = t_guard.buffer.try_get_cpu_f32()?;
            let mut output_data = vec![0.0f32; output_numel];
            let _output_strides = calculate_strides(&output_shape_vec); // Prefix with underscore

            let input_numel: usize = input_shape.iter().product();
            for linear_input_idx in 0..input_numel {
                // 1. Get logical coordinates in the input tensor
                let input_coords = index_to_coord(linear_input_idx, &input_shape);

                // 2. Calculate physical offset in the input buffer
                let mut physical_input_offset = input_offset;
                for d in 0..rank {
                    physical_input_offset += input_coords[d] * input_strides[d];
                }
                let input_val = input_buffer[physical_input_offset];

                // 3. Determine the corresponding logical coordinates in the *output* tensor
                let mut output_coords = Vec::with_capacity(rank);
                for d in 0..rank {
                    if axes_to_reduce.contains(&d) {
                       if keep_dims { output_coords.push(0); } // Index is always 0 for reduced dim if kept
                       // If not keep_dims, this dimension doesn't exist in output, do nothing
                    } else {
                        output_coords.push(input_coords[d]); // Keep original coordinate
                    }
                }
                // If not keep_dims, filter output_coords to remove reduced dimensions
                let final_output_coords: Vec<usize> = if keep_dims {
                     output_coords.clone() // Clone here to avoid moving
                } else {
                     input_coords.iter().enumerate()
                        .filter(|(i, _)| !axes_to_reduce.contains(i))
                        .map(|(_, &coord)| coord)
                        .collect()
                };

                // 4. Calculate linear index in the output buffer
                // Need strides for the *final* output shape
                 let final_output_strides = calculate_strides(&final_output_shape);
                 let mut linear_output_idx = 0;
                 if !final_output_shape.is_empty() { // Avoid calculating for scalar output
                    for d in 0..final_output_coords.len() {
                        linear_output_idx += final_output_coords[d] * final_output_strides[d];
                    }
                 }
                
                // Check bounds (should not happen with correct logic, but safeguard)
                 if linear_output_idx < output_numel {
                     output_data[linear_output_idx] += input_val;
                 } else if output_numel == 1 && linear_output_idx == 0 {
                      // Handle scalar case where index might be calculated slightly off
                      output_data[0] += input_val;
                 } else if output_numel > 0 {
                     // This indicates an error in index calculation
                     eprintln!("Sum Op Error: Output index {} out of bounds {}", linear_output_idx, output_numel);
                     eprintln!("Input Coords: {:?}, Output Coords: {:?}, Final Output Coords: {:?}, Final Shape: {:?}", input_coords, output_coords, final_output_coords, final_output_shape);
                     return Err(NeuraRustError::InternalError("Sum op output index calculation error".to_string()));
                 }
                 // If output_numel is 0, do nothing (should not happen if input_numel > 0)
            }
            
            drop(t_guard);
            Tensor::new(output_data, final_output_shape)?
        }
        DType::F64 => { // Similar logic for F64
             let input_buffer = t_guard.buffer.try_get_cpu_f64()?;
            let mut output_data = vec![0.0f64; output_numel];
            let _output_strides = calculate_strides(&output_shape_vec); // Prefix with underscore

            let input_numel: usize = input_shape.iter().product();
            for linear_input_idx in 0..input_numel {
                let input_coords = index_to_coord(linear_input_idx, &input_shape);
                let mut physical_input_offset = input_offset;
                for d in 0..rank {
                    physical_input_offset += input_coords[d] * input_strides[d];
                }
                let input_val = input_buffer[physical_input_offset];

                let mut output_coords = Vec::with_capacity(rank);
                 for d in 0..rank {
                    if axes_to_reduce.contains(&d) {
                       if keep_dims { output_coords.push(0); } 
                    } else {
                        output_coords.push(input_coords[d]);
                    }
                }
                 let final_output_coords: Vec<usize> = if keep_dims {
                     output_coords.clone() // Clone here to avoid moving
                } else {
                     input_coords.iter().enumerate()
                        .filter(|(i, _)| !axes_to_reduce.contains(i))
                        .map(|(_, &coord)| coord)
                        .collect()
                };

                 let final_output_strides = calculate_strides(&final_output_shape);
                 let mut linear_output_idx = 0;
                 if !final_output_shape.is_empty() { 
                     for d in 0..final_output_coords.len() {
                        linear_output_idx += final_output_coords[d] * final_output_strides[d];
                    }
                 }
                
                 if linear_output_idx < output_numel {
                     output_data[linear_output_idx] += input_val;
                 } else if output_numel == 1 && linear_output_idx == 0 {
                      output_data[0] += input_val;
                 } else if output_numel > 0 {
                     eprintln!("Sum Op Error F64: Output index {} out of bounds {}", linear_output_idx, output_numel);
                     eprintln!("Input Coords: {:?}, Output Coords: {:?}, Final Output Coords: {:?}, Final Shape: {:?}", input_coords, output_coords, final_output_coords, final_output_shape);
                      return Err(NeuraRustError::InternalError("Sum op output index calculation error F64".to_string()));
                 }
            }
            
            drop(t_guard);
            Tensor::new_f64(output_data, final_output_shape)?
        }
    };

    // --- Autograd Setup ---
    if requires_grad {
        if let Some(input_arc) = input_node_arc {
             // Use the correct struct SumAxesBackward and field name 'axes'
             let backward_context: Arc<dyn BackwardOp> = Arc::new(SumAxesBackward {
                    input_node: input_arc,
                    input_shape,
                    axes: axes_to_reduce, // Assign axes_to_reduce to the 'axes' field
                    keep_dims,
                });
            let mut output_guard = output_tensor.write_data();
            output_guard.requires_grad = true;
            output_guard.grad_fn = Some(backward_context);
        } else {
             return Err(NeuraRustError::InternalError("Sum op requires grad but Arc node unavailable".to_string()));
        }
    }

    Ok(output_tensor)
}

// Link the tests from the separate file
#[cfg(test)]
#[path = "sum_test.rs"]
mod tests;
