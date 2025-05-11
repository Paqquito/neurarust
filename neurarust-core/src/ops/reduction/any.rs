use crate::{tensor::Tensor, error::NeuraRustError, types::DType};
use crate::ops::reduction::utils::{process_reduction_axes, calculate_reduction_output_shape};
use crate::tensor::create::full_bool;

/// Effectue une réduction logique 'any' sur un tenseur booléen.
pub(crate) fn any_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let t_guard = tensor.read_data();
    if t_guard.dtype != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::Bool,
            actual: t_guard.dtype,
            operation: "any_op".to_string(),
        });
    }
    let input_shape = &t_guard.shape;
    let rank = input_shape.len();
    let axes_vec = process_reduction_axes(rank, axes)?;
    let output_shape = calculate_reduction_output_shape(input_shape, &axes_vec, keep_dims);
    let input_data = t_guard.buffer().try_get_cpu_bool()?;
    let numel = t_guard.numel();
    if numel == 0 {
        // Par convention, any([]) = false
        return Ok(full_bool(&output_shape, false)?);
    }
    // Si aucune réduction, on renvoie une copie
    if axes_vec.is_empty() {
        return Ok(tensor.clone());
    }
    // Réduction sur axes
    if axes_vec.len() == rank {
        let any_true = input_data.iter().any(|&b| b);
        return Ok(full_bool(&output_shape, any_true)?);
    }
    // Réduction par axes (implémentation naïve)
    let mut result = vec![false; output_shape.iter().product()];
    let mut idx = vec![0; rank];
    for i in 0..numel {
        // Calculer l'index de sortie
        let mut out_idx = 0;
        let mut stride = 1;
        for (d, &dim) in input_shape.iter().enumerate().rev() {
            if axes_vec.contains(&d) {
                continue;
            }
            out_idx += idx[d] * stride;
            stride *= output_shape.get(d - axes_vec.iter().filter(|&&a| a < d).count()).unwrap_or(&1);
        }
        result[out_idx] |= input_data[i];
        // Incrémentation de l'index multi-dim
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < input_shape[d] {
                break;
            } else {
                idx[d] = 0;
            }
        }
    }
    Ok(Tensor::from_vec_bool(result, output_shape))
} 