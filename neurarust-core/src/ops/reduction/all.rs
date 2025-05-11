use crate::{tensor::Tensor, error::NeuraRustError, types::DType};
use crate::ops::reduction::utils::{process_reduction_axes, calculate_reduction_output_shape};

/// Effectue une réduction logique 'all' sur un tenseur booléen.
#[allow(dead_code)]
pub(crate) fn all_op(tensor: &Tensor, axes: Option<&[usize]>, keep_dims: bool) -> Result<Tensor, NeuraRustError> {
    let t_guard = tensor.read_data();
    if t_guard.dtype != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::Bool,
            actual: t_guard.dtype,
            operation: "all_op".to_string(),
        });
    }
    let input_shape = &t_guard.shape;
    let rank = input_shape.len();
    let axes_vec = process_reduction_axes(rank, axes)?;
    let output_shape = calculate_reduction_output_shape(input_shape, &axes_vec, keep_dims);
    let input_data = t_guard.buffer().try_get_cpu_bool()?;
    let numel = t_guard.numel();
    if numel == 0 {
        // Par convention, all([]) = true
        return Ok(Tensor::new_bool(vec![true; output_shape.iter().product()], output_shape)?);
    }
    // Si aucune réduction, on renvoie une copie
    if axes_vec.is_empty() {
        return Ok(tensor.clone());
    }
    // Réduction sur axes
    // Pour l'instant, on ne gère que le cas global (tous axes)
    if axes_vec.len() == rank {
        let all_true = input_data.iter().all(|&b| b);
        return Ok(Tensor::new_bool(vec![all_true; output_shape.iter().product()], output_shape)?);
    }
    // Réduction par axes (implémentation naïve, à optimiser si besoin)
    // On utilise un indexeur multi-dim pour accumuler
    let mut result = vec![true; output_shape.iter().product()];
    let mut idx = vec![0; rank];
    for i in 0..numel {
        // Calculer l'index de sortie
        let mut out_idx = 0;
        let mut stride = 1;
        for (d, _) in input_shape.iter().enumerate().rev() {
            if axes_vec.contains(&d) {
                // Cette dimension est réduite, donc pas dans l'output
                continue;
            }
            out_idx += idx[d] * stride;
            stride *= output_shape.get(d - axes_vec.iter().filter(|&&a| a < d).count()).unwrap_or(&1);
        }
        result[out_idx] &= input_data[i];
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
    Tensor::new_bool(result, output_shape)
} 