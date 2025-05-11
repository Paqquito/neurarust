use crate::{tensor::Tensor, error::NeuraRustError, types::DType};
use std::sync::Arc;

/// Remplit les éléments du tenseur self avec la valeur donnée là où le masque est vrai (in-place si possible).
///
/// # Arguments
/// * `self_tensor` - Le tenseur à modifier (in-place si possible).
/// * `mask` - Un tenseur booléen, broadcastable à la shape de self_tensor.
/// * `value` - La valeur scalaire à écrire (même DType que self_tensor).
///
/// # Retour
/// Ok(()) si succès, Err sinon.
pub fn masked_fill_op<S: Clone + PartialEq + 'static>(
    self_tensor: &mut Tensor,
    mask: &Tensor,
    value: S,
) -> Result<(), NeuraRustError> {
    use crate::tensor::utils::{broadcast_shapes, index_to_coord};

    let dtype = self_tensor.dtype();
    if mask.dtype() != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::Bool,
            actual: mask.dtype(),
            operation: "masked_fill_op".to_string(),
        });
    }
    if self_tensor.device() != mask.device() {
        return Err(NeuraRustError::DeviceMismatch {
            expected: self_tensor.device(),
            actual: mask.device(),
            operation: "masked_fill_op".to_string(),
        });
    }
    let shape = self_tensor.shape();
    let mask_shape = mask.shape();
    let out_shape = broadcast_shapes(&shape, &mask_shape)?;
    if out_shape != shape {
        return Err(NeuraRustError::ShapeMismatch {
            expected: format!("{:?}", shape),
            actual: format!("{:?}", out_shape),
            operation: "masked_fill_op".to_string(),
        });
    }
    let numel: usize = shape.iter().product();
    let shape_vec = shape.to_vec();
    let strides_vec;
    let offset_val;
    {
        let data = self_tensor.read_data();
        strides_vec = data.strides.clone();
        offset_val = data.offset;
    }
    let mask_shape_vec = mask.shape().to_vec();
    let mask_strides_vec;
    let mask_offset_val;
    {
        let data = mask.read_data();
        mask_strides_vec = data.strides.clone();
        mask_offset_val = data.offset;
    }
    match dtype {
        DType::F32 => {
            let mut data = self_tensor.write_data();
            let buffer = Arc::get_mut(&mut data.buffer).ok_or_else(|| NeuraRustError::BufferSharedError { operation: "masked_fill_op".to_string() })?;
            let buffer = buffer.try_get_cpu_f32_mut()?;
            let mask_guard = mask.read_data();
            let mask_buffer = mask_guard.buffer().try_get_cpu_bool()?;
            let val = *(&value as &dyn std::any::Any).downcast_ref::<f32>().unwrap();
            for logical_idx in 0..numel {
                let coords = index_to_coord(logical_idx, &shape_vec);
                let mut mask_coords = vec![0; mask_shape_vec.len()];
                let offset = shape_vec.len() - mask_shape_vec.len();
                for (i, mask_dim) in mask_shape_vec.iter().enumerate() {
                    let target_dim = i + offset;
                    if *mask_dim == 1 {
                        mask_coords[i] = 0;
                    } else {
                        mask_coords[i] = coords[target_dim];
                    }
                }
                let mut mask_phys_idx = mask_offset_val;
                for (d, &c) in mask_coords.iter().enumerate() {
                    mask_phys_idx += c * mask_strides_vec[d];
                }
                let mask_val = mask_buffer[mask_phys_idx];
                if mask_val {
                    let mut phys_idx = offset_val;
                    for (d, &c) in coords.iter().enumerate() {
                        phys_idx += c * strides_vec[d];
                    }
                    buffer[phys_idx] = val;
                }
            }
        }
        DType::F64 => {
            let mut data = self_tensor.write_data();
            let buffer = Arc::get_mut(&mut data.buffer).ok_or_else(|| NeuraRustError::BufferSharedError { operation: "masked_fill_op".to_string() })?;
            let buffer = buffer.try_get_cpu_f64_mut()?;
            let mask_guard = mask.read_data();
            let mask_buffer = mask_guard.buffer().try_get_cpu_bool()?;
            let val = *(&value as &dyn std::any::Any).downcast_ref::<f64>().unwrap();
            for logical_idx in 0..numel {
                let coords = index_to_coord(logical_idx, &shape_vec);
                let mut mask_coords = vec![0; mask_shape_vec.len()];
                let offset = shape_vec.len() - mask_shape_vec.len();
                for (i, mask_dim) in mask_shape_vec.iter().enumerate() {
                    let target_dim = i + offset;
                    if *mask_dim == 1 {
                        mask_coords[i] = 0;
                    } else {
                        mask_coords[i] = coords[target_dim];
                    }
                }
                let mut mask_phys_idx = mask_offset_val;
                for (d, &c) in mask_coords.iter().enumerate() {
                    mask_phys_idx += c * mask_strides_vec[d];
                }
                if mask_buffer[mask_phys_idx] {
                    let mut phys_idx = offset_val;
                    for (d, &c) in coords.iter().enumerate() {
                        phys_idx += c * strides_vec[d];
                    }
                    buffer[phys_idx] = val;
                }
            }
        }
        DType::I32 => {
            let mut data = self_tensor.write_data();
            let buffer = Arc::get_mut(&mut data.buffer).ok_or_else(|| NeuraRustError::BufferSharedError { operation: "masked_fill_op".to_string() })?;
            let buffer = buffer.try_get_cpu_i32_mut()?;
            let mask_guard = mask.read_data();
            let mask_buffer = mask_guard.buffer().try_get_cpu_bool()?;
            let val = *(&value as &dyn std::any::Any).downcast_ref::<i32>().unwrap();
            for logical_idx in 0..numel {
                let coords = index_to_coord(logical_idx, &shape_vec);
                let mut mask_coords = vec![0; mask_shape_vec.len()];
                let offset = shape_vec.len() - mask_shape_vec.len();
                for (i, mask_dim) in mask_shape_vec.iter().enumerate() {
                    let target_dim = i + offset;
                    if *mask_dim == 1 {
                        mask_coords[i] = 0;
                    } else {
                        mask_coords[i] = coords[target_dim];
                    }
                }
                let mut mask_phys_idx = mask_offset_val;
                for (d, &c) in mask_coords.iter().enumerate() {
                    mask_phys_idx += c * mask_strides_vec[d];
                }
                if mask_buffer[mask_phys_idx] {
                    let mut phys_idx = offset_val;
                    for (d, &c) in coords.iter().enumerate() {
                        phys_idx += c * strides_vec[d];
                    }
                    buffer[phys_idx] = val;
                }
            }
        }
        DType::I64 => {
            let mut data = self_tensor.write_data();
            let buffer = Arc::get_mut(&mut data.buffer).ok_or_else(|| NeuraRustError::BufferSharedError { operation: "masked_fill_op".to_string() })?;
            let buffer = buffer.try_get_cpu_i64_mut()?;
            let mask_guard = mask.read_data();
            let mask_buffer = mask_guard.buffer().try_get_cpu_bool()?;
            let val = *(&value as &dyn std::any::Any).downcast_ref::<i64>().unwrap();
            for logical_idx in 0..numel {
                let coords = index_to_coord(logical_idx, &shape_vec);
                let mut mask_coords = vec![0; mask_shape_vec.len()];
                let offset = shape_vec.len() - mask_shape_vec.len();
                for (i, mask_dim) in mask_shape_vec.iter().enumerate() {
                    let target_dim = i + offset;
                    if *mask_dim == 1 {
                        mask_coords[i] = 0;
                    } else {
                        mask_coords[i] = coords[target_dim];
                    }
                }
                let mut mask_phys_idx = mask_offset_val;
                for (d, &c) in mask_coords.iter().enumerate() {
                    mask_phys_idx += c * mask_strides_vec[d];
                }
                if mask_buffer[mask_phys_idx] {
                    let mut phys_idx = offset_val;
                    for (d, &c) in coords.iter().enumerate() {
                        phys_idx += c * strides_vec[d];
                    }
                    buffer[phys_idx] = val;
                }
            }
        }
        DType::Bool => {
            let mut data = self_tensor.write_data();
            let buffer = Arc::get_mut(&mut data.buffer).ok_or_else(|| NeuraRustError::BufferSharedError { operation: "masked_fill_op".to_string() })?;
            let buffer = buffer.try_get_cpu_bool_mut()?;
            let mask_guard = mask.read_data();
            let mask_buffer = mask_guard.buffer().try_get_cpu_bool()?;
            let val = *(&value as &dyn std::any::Any).downcast_ref::<bool>().unwrap();
            for logical_idx in 0..numel {
                let coords = index_to_coord(logical_idx, &shape_vec);
                let mut mask_coords = vec![0; mask_shape_vec.len()];
                let rank_diff = shape_vec.len() as isize - mask_shape_vec.len() as isize;
                for (i, mask_dim) in mask_shape_vec.iter().enumerate() {
                    let target_dim = (i as isize + rank_diff) as usize;
                    if *mask_dim == 1 {
                        mask_coords[i] = 0;
                    } else {
                        mask_coords[i] = coords[target_dim];
                    }
                }
                let mut mask_phys_idx = mask_offset_val;
                for (d, &c) in mask_coords.iter().enumerate() {
                    mask_phys_idx += c * mask_strides_vec[d];
                }
                if mask_buffer[mask_phys_idx] {
                    let mut phys_idx = offset_val;
                    for (d, &c) in coords.iter().enumerate() {
                        phys_idx += c * strides_vec[d];
                    }
                    buffer[phys_idx] = val;
                }
            }
        }
//        _ => return Err(NeuraRustError::UnsupportedOperation("masked_fill_op: DType non supporté".to_string())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_masked_fill_f32() {
        let mut t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mask = Tensor::new_bool(vec![true, false, false, true], vec![2, 2]).unwrap();
        masked_fill_op(&mut t, &mask, 9.0f32).unwrap();
        assert_eq!(t.get_f32_data().unwrap(), vec![9.0, 2.0, 3.0, 9.0]);
    }

    #[test]
    fn test_masked_fill_i32() {
        let mut t = Tensor::new_i32(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let mask = Tensor::new_bool(vec![false, true, true, false], vec![2, 2]).unwrap();
        masked_fill_op(&mut t, &mask, 7i32).unwrap();
        assert_eq!(t.get_i32_data().unwrap(), vec![1, 7, 7, 4]);
    }

    #[test]
    fn test_masked_fill_bool() {
        let mut t = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let mask = Tensor::new_bool(vec![false, true, true, false], vec![2, 2]).unwrap();
        masked_fill_op(&mut t, &mask, false).unwrap();
        assert_eq!(t.get_bool_data().unwrap(), vec![true, false, false, false]);
    }

    #[test]
    fn test_masked_fill_broadcast() {
        let mut t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mask = Tensor::new_bool(vec![true, false], vec![2, 1]).unwrap();
        masked_fill_op(&mut t, &mask, 0.0f32).unwrap();
        // Résultat standard numpy/pytorch : broadcast ligne
        assert_eq!(t.get_f32_data().unwrap(), vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_masked_fill_broadcast_col() {
        let mut t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mask = Tensor::new_bool(vec![true, false], vec![1, 2]).unwrap();
        masked_fill_op(&mut t, &mask, 0.0f32).unwrap();
        // Résultat attendu : broadcast colonne
        assert_eq!(t.get_f32_data().unwrap(), vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_masked_fill_shape_mismatch() {
        let mut t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mask = Tensor::new_bool(vec![true, false, true], vec![3, 1]).unwrap();
        let res = masked_fill_op(&mut t, &mask, 0.0f32);
        assert!(res.is_err());
    }
} 