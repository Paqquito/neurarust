use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;

/// Sélectionne des éléments le long d'une dimension selon un tenseur d'indices (I32/I64).
pub fn index_select_op(input: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor, NeuraRustError> {
    let input_guard = input.read_data();
    let indices_guard = indices.read_data();

    // Vérification device CPU
    if input_guard.device != crate::device::StorageDevice::CPU || indices_guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "index_select_op".to_string(),
            expected: crate::device::StorageDevice::CPU,
            actual: input_guard.device,
        });
    }
    // Vérification dtype indices
    if indices_guard.dtype != DType::I64 {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "index_select_op (indices)".to_string(),
            expected: DType::I64,
            actual: indices_guard.dtype,
        });
    }
    // Vérification dim
    if dim >= input_guard.shape.len() {
        return Err(NeuraRustError::IndexOutOfBounds {
            index: vec![dim],
            shape: input_guard.shape.clone(),
        });
    }
    let input_shape = &input_guard.shape;
    let indices_vec = indices_guard.buffer.try_get_cpu_i64()?;
    let n_indices = indices_vec.len();
    let mut output_shape = input_shape.clone();
    output_shape[dim] = n_indices;
    let rank = input_shape.len();
    let mut coord = vec![0; rank];
    let total = output_shape.iter().product();
    match input_guard.dtype {
        DType::F32 => {
            let input_data = input_guard.buffer.try_get_cpu_f32()?;
            let input_strides = &input_guard.strides;
            let mut output = Vec::with_capacity(input_guard.numel() / input_shape[dim] * n_indices);
            for out_idx in 0..total {
                let mut rem = out_idx;
                for d in (0..rank).rev() {
                    coord[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }
                let idx_in_indices = coord[dim];
                let idx_value = indices_vec[idx_in_indices];
                if idx_value < 0 || idx_value as usize >= input_shape[dim] {
                    return Err(NeuraRustError::IndexOutOfBounds {
                        index: vec![idx_value as usize],
                        shape: input_shape.clone(),
                    });
                }
                let mut input_coord = coord.clone();
                input_coord[dim] = idx_value as usize;
                let mut offset = input_guard.offset;
                for d in 0..rank {
                    offset += input_coord[d] * input_strides[d];
                }
                output.push(input_data[offset]);
            }
            drop(input_guard);
            drop(indices_guard);
            Tensor::new(output, output_shape)
        }
        DType::F64 => {
            let input_data = input_guard.buffer.try_get_cpu_f64()?;
            let input_strides = &input_guard.strides;
            let mut output = Vec::with_capacity(input_guard.numel() / input_shape[dim] * n_indices);
            for out_idx in 0..total {
                let mut rem = out_idx;
                for d in (0..rank).rev() {
                    coord[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }
                let idx_in_indices = coord[dim];
                let idx_value = indices_vec[idx_in_indices];
                if idx_value < 0 || idx_value as usize >= input_shape[dim] {
                    return Err(NeuraRustError::IndexOutOfBounds {
                        index: vec![idx_value as usize],
                        shape: input_shape.clone(),
                    });
                }
                let mut input_coord = coord.clone();
                input_coord[dim] = idx_value as usize;
                let mut offset = input_guard.offset;
                for d in 0..rank {
                    offset += input_coord[d] * input_strides[d];
                }
                output.push(input_data[offset]);
            }
            drop(input_guard);
            drop(indices_guard);
            Tensor::new_f64(output, output_shape)
        }
        DType::I32 => {
            let input_data = input_guard.buffer.try_get_cpu_i32()?;
            let input_strides = &input_guard.strides;
            let mut output = Vec::with_capacity(input_guard.numel() / input_shape[dim] * n_indices);
            for out_idx in 0..total {
                let mut rem = out_idx;
                for d in (0..rank).rev() {
                    coord[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }
                let idx_in_indices = coord[dim];
                let idx_value = indices_vec[idx_in_indices];
                if idx_value < 0 || idx_value as usize >= input_shape[dim] {
                    return Err(NeuraRustError::IndexOutOfBounds {
                        index: vec![idx_value as usize],
                        shape: input_shape.clone(),
                    });
                }
                let mut input_coord = coord.clone();
                input_coord[dim] = idx_value as usize;
                let mut offset = input_guard.offset;
                for d in 0..rank {
                    offset += input_coord[d] * input_strides[d];
                }
                output.push(input_data[offset]);
            }
            drop(input_guard);
            drop(indices_guard);
            Tensor::new_i32(output, output_shape)
        }
        DType::I64 => {
            let input_data = input_guard.buffer.try_get_cpu_i64()?;
            let input_strides = &input_guard.strides;
            let mut output = Vec::with_capacity(input_guard.numel() / input_shape[dim] * n_indices);
            for out_idx in 0..total {
                let mut rem = out_idx;
                for d in (0..rank).rev() {
                    coord[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }
                let idx_in_indices = coord[dim];
                let idx_value = indices_vec[idx_in_indices];
                if idx_value < 0 || idx_value as usize >= input_shape[dim] {
                    return Err(NeuraRustError::IndexOutOfBounds {
                        index: vec![idx_value as usize],
                        shape: input_shape.clone(),
                    });
                }
                let mut input_coord = coord.clone();
                input_coord[dim] = idx_value as usize;
                let mut offset = input_guard.offset;
                for d in 0..rank {
                    offset += input_coord[d] * input_strides[d];
                }
                output.push(input_data[offset]);
            }
            drop(input_guard);
            drop(indices_guard);
            Tensor::new_i64(output, output_shape)
        }
        DType::Bool => {
            let input_data = input_guard.buffer.try_get_cpu_bool()?;
            let input_strides = &input_guard.strides;
            let mut output = Vec::with_capacity(input_guard.numel() / input_shape[dim] * n_indices);
            for out_idx in 0..total {
                let mut rem = out_idx;
                for d in (0..rank).rev() {
                    coord[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }
                let idx_in_indices = coord[dim];
                let idx_value = indices_vec[idx_in_indices];
                if idx_value < 0 || idx_value as usize >= input_shape[dim] {
                    return Err(NeuraRustError::IndexOutOfBounds {
                        index: vec![idx_value as usize],
                        shape: input_shape.clone(),
                    });
                }
                let mut input_coord = coord.clone();
                input_coord[dim] = idx_value as usize;
                let mut offset = input_guard.offset;
                for d in 0..rank {
                    offset += input_coord[d] * input_strides[d];
                }
                output.push(input_data[offset]);
            }
            drop(input_guard);
            drop(indices_guard);
            Tensor::new_bool(output, output_shape)
        }
//        _ => Err(NeuraRustError::UnsupportedOperation("index_select_op: DType non supporté".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_index_select_f32_dim0() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let idx = Tensor::new_i64(vec![1, 0], vec![2]).unwrap();
        let out = index_select_op(&t, 0, &idx).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        assert_eq!(out.get_f32_data().unwrap(), vec![3.0, 4.0, 1.0, 2.0]);
    }
    #[test]
    fn test_index_select_f64_dim1() {
        let t = Tensor::new_f64(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let idx = Tensor::new_i64(vec![0, 1], vec![2]).unwrap();
        let out = index_select_op(&t, 1, &idx).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        assert_eq!(out.get_f64_data().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_index_select_i32_dim0() {
        let t = Tensor::new_i32(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let idx = Tensor::new_i64(vec![1, 1], vec![2]).unwrap();
        let out = index_select_op(&t, 0, &idx).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        assert_eq!(out.get_i32_data().unwrap(), vec![3, 4, 3, 4]);
    }
    #[test]
    fn test_index_select_i64_dim1() {
        let t = Tensor::new_i64(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let idx = Tensor::new_i64(vec![0, 0], vec![2]).unwrap();
        let out = index_select_op(&t, 1, &idx).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        assert_eq!(out.get_i64_data().unwrap(), vec![10, 10, 30, 30]);
    }
    #[test]
    fn test_index_select_bool_dim0() {
        let t = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let idx = Tensor::new_i64(vec![1, 0], vec![2]).unwrap();
        let out = index_select_op(&t, 0, &idx).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        assert_eq!(out.get_bool_data().unwrap(), vec![true, false, true, false]);
    }
} 