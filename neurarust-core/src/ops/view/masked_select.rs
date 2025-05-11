use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;

/// Sélectionne les éléments du tenseur d'entrée pour lesquels le masque est vrai.
pub fn masked_select_op(input: &Tensor, mask: &Tensor) -> Result<Tensor, NeuraRustError> {
    let input_guard = input.read_data();
    let mask_guard = mask.read_data();

    // Vérification device CPU
    if input_guard.device != crate::device::StorageDevice::CPU || mask_guard.device != crate::device::StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "masked_select_op".to_string(),
            expected: crate::device::StorageDevice::CPU,
            actual: input_guard.device,
        });
    }
    if mask_guard.dtype != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "masked_select_op (mask)".to_string(),
            expected: DType::Bool,
            actual: mask_guard.dtype,
        });
    }
    if input_guard.shape != mask_guard.shape {
        return Err(NeuraRustError::ShapeMismatch {
            operation: "masked_select_op".to_string(),
            expected: format!("shape {:?}", input_guard.shape),
            actual: format!("shape {:?}", mask_guard.shape),
        });
    }
    let numel = input_guard.numel();
    match input_guard.dtype {
        DType::F32 => {
            let input_data = input_guard.buffer.try_get_cpu_f32()?;
            let mask_data = mask_guard.buffer.try_get_cpu_bool()?;
            let mut output = Vec::new();
            for i in 0..numel {
                if mask_data[i] {
                    output.push(input_data[i]);
                }
            }
            let out_len = output.len();
            drop(input_guard);
            drop(mask_guard);
            Tensor::new(output, vec![out_len])
        }
        DType::F64 => {
            let input_data = input_guard.buffer.try_get_cpu_f64()?;
            let mask_data = mask_guard.buffer.try_get_cpu_bool()?;
            let mut output = Vec::new();
            for i in 0..numel {
                if mask_data[i] {
                    output.push(input_data[i]);
                }
            }
            let out_len = output.len();
            drop(input_guard);
            drop(mask_guard);
            Tensor::new_f64(output, vec![out_len])
        }
        DType::I32 => {
            let input_data = input_guard.buffer.try_get_cpu_i32()?;
            let mask_data = mask_guard.buffer.try_get_cpu_bool()?;
            let mut output = Vec::new();
            for i in 0..numel {
                if mask_data[i] {
                    output.push(input_data[i]);
                }
            }
            let out_len = output.len();
            drop(input_guard);
            drop(mask_guard);
            Tensor::new_i32(output, vec![out_len])
        }
        DType::I64 => {
            let input_data = input_guard.buffer.try_get_cpu_i64()?;
            let mask_data = mask_guard.buffer.try_get_cpu_bool()?;
            let mut output = Vec::new();
            for i in 0..numel {
                if mask_data[i] {
                    output.push(input_data[i]);
                }
            }
            let out_len = output.len();
            drop(input_guard);
            drop(mask_guard);
            Tensor::new_i64(output, vec![out_len])
        }
        DType::Bool => {
            let input_data = input_guard.buffer.try_get_cpu_bool()?;
            let mask_data = mask_guard.buffer.try_get_cpu_bool()?;
            let mut output = Vec::new();
            for i in 0..numel {
                if mask_data[i] {
                    output.push(input_data[i]);
                }
            }
            let out_len = output.len();
            drop(input_guard);
            drop(mask_guard);
            Tensor::new_bool(output, vec![out_len])
        }
//        _ => Err(NeuraRustError::UnsupportedOperation("masked_select_op: DType non supporté".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_masked_select_f32() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mask = Tensor::new_bool(vec![true, false, true, false], vec![4]).unwrap();
        let out = masked_select_op(&t, &mask).unwrap();
        assert_eq!(out.shape(), vec![2]);
        assert_eq!(out.get_f32_data().unwrap(), vec![1.0, 3.0]);
    }
    #[test]
    fn test_masked_select_f64() {
        let t = Tensor::new_f64(vec![1.0f64, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mask = Tensor::new_bool(vec![false, true, true, false], vec![4]).unwrap();
        let out = masked_select_op(&t, &mask).unwrap();
        assert_eq!(out.shape(), vec![2]);
        assert_eq!(out.get_f64_data().unwrap(), vec![2.0, 3.0]);
    }
    #[test]
    fn test_masked_select_i32() {
        let t = Tensor::new_i32(vec![1, 2, 3, 4], vec![4]).unwrap();
        let mask = Tensor::new_bool(vec![true, true, false, false], vec![4]).unwrap();
        let out = masked_select_op(&t, &mask).unwrap();
        assert_eq!(out.shape(), vec![2]);
        assert_eq!(out.get_i32_data().unwrap(), vec![1, 2]);
    }
    #[test]
    fn test_masked_select_i64() {
        let t = Tensor::new_i64(vec![10, 20, 30, 40], vec![4]).unwrap();
        let mask = Tensor::new_bool(vec![false, false, true, true], vec![4]).unwrap();
        let out = masked_select_op(&t, &mask).unwrap();
        assert_eq!(out.shape(), vec![2]);
        assert_eq!(out.get_i64_data().unwrap(), vec![30, 40]);
    }
    #[test]
    fn test_masked_select_bool() {
        let t = Tensor::new_bool(vec![true, false, true, false], vec![4]).unwrap();
        let mask = Tensor::new_bool(vec![true, false, false, true], vec![4]).unwrap();
        let out = masked_select_op(&t, &mask).unwrap();
        assert_eq!(out.shape(), vec![2]);
        assert_eq!(out.get_bool_data().unwrap(), vec![true, false]);
    }
} 