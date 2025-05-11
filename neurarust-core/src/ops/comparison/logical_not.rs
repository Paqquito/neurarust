use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::types::DType;

/// Effectue un NON logique élément par élément sur un tenseur booléen.
/// Le résultat est un tenseur booléen (true devient false et vice versa).
pub fn logical_not_op(a: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();

    if a_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "logical_not_op".to_string(),
            expected: StorageDevice::CPU,
            actual: a_guard.device,
        });
    }

    if a_guard.dtype != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::Bool,
            actual: a_guard.dtype,
            operation: "logical_not_op".to_string(),
        });
    }

    let output_shape = a_guard.shape.clone();
    let numel = output_shape.iter().product();

    let a_buffer = a_guard.buffer.try_get_cpu_bool()?;
    let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(
        a_buffer,
        &a_guard.shape,
        &a_guard.strides,
        a_guard.offset,
        &output_shape,
    )?;

    let result_data_vec: Vec<bool> = iter_a.map(|va| !va).collect();

    if result_data_vec.len() != numel {
        return Err(NeuraRustError::InternalError(format!(
            "logical_not_op: Output vec len {} mismatch with expected numel {}",
            result_data_vec.len(),
            numel
        )));
    }

    drop(a_guard);
    Tensor::new_bool(result_data_vec, output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_logical_not_basic() {
        let t = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let result = logical_not_op(&t).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, false, true]);
    }

    #[test]
    fn test_logical_not_non_bool() {
        let t = Tensor::new(vec![1.0f32, 2.0], vec![2]).unwrap();
        let result = logical_not_op(&t);
        assert!(result.is_err());
    }

    #[test]
    fn test_logical_not_empty() {
        let t = Tensor::new_bool(vec![], vec![0]).unwrap();
        let result = logical_not_op(&t).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![]);
    }
} 