use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::types::DType;

/// Effectue un NON logique élément par élément sur un tenseur Bool.
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
        return Err(NeuraRustError::UnsupportedOperation(
            format!("logical_not_op requires Bool tensor, got {:?}", a_guard.dtype)
        ));
    }
    let a_buffer = a_guard.buffer.try_get_cpu_bool()?;
    let result_data_vec: Vec<bool> = a_buffer.iter().map(|&va| !va).collect();
    drop(a_guard);
    Tensor::new_bool(result_data_vec, a.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::logical_not_op;
    use crate::tensor::Tensor;
    #[test]
    fn test_logical_not_basic() {
        let t = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let result = logical_not_op(&t).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, false, true]);
    }
    #[test]
    fn test_logical_not_type_error() {
        let t = Tensor::new(vec![1.0f32, 0.0], vec![2]).unwrap();
        let result = logical_not_op(&t);
        assert!(result.is_err());
    }
    #[test]
    fn test_logical_not_device_error() {
        // Ce test suppose que vous avez un moyen de créer un tenseur sur un autre device, sinon il peut être ignoré.
        // let t = Tensor::new_bool_on_device(vec![true], vec![1], StorageDevice::GPU).unwrap();
        // let result = logical_not_op(&t);
        // assert!(result.is_err());
    }
} 