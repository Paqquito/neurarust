use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

/// Effectue un ET logique élément par élément entre deux tenseurs booléens.
/// Le résultat est un tenseur booléen (true si les deux entrées sont true, false sinon).
pub fn logical_and_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "logical_and_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }

    if a_guard.dtype != DType::Bool || b_guard.dtype != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::Bool,
            actual: if a_guard.dtype != DType::Bool { a_guard.dtype } else { b_guard.dtype },
            operation: "logical_and_op".to_string(),
        });
    }

    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    let a_buffer = a_guard.buffer.try_get_cpu_bool()?;
    let b_buffer = b_guard.buffer.try_get_cpu_bool()?;
    let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(
        a_buffer,
        &a_guard.shape,
        &a_guard.strides,
        a_guard.offset,
        &output_shape,
    )?;
    let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(
        b_buffer,
        &b_guard.shape,
        &b_guard.strides,
        b_guard.offset,
        &output_shape,
    )?;

    let result_data_vec: Vec<bool> = iter_a.zip(iter_b).map(|(va, vb)| va && vb).collect();

    if result_data_vec.len() != numel {
        return Err(NeuraRustError::InternalError(format!(
            "logical_and_op: Output vec len {} mismatch with expected numel {}",
            result_data_vec.len(),
            numel
        )));
    }

    drop(a_guard);
    drop(b_guard);
    Tensor::new_bool(result_data_vec, output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_logical_and_basic() {
        let t1 = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let t2 = Tensor::new_bool(vec![true, true, false, false], vec![2, 2]).unwrap();
        let result = logical_and_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, false, false]);
    }

    #[test]
    fn test_logical_and_broadcast() {
        let t1 = Tensor::new_bool(vec![true, false], vec![2, 1]).unwrap();
        let t2 = Tensor::new_bool(vec![true, false, true, false], vec![2, 2]).unwrap();
        let result = logical_and_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, false, false, false]);
    }

    #[test]
    fn test_logical_and_non_bool() {
        let t1 = Tensor::new(vec![1.0f32, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new_bool(vec![true, false], vec![2]).unwrap();
        let result = logical_and_op(&t1, &t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_logical_and_shape_mismatch() {
        let t1 = Tensor::new_bool(vec![true, false], vec![2]).unwrap();
        let t2 = Tensor::new_bool(vec![true, false, true], vec![3]).unwrap();
        let result = logical_and_op(&t1, &t2);
        assert!(result.is_err());
    }
} 