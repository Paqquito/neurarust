use crate::device::StorageDevice;
use crate::error::NeuraRustError;
// use crate::ops::traits::NeuraNumeric; // supprimé car non utilisé
// use crate::tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64}; // supprimé car non utilisé
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

/// Effectue une comparaison élément par élément (a >= b) entre deux tenseurs.
///
/// # Types supportés
/// - Tous les types numériques (`F32`, `F64`, `I32`, `I64`, `Bool`).
///
/// # Arguments
/// * `a` - Premier tenseur.
/// * `b` - Second tenseur.
///
/// # Retour
/// Un `Tensor` Bool de même forme (broadcastée) que les entrées.
///
/// # Erreurs
/// - `DeviceMismatch` si les tenseurs ne sont pas sur le même device.
/// - `UnsupportedOperation` si les dtypes ne correspondent pas.
/// - `InternalError` si la taille de la sortie ne correspond pas à l'attendu.
///
/// # Exemple
/// ```
/// use neurarust_core::{Tensor};
/// use neurarust_core::ops::comparison::ge_op;
/// let a = Tensor::new(vec![1.0, 5.0, 3.0], vec![3]).unwrap();
/// let b = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]).unwrap();
/// let ge = ge_op(&a, &b).unwrap();
/// assert_eq!(ge.get_bool_data().unwrap(), vec![true, true, false]);
/// ```
pub fn ge_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "ge_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }
    if a_guard.dtype != b_guard.dtype {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("ge_op requires same dtype, got {:?} and {:?}", a_guard.dtype, b_guard.dtype)
        ));
    }
    let dtype = a_guard.dtype;
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();
    let result_data_vec: Vec<bool> = match dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIter::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIter::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va >= vb).collect()
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va >= vb).collect()
        }
        DType::I32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_i32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_i32()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterI32::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterI32::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va >= vb).collect()
        }
        DType::I64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_i64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_i64()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterI64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterI64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va >= vb).collect()
        }
        DType::Bool => {
            let a_buffer = a_guard.buffer.try_get_cpu_bool()?;
            let b_buffer = b_guard.buffer.try_get_cpu_bool()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va >= vb).collect()
        }
    };
    if result_data_vec.len() != numel {
        return Err(NeuraRustError::InternalError(format!(
            "ge_op: Output vec len {} mismatch with expected numel {}",
            result_data_vec.len(), numel
        )));
    }
    drop(a_guard);
    drop(b_guard);
    Tensor::new_bool(result_data_vec, output_shape)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::ge_op;
    use crate::tensor::Tensor;
    #[test]
    fn test_ge_f32() {
        let t1 = Tensor::new(vec![1.0f32, 5.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 2.0, 4.0], vec![3]).unwrap();
        let result = ge_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, true, false]);
    }
    #[test]
    fn test_ge_f32_broadcast() {
        let t1 = Tensor::new(vec![3.0f32], vec![1]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 3.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let result = ge_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, true, false, true]);
    }
    #[test]
    fn test_ge_f64() {
        let t1 = Tensor::new_f64(vec![1.0, 5.0, 3.0], vec![3]).unwrap();
        let t2 = Tensor::new_f64(vec![1.0, 2.0, 4.0], vec![3]).unwrap();
        let result = ge_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, true, false]);
    }
    #[test]
    fn test_ge_i32() {
        let t1 = Tensor::new_i32(vec![1, 5, 3], vec![3]).unwrap();
        let t2 = Tensor::new_i32(vec![1, 2, 4], vec![3]).unwrap();
        let result = ge_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, true, false]);
    }
    #[test]
    fn test_ge_i64() {
        let t1 = Tensor::new_i64(vec![1, 5, 3], vec![3]).unwrap();
        let t2 = Tensor::new_i64(vec![1, 2, 4], vec![3]).unwrap();
        let result = ge_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![true, true, false]);
    }
    #[test]
    fn test_ge_bool() {
        let t1 = Tensor::new_bool(vec![false, true, true, false], vec![2, 2]).unwrap();
        let t2 = Tensor::new_bool(vec![true, true, false, false], vec![2, 2]).unwrap();
        let result = ge_op(&t1, &t2).unwrap();
        assert_eq!(result.get_bool_data().unwrap(), vec![false, true, true, true]);
    }
    #[test]
    fn test_ge_shape_mismatch() {
        let t1 = Tensor::new(vec![1.0f32, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let result = ge_op(&t1, &t2);
        assert!(result.is_err());
    }
} 