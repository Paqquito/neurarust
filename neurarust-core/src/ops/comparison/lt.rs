use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

/// Effectue une comparaison élément par élément (a < b) entre deux tenseurs.
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
/// - `DataTypeMismatch` si les dtypes ne correspondent pas.
/// - `InternalError` si la taille de la sortie ne correspond pas à l'attendu.
///
/// # Exemple
/// ```
/// use neurarust_core::{Tensor};
/// use neurarust_core::ops::comparison::lt_op;
/// let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
/// let b = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]).unwrap();
/// let lt = lt_op(&a, &b).unwrap();
/// assert_eq!(lt.get_bool_data().unwrap(), vec![true, false, false]);
/// ```
pub fn lt_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "lt_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }
    if a_guard.dtype != b_guard.dtype {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "lt_op".to_string(),
            expected: a_guard.dtype,
            actual: b_guard.dtype,
        });
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
            iter_a.zip(iter_b).map(|(va, vb)| va < vb).collect()
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va < vb).collect()
        }
        DType::I32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_i32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_i32()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterI32::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterI32::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va < vb).collect()
        }
        DType::I64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_i64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_i64()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterI64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterI64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| va < vb).collect()
        }
        DType::Bool => {
            let a_buffer = a_guard.buffer.try_get_cpu_bool()?;
            let b_buffer = b_guard.buffer.try_get_cpu_bool()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b).map(|(va, vb)| !va && vb).collect()
        }
        // _ => return Err(NeuraRustError::DataTypeMismatch {
        //     operation: "lt_op".to_string(),
        //     expected: dtype,
        //     actual: dtype,
        // }),
    };
    if result_data_vec.len() != numel {
        return Err(NeuraRustError::InternalError(format!(
            "lt_op: Output vec len {} mismatch with expected numel {}",
            result_data_vec.len(), numel
        )));
    }
    drop(a_guard);
    drop(b_guard);
    Tensor::new_bool(result_data_vec, output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::create::from_vec_f32;
    #[test]
    fn test_lt_op_basic() {
        let a = from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = from_vec_f32(vec![2.0, 2.0, 2.0], vec![3]).unwrap();
        let out = lt_op(&a, &b).unwrap();
        assert_eq!(out.dtype(), DType::Bool);
        assert_eq!(out.get_bool_data().unwrap(), vec![true, false, false]);
    }
} 