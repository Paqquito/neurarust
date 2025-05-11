use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::Tensor;
use crate::types::DType;

/// Effectue une comparaison élément par élément (a != b) entre deux tenseurs.
/// Le résultat est un tensor Bool (true si différent, false sinon).
pub fn ne_op(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuraRustError> {
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // Vérification du device (CPU uniquement)
    if a_guard.device != StorageDevice::CPU || b_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "ne_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if a_guard.device != StorageDevice::CPU { a_guard.device } else { b_guard.device },
        });
    }

    // Vérification du dtype (doit être identique)
    if a_guard.dtype != b_guard.dtype {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("ne_op requires same dtype, got {:?} and {:?}", a_guard.dtype, b_guard.dtype)
        ));
    }
    let dtype = a_guard.dtype;

    // Broadcasting
    let output_shape = broadcast_shapes(&a_guard.shape, &b_guard.shape)?;
    let numel = output_shape.iter().product();

    // Dispatch selon le dtype
    let result_data_vec: Vec<bool> = match dtype {
        DType::F32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f32()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIter::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIter::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b)
                .map(|(va, vb)| (va - vb).abs() > 1e-6)
                .collect()
        }
        DType::F64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_f64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_f64()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterF64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterF64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b)
                .map(|(va, vb)| va != vb)
                .collect()
        }
        DType::I32 => {
            let a_buffer = a_guard.buffer.try_get_cpu_i32()?;
            let b_buffer = b_guard.buffer.try_get_cpu_i32()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterI32::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterI32::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b)
                .map(|(va, vb)| va != vb)
                .collect()
        }
        DType::I64 => {
            let a_buffer = a_guard.buffer.try_get_cpu_i64()?;
            let b_buffer = b_guard.buffer.try_get_cpu_i64()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterI64::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterI64::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b)
                .map(|(va, vb)| va != vb)
                .collect()
        }
        DType::Bool => {
            let a_buffer = a_guard.buffer.try_get_cpu_bool()?;
            let b_buffer = b_guard.buffer.try_get_cpu_bool()?;
            let iter_a = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(a_buffer, &a_guard.shape, &a_guard.strides, a_guard.offset, &output_shape)?;
            let iter_b = crate::tensor::iter_utils::NdArrayBroadcastingIterBool::new(b_buffer, &b_guard.shape, &b_guard.strides, b_guard.offset, &output_shape)?;
            iter_a.zip(iter_b)
                .map(|(va, vb)| va != vb)
                .collect()
        }
    };

    if result_data_vec.len() != numel {
         return Err(NeuraRustError::InternalError(format!(
            "ne_op: Output vec len {} mismatch with expected numel {}",
             result_data_vec.len(), numel
        )));
    }
    drop(a_guard);
    drop(b_guard);
    Tensor::new_bool(result_data_vec, output_shape)
}

// --- Tests ---
#[cfg(test)]
#[path = "ne_test.rs"]
mod tests; // Fichier de tests associé 