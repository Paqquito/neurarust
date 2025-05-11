use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;
use crate::device::StorageDevice;

pub fn cast_op(tensor: &Tensor, new_dtype: DType) -> Result<Tensor, NeuraRustError> {
    let tensor_guard = tensor.read_data();

    if tensor_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "cast_op".to_string(),
            expected: StorageDevice::CPU,
            actual: tensor_guard.device,
        });
    }

    if tensor_guard.dtype == new_dtype {
        return Ok(tensor.clone()); // No-op if dtypes are the same
    }

    let output_shape = tensor_guard.shape.clone();
    let numel = tensor_guard.numel();
    let offset = tensor_guard.offset; // Important for views

    // Ensure contiguous before casting if not already.
    // This simplifies buffer access. For non-contiguous, a new contiguous buffer
    // would need to be created first. For now, error if not contiguous.
    // TODO: Handle non-contiguous tensors in cast_op by making them contiguous first.
    if !tensor_guard.is_contiguous() {
        return Err(NeuraRustError::UnsupportedOperation(
            "cast_op currently requires contiguous tensors. Please call .contiguous() first.".to_string()
        ));
    }

    let output_tensor = match (tensor_guard.dtype, new_dtype) {
        (DType::F32, DType::F64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f32()?.as_slice();
            let output_data: Vec<f64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as f64).collect();
            drop(tensor_guard);
            Tensor::new_f64(output_data, output_shape)?
        }
        (DType::F64, DType::F32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f64()?.as_slice();
            let output_data: Vec<f32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as f32).collect();
            drop(tensor_guard);
            Tensor::new(output_data, output_shape)?
        }
        (DType::F32, DType::I32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f32()?.as_slice();
            let output_data: Vec<i32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as i32).collect();
            drop(tensor_guard);
            Tensor::new_i32(output_data, output_shape)?
        }
        (DType::F32, DType::I64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f32()?.as_slice();
            let output_data: Vec<i64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as i64).collect();
            drop(tensor_guard);
            Tensor::new_i64(output_data, output_shape)?
        }
        (DType::F32, DType::Bool) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f32()?.as_slice();
            let output_data: Vec<bool> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x != 0.0).collect();
            drop(tensor_guard);
            Tensor::new_bool(output_data, output_shape)?
        }
        (DType::F64, DType::I32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f64()?.as_slice();
            let output_data: Vec<i32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as i32).collect();
            drop(tensor_guard);
            Tensor::new_i32(output_data, output_shape)?
        }
        (DType::F64, DType::I64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f64()?.as_slice();
            let output_data: Vec<i64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as i64).collect();
            drop(tensor_guard);
            Tensor::new_i64(output_data, output_shape)?
        }
        (DType::F64, DType::Bool) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f64()?.as_slice();
            let output_data: Vec<bool> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x != 0.0).collect();
            drop(tensor_guard);
            Tensor::new_bool(output_data, output_shape)?
        }
        (DType::I32, DType::F32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i32()?.as_slice();
            let output_data: Vec<f32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as f32).collect();
            drop(tensor_guard);
            Tensor::new(output_data, output_shape)?
        }
        (DType::I32, DType::F64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i32()?.as_slice();
            let output_data: Vec<f64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as f64).collect();
            drop(tensor_guard);
            Tensor::new_f64(output_data, output_shape)?
        }
        (DType::I32, DType::I64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i32()?.as_slice();
            let output_data: Vec<i64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as i64).collect();
            drop(tensor_guard);
            Tensor::new_i64(output_data, output_shape)?
        }
        (DType::I32, DType::Bool) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i32()?.as_slice();
            let output_data: Vec<bool> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x != 0).collect();
            drop(tensor_guard);
            Tensor::new_bool(output_data, output_shape)?
        }
        (DType::I64, DType::F32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i64()?.as_slice();
            let output_data: Vec<f32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as f32).collect();
            drop(tensor_guard);
            Tensor::new(output_data, output_shape)?
        }
        (DType::I64, DType::F64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i64()?.as_slice();
            let output_data: Vec<f64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as f64).collect();
            drop(tensor_guard);
            Tensor::new_f64(output_data, output_shape)?
        }
        (DType::I64, DType::I32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i64()?.as_slice();
            let output_data: Vec<i32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x as i32).collect();
            drop(tensor_guard);
            Tensor::new_i32(output_data, output_shape)?
        }
        (DType::I64, DType::Bool) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_i64()?.as_slice();
            let output_data: Vec<bool> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| x != 0).collect();
            drop(tensor_guard);
            Tensor::new_bool(output_data, output_shape)?
        }
        (DType::Bool, DType::F32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_bool()?.as_slice();
            let output_data: Vec<f32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
            drop(tensor_guard);
            Tensor::new(output_data, output_shape)?
        }
        (DType::Bool, DType::F64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_bool()?.as_slice();
            let output_data: Vec<f64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
            drop(tensor_guard);
            Tensor::new_f64(output_data, output_shape)?
        }
        (DType::Bool, DType::I32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_bool()?.as_slice();
            let output_data: Vec<i32> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| if x { 1 } else { 0 }).collect();
            drop(tensor_guard);
            Tensor::new_i32(output_data, output_shape)?
        }
        (DType::Bool, DType::I64) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_bool()?.as_slice();
            let output_data: Vec<i64> = input_data_slice[offset..offset + numel]
                .iter().map(|&x| if x { 1 } else { 0 }).collect();
            drop(tensor_guard);
            Tensor::new_i64(output_data, output_shape)?
        }
        (DType::I32, DType::I32) | (DType::I64, DType::I64) | (DType::F32, DType::F32) | (DType::F64, DType::F64) | (DType::Bool, DType::Bool) => {
            drop(tensor_guard);
            tensor.clone()
        }
//        _ => {
//            return Err(NeuraRustError::UnsupportedOperation(format!(
//                "cast_op from {:?} to {:?} is not supported (cas inattendu).",
//                tensor_guard.dtype, new_dtype
//            )));
//        }
    };
    
    // Cast operation usually doesn't propagate gradients in a typical way,
    // or the gradient is 1. For simplicity, new tensor won't require grad by default.
    // If needed, the user can set it.
    // output_tensor.write_data().requires_grad = tensor_guard.requires_grad; // Optionnel

    Ok(output_tensor)
}

#[cfg(test)]
mod cast_test {
    include!("cast_test.rs");
}