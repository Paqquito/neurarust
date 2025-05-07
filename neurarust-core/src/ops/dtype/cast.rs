use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;
use crate::device::StorageDevice;
use crate::ops::view::SliceArg;

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
                .iter()
                .map(|&x| x as f64)
                .collect();
            drop(tensor_guard);
            Tensor::new_f64(output_data, output_shape)?
        }
        (DType::F64, DType::F32) => {
            let input_data_slice = tensor_guard.buffer().try_get_cpu_f64()?.as_slice();
            let output_data: Vec<f32> = input_data_slice[offset..offset + numel]
                .iter()
                .map(|&x| x as f32)
                .collect();
            drop(tensor_guard);
            Tensor::new(output_data, output_shape)?
        }
        _ => {
            return Err(NeuraRustError::UnsupportedOperation(format!(
                "cast_op from {:?} to {:?} is not supported yet.",
                tensor_guard.dtype, new_dtype
            )));
        }
    };
    
    // Cast operation usually doesn't propagate gradients in a typical way,
    // or the gradient is 1. For simplicity, new tensor won't require grad by default.
    // If needed, the user can set it.
    // output_tensor.write_data().requires_grad = tensor_guard.requires_grad; // Optionnel

    Ok(output_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::types::DType;

    #[test]
    fn test_cast_f32_to_f64() -> Result<(), NeuraRustError> {
        let t_f32 = Tensor::new(vec![3.14f32, 0.0, -2.71], vec![3])?;
        let t_f64 = cast_op(&t_f32, DType::F64)?;

        assert_eq!(t_f64.dtype(), DType::F64);
        let data_f64 = t_f64.get_f64_data()?;
        assert_eq!(data_f64.len(), 3);
        crate::assert_relative_eq!(data_f64[0], 3.14f32 as f64, epsilon = 1e-7);
        crate::assert_relative_eq!(data_f64[1], 0.0f32 as f64, epsilon = 1e-7);
        crate::assert_relative_eq!(data_f64[2], -2.71f32 as f64, epsilon = 1e-7);
        Ok(())
    }

    #[test]
    fn test_cast_f64_to_f32() -> Result<(), NeuraRustError> {
        let t_f64 = Tensor::new_f64(vec![1.23456789f64, -9.87654321f64], vec![2])?;
        let t_f32 = cast_op(&t_f64, DType::F32)?;

        assert_eq!(t_f32.dtype(), DType::F32);
        let data_f32 = t_f32.get_f32_data()?;
        assert_eq!(data_f32.len(), 2);
        crate::assert_relative_eq!(data_f32[0] as f64, 1.23456789f64, epsilon = 1e-6);
        crate::assert_relative_eq!(data_f32[1] as f64, -9.87654321f64, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_cast_no_op() -> Result<(), NeuraRustError> {
        let t_f32 = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3])?;
        let t_f32_casted = cast_op(&t_f32, DType::F32)?;
        assert_eq!(t_f32_casted.dtype(), DType::F32);
        assert_eq!(t_f32_casted.get_f32_data()?, vec![1.0, 2.0, 3.0]);

        let t_f64 = Tensor::new_f64(vec![1.0, 2.0, 3.0], vec![3])?;
        let t_f64_casted = cast_op(&t_f64, DType::F64)?;
        assert_eq!(t_f64_casted.dtype(), DType::F64);
        assert_eq!(t_f64_casted.get_f64_data()?, vec![1.0, 2.0, 3.0]);
        assert_eq!(t_f64_casted.get_f32_data().is_err(), true);
        Ok(())
    }

    #[test]
    fn test_cast_unsupported_types() -> Result<(), NeuraRustError> {
        let t_f32 = Tensor::new(vec![1.0f32], vec![1])?;
        let res_f32_f32 = cast_op(&t_f32, DType::F32)?;
        assert_eq!(res_f32_f32.dtype(), DType::F32);

        let t_f64 = Tensor::new_f64(vec![1.0f64], vec![1])?;
        let res_f64_f64 = cast_op(&t_f64, DType::F64)?;
        assert_eq!(res_f64_f64.dtype(), DType::F64);
        
        Ok(())
    }
    
    #[test]
    fn test_cast_non_contiguous_error() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let v = t.slice(&[SliceArg::Slice(0, 2, 1), SliceArg::Slice(0, 3, 2)])?;
        assert!(!v.is_contiguous(), "Slice should be non-contiguous for this test");

        let result = cast_op(&v, DType::F64);
        assert!(result.is_err());
        if let Err(NeuraRustError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("cast_op currently requires contiguous tensors."));
        } else {
            panic!("Expected UnsupportedOperation error for non-contiguous tensor");
        }
        Ok(())
    }
} 