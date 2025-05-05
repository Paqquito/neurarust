// neurarust-core/src/ops/math_elem/ln_test.rs

#[cfg(test)]
mod tests {
    use crate::ops::math_elem::ln::ln_op;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;
    use crate::error::NeuraRustError;
    use crate::utils::testing::check_tensor_near; // Utiliser pour comparer f32
    // Pas besoin de check_grad ici

    // SUPPRIMER les anciens helpers create_test_tensor*<f64>

    // Helper (non-generic) pour obtenir les données f32
    fn get_f32_data(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
        let guard = tensor.read_data();
        if guard.dtype != crate::types::DType::F32 || guard.device != crate::device::StorageDevice::CPU {
            return Err(NeuraRustError::UnsupportedOperation("Test helper requires F32 CPU tensor".to_string()));
        }
        match &*guard.buffer {
            crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F32(data_arc)) => Ok(data_arc.to_vec()),
            _ => Err(NeuraRustError::UnsupportedOperation("Buffer type not CpuF32".to_string())),
        }
    }

    #[test]
    fn test_ln_forward_basic() -> Result<(), NeuraRustError> {
        let a = crate::tensor::from_vec_f32(vec![1.0, std::f32::consts::E, 10.0], vec![3])?;
        let result = ln_op(&a)?;
        let expected_data = vec![0.0, 1.0, 10.0f32.ln()];
        assert_eq!(result.shape(), &[3]);
        let res_data = get_f32_data(&result)?;
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_ln_forward_non_positive() -> Result<(), NeuraRustError> {
        let a = crate::tensor::from_vec_f32(vec![1.0, 0.0], vec![2])?; // Includes zero
        let result_a = ln_op(&a)?;
        let data_a = get_f32_data(&result_a)?;
        assert!(data_a[0].is_finite()); // ln(1) is 0
        assert!(data_a[1].is_infinite(), "ln(0.0) should be infinite");
        assert!(data_a[1].is_sign_negative(), "ln(0.0) should be negative infinity");

        let b = crate::tensor::from_vec_f32(vec![-1.0], vec![1])?; // Negative input
        let result_b = ln_op(&b)?;
        let data_b = get_f32_data(&result_b)?;
        assert!(data_b[0].is_nan());    // ln(-1) should produce NaN
        Ok(())
    }

    #[test]
    fn test_ln_backward() -> Result<(), NeuraRustError> {
        let a = crate::tensor::from_vec_f32(vec![1.0, 2.0, 4.0], vec![3])?;
        a.set_requires_grad(true)?;

        let result = ln_op(&a)?;
        let output_grad = crate::tensor::from_vec_f32(vec![0.1, 0.2, 0.3], vec![3])?;
        result.backward(Some(output_grad))?;

        // grad_a = output_grad / a
        let expected_grad = vec![0.1 / 1.0, 0.2 / 2.0, 0.3 / 4.0]; // [0.1, 0.1, 0.075]
        check_tensor_near(&a.grad().unwrap(), &[3], &expected_grad, 1e-6);
        Ok(())
    }
} 