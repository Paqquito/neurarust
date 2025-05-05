#[cfg(test)]
mod tests {
    use crate::ops::arithmetic::pow_op;
    use crate::error::NeuraRustError;
    use crate::tensor::{Tensor, create};
    use crate::utils::testing::check_tensor_near;
    use approx::assert_relative_eq;
    use crate::autograd::grad_check::{check_grad, GradCheckError};

    // Helper (non-generic)
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
    fn test_pow_forward() -> Result<(), NeuraRustError> {
        let base = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 3.0, 2.0], vec![3]).unwrap();
        let result = pow_op(&base, &exponent)?;
        let expected_data = vec![1.0, 8.0, 9.0]; // 1^2, 2^3, 3^2
        assert_eq!(result.shape(), &[3]);
        let res_data = get_f32_data(&result)?;
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_pow_forward_broadcast() -> Result<(), NeuraRustError> {
        let base = crate::tensor::from_vec_f32(vec![1.0, 2.0], vec![1, 2]).unwrap(); // [[1.0, 2.0]]
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 3.0], vec![2, 1]).unwrap(); // [[2.0], [3.0]]
        let result = pow_op(&base, &exponent)?;
        let expected_data = vec![1.0, 8.0, 9.0]; // [1^2, 2^3, 3^2]
        assert_eq!(result.shape(), &[2, 2]);
        let res_data = get_f32_data(&result)?;
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    // --- Autograd Tests ---

    #[test]
    fn test_pow_backward_simple() -> Result<(), GradCheckError> {
        let base = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 3.0, 2.0], vec![3]).unwrap();
        base.set_requires_grad(true)?;
        exponent.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = Tensor::from_vec_f32(vec![1.0, 1.0, 1.0], vec![3])?;
        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        check_grad(func, &[base, exponent], &output_grad, epsilon, abs_tol, rel_tol)
    }

    #[test]
    fn test_pow_backward_only_base_grad() -> Result<(), GradCheckError> {
        let base = crate::tensor::from_vec_f32(vec![2.0, 3.0], vec![2]).unwrap();
        base.set_requires_grad(true)?;
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 1.0], vec![2]).unwrap();

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = Tensor::from_vec_f32(vec![1.0, 1.0], vec![2])?;
        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        check_grad(func, &[base, exponent], &output_grad, epsilon, abs_tol, rel_tol)
    }

    #[test]
    #[ignore = "Gradient for exponent is currently not implemented/supported in pow_op backward"]
    fn test_pow_backward_only_exponent_grad() -> Result<(), GradCheckError> {
        let base = crate::tensor::from_vec_f32(vec![2.0, 3.0], vec![2]).unwrap();
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 1.0], vec![2]).unwrap();
        exponent.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = Tensor::from_vec_f32(vec![1.0, 1.0], vec![2])?;
        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        check_grad(func, &[base, exponent], &output_grad, epsilon, abs_tol, rel_tol)
    }

    #[test]
    fn test_pow_backward_broadcast_base() -> Result<(), GradCheckError> {
        let base = crate::tensor::from_vec_f32(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 1.0], vec![2, 1]).unwrap();
        base.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = Tensor::from_vec_f32(vec![0.1, 0.2], vec![2, 2])?;
        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        check_grad(func, &[base, exponent], &output_grad, epsilon, abs_tol, rel_tol)
    }
    
    #[test]
    #[ignore = "Gradient for exponent is currently not implemented/supported in pow_op backward"]
    fn test_pow_backward_broadcast_exponent() -> Result<(), GradCheckError> {
        let base = crate::tensor::from_vec_f32(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let exponent = crate::tensor::from_vec_f32(vec![2.0, 1.0], vec![2, 1]).unwrap();
        exponent.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = Tensor::from_vec_f32(vec![0.1, 0.2], vec![2, 2])?;
        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        check_grad(func, &[base, exponent], &output_grad, epsilon, abs_tol, rel_tol)
    }
} 