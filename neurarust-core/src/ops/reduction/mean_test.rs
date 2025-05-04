#[cfg(test)]
mod tests {
    use crate::Tensor;
    use approx::assert_relative_eq;
    
    use crate::ops::reduction::mean::mean_axes;
    use crate::error::NeuraRustError;
    use crate::utils::testing::check_tensor_near;

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
    fn test_mean_all() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let all_axes: Vec<usize> = (0..t.shape().len()).collect();
        let result = mean_axes(&t, &all_axes, false)?;
        assert_eq!(result.shape(), &[] as &[usize], "Result shape should be scalar");
        let result_data = get_f32_data(&result)?;
        assert_eq!(result_data.len(), 1, "Result should have 1 element");
        assert_relative_eq!(result_data[0], 3.5, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_axis_0() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = mean_axes(&t, &[0], false)?;
        let expected_data = vec![2.5, 3.5, 4.5]; // (1+4)/2, (2+5)/2, (3+6)/2
        check_tensor_near(&result, &[3], &expected_data, 1e-6);
        Ok(())
    }

    // TODO: Add tests for keep_dims

    // --- Autograd Tests ---

    #[test]
    fn test_mean_all_backward() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        t.set_requires_grad(true)?;

        let all_axes: Vec<usize> = (0..t.shape().len()).collect();
        let output = mean_axes(&t, &all_axes, false)?;
        assert!(output.requires_grad(), "Output should require grad");
        assert!(output.grad_fn().is_some(), "Output should have grad_fn");

        let grad_output = Tensor::from_vec_f32(vec![1.0], vec![])?;
        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist");
        
        let n = t.numel() as f32;
        let expected_scale = 1.0 / n;
        let expected_data: Vec<f32> = vec![expected_scale; t.numel()];
        let expected_shape = &[2, 3];

        check_tensor_near(&input_grad.contiguous()?, expected_shape, &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_axis_0_backward() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        t.set_requires_grad(true)?;

        let axis = [0];
        let output = mean_axes(&t, &axis, false)?;
        assert!(output.requires_grad(), "Output should require grad");
        assert!(output.grad_fn().is_some(), "Output should have grad_fn");

        let grad_output_data = vec![0.1, 0.2, 0.3];
        let grad_output = Tensor::from_vec_f32(grad_output_data.clone(), vec![3])?;
        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist");
        
        let n = t.shape()[axis[0]] as f32; // N = 2 (size of dimension 0)
        let expected_scale = 1.0 / n;
        let expected_data: Vec<f32> = grad_output_data.iter()
            .cycle()
            .take(t.numel())
            .map(|&g| g * expected_scale)
            .collect();
            
        let expected_shape = &[2, 3];

        check_tensor_near(&input_grad.contiguous()?, expected_shape, &expected_data, 1e-6);
        Ok(())
    }
} 