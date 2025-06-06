#[cfg(test)]
mod tests {
    use crate::Tensor;
    use approx::assert_relative_eq;
    
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
        let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = t.mean(None, false)?;
        assert_eq!(result.shape(), &[] as &[usize], "Result shape should be scalar");
        let result_data = get_f32_data(&result)?;
        assert_eq!(result_data.len(), 1, "Result should have 1 element");
        assert_relative_eq!(result_data[0], 3.5, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_axis_0() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = t.mean(Some(&[0]), false)?;
        let expected_data = vec![2.5, 3.5, 4.5]; // (1+4)/2, (2+5)/2, (3+6)/2
        check_tensor_near(&result, &[3], &expected_data, 1e-6);
        Ok(())
    }

    // --- Test pour keep_dims --- 
    #[test]
    fn test_mean_keep_dims() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Keep dims on axis 0
        let result0 = t.mean(Some(&[0]), true)?;
        let expected_data0 = vec![2.5, 3.5, 4.5];
        check_tensor_near(&result0, &[1, 3], &expected_data0, 1e-6);

        // Keep dims on axis 1
        let result1 = t.mean(Some(&[1]), true)?;
        let expected_data1 = vec![2.0, 5.0]; // (1+2+3)/3, (4+5+6)/3
        check_tensor_near(&result1, &[2, 1], &expected_data1, 1e-6);

        // Keep dims on all axes
        let result_all = t.mean(None, true)?;
        let expected_data_all = vec![3.5];
        check_tensor_near(&result_all, &[1, 1], &expected_data_all, 1e-6); // Shape should be [1, 1] for 2D input
        
        Ok(())
    }

    // --- Autograd Tests ---

    #[test]
    fn test_mean_all_backward() -> Result<(), NeuraRustError> {
        let t_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t_shape = vec![2, 3];
        let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
        t.set_requires_grad(true).unwrap();

        let output = t.mean(None, false)?;
        assert!(output.requires_grad(), "Output should require grad");
        assert!(output.grad_fn().is_some(), "Output should have grad_fn");

        let output_shape = vec![]; // Scalar output shape
        let grad_output = crate::tensor::from_vec_f32(vec![1.0], output_shape).unwrap();
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
        let t_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t_shape = vec![2, 3];
        let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
        t.set_requires_grad(true).unwrap();

        let output = t.mean(Some(&[0]), false)?;
        assert!(output.requires_grad(), "Output should require grad");
        assert!(output.grad_fn().is_some(), "Output should have grad_fn");

        let output_shape = vec![3]; // Shape after reduction
        let grad_output_data = vec![0.1, 0.2, 0.3];
        let grad_output = crate::tensor::from_vec_f32(grad_output_data.clone(), output_shape).unwrap();
        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist");
        
        let n = t.shape()[0] as f32; // N = 2 (size of dimension 0)
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

    // --- Test Backward avec keep_dims --- 
    #[test]
    fn test_mean_axis_1_keep_dims_backward() -> Result<(), NeuraRustError> {
        let t_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t_shape = vec![2, 3];
        let t = crate::tensor::from_vec_f32(t_data.clone(), t_shape.clone()).unwrap();
        t.set_requires_grad(true).unwrap();

        let output_shape = vec![2, 1]; // Shape with keep_dims=true
        let grad_output_data = vec![0.1, 0.4];
        let grad_output = crate::tensor::from_vec_f32(grad_output_data.clone(), output_shape).unwrap();

        let func = |inputs: &[Tensor]| inputs[0].mean(Some(&[1]), true);
        // Recalculate output for backward call
        let output = func(&[t.clone()])?;
        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist");
        
        let n = t.shape()[1] as f32; // N = 3 (size of dimension 1)
        let expected_scale = 1.0 / n;

        // Le gradient [0.1, 0.4] de shape [2, 1] doit être broadcasté
        // sur la dimension 1 de l'input [2, 3]. Chaque ligne reçoit le scale * grad correspondant.
        // [[0.1*scale, 0.1*scale, 0.1*scale],
        //  [0.4*scale, 0.4*scale, 0.4*scale]]
        let expected_data = vec![
            0.1 * expected_scale, 0.1 * expected_scale, 0.1 * expected_scale,
            0.4 * expected_scale, 0.4 * expected_scale, 0.4 * expected_scale
        ];
            
        let expected_shape = &[2, 3];
        check_tensor_near(&input_grad.contiguous()?, expected_shape, &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_i32() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_i32(vec![1, 2, 3, 4, 5, 6], vec![2, 3])?;
        let result = t.mean(None, false)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, crate::types::DType::F32);
        let data = match &*guard.buffer {
            crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F32(data_arc)) => data_arc.to_vec(),
            _ => panic!("Buffer type not CpuF32"),
        };
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(data.len(), 1);
        assert_relative_eq!(data[0], 3.5, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_i64_axis1_keepdims() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_i64(vec![10, 20, 30, 40, 50, 60], vec![2, 3])?;
        let result = t.mean(Some(&[1]), true)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, crate::types::DType::F64);
        let data = match &*guard.buffer {
            crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F64(data_arc)) => data_arc.to_vec(),
            _ => panic!("Buffer type not CpuF64"),
        };
        assert_eq!(result.shape(), &[2, 1]);
        assert_relative_eq!(data.as_slice(), &[20.0f64, 50.0f64][..], epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_bool() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_bool(vec![true, false, true, true], vec![2, 2])?;
        let result = t.mean(None, false)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, crate::types::DType::F32);
        let data = match &*guard.buffer {
            crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F32(data_arc)) => data_arc.to_vec(),
            _ => panic!("Buffer type not CpuF32"),
        };
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(data[0], 0.75); // 3/4 true
        Ok(())
    }

    #[test]
    fn test_mean_bool_axis1() -> Result<(), NeuraRustError> {
        let t = crate::tensor::from_vec_bool(vec![true, false, true, true], vec![2, 2])?;
        let result = t.mean(Some(&[1]), false)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, crate::types::DType::F32);
        let data = match &*guard.buffer {
            crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F32(data_arc)) => data_arc.to_vec(),
            _ => panic!("Buffer type not CpuF32"),
        };
        assert_eq!(result.shape(), &[2]);
        assert_relative_eq!(data.as_slice(), &[0.5f32, 1.0f32][..], epsilon = 1e-6);
        Ok(())
    }
} 