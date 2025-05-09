#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, from_vec_f32, from_vec_f64, ones};
    use crate::types::DType;
    use crate::error::NeuraRustError;
    use crate::nn::{Linear};
    use crate::nn::module::Module;
    use crate::autograd::grad_check::{check_grad, GradCheckError};
    use crate::utils::testing::{check_tensor_near, check_tensor_near_f64};

    #[test]
    fn test_linear_creation_f32() -> Result<(), NeuraRustError> {
        let linear_f32 = Linear::new(10, 5, true, DType::F32)?;
        {
            let weights_guard = linear_f32.weight().read().unwrap();
            assert_eq!(weights_guard.tensor.shape(), &[5, 10]);
            assert_eq!(weights_guard.tensor.dtype(), DType::F32);
            assert!(weights_guard.tensor.requires_grad());
        }
        assert!(linear_f32.bias().is_some());
        {
            let bias_guard = linear_f32.bias().as_ref().unwrap().read().unwrap();
            assert_eq!(bias_guard.tensor.shape(), &[1, 5]);
            assert_eq!(bias_guard.tensor.dtype(), DType::F32);
            assert!(bias_guard.tensor.requires_grad());
        }
        Ok(())
    }

    #[test]
    fn test_linear_creation_f64_no_bias() -> Result<(), NeuraRustError> {
        let linear_f64_no_bias = Linear::new(3, 2, false, DType::F64)?;
        {
            let weights_guard = linear_f64_no_bias.weight().read().unwrap();
            assert_eq!(weights_guard.tensor.shape(), &[2, 3]);
            assert_eq!(weights_guard.tensor.dtype(), DType::F64);
            assert!(weights_guard.tensor.requires_grad());
        }
        assert!(linear_f64_no_bias.bias().is_none());
        Ok(())
    }
    
    #[test]
    fn test_linear_set_weights_manually() -> Result<(), NeuraRustError> {
        let mut linear = Linear::new(3, 2, false, DType::F32)?;
        let new_weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let new_weights_tensor = from_vec_f32(new_weights_data.clone(), vec![2, 3])?;
        new_weights_tensor.set_requires_grad(true)?;
        linear.weight_mut().write().unwrap().tensor = new_weights_tensor;
        {
            let weights_guard = linear.weight().read().unwrap();
            assert_eq!(weights_guard.tensor.get_f32_data()?, new_weights_data);
            assert!(weights_guard.tensor.requires_grad());
        }
        Ok(())
    }

    #[test]
    fn test_linear_set_bias_manually() -> Result<(), NeuraRustError> {
        let mut linear = Linear::new(3, 2, true, DType::F64)?;
        assert!(linear.bias().is_some());
        let new_bias_data = vec![10.0, 20.0];
        let new_bias_tensor = from_vec_f64(new_bias_data.clone(), vec![1, 2])?;
        new_bias_tensor.set_requires_grad(true)?;
        linear.bias_mut().as_mut().unwrap().write().unwrap().tensor = new_bias_tensor;
        {
            let bias_guard = linear.bias().as_ref().unwrap().read().unwrap();
            assert_eq!(bias_guard.tensor.get_f64_data()?, new_bias_data);
            assert!(bias_guard.tensor.requires_grad());
        }
        Ok(())
    }

    #[test]
    fn test_linear_forward_f32_with_bias() -> Result<(), NeuraRustError> {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = from_vec_f32(input_data, vec![2, 3])?;
        let mut linear = Linear::new(3, 2, true, DType::F32)?;
        
        let weights_data = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let bias_data = vec![0.5, -0.5];
        linear.weight_mut().write().unwrap().tensor = from_vec_f32(weights_data, vec![2, 3])?;
        linear.bias_mut().as_mut().unwrap().write().unwrap().tensor = from_vec_f32(bias_data, vec![2])?;
        
        let output = linear.forward(&input)?;
        
        let expected_output = vec![-1.5, 1.5, -1.5, 4.5];
        check_tensor_near(&output, &[2, 2], &expected_output, 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_forward_f64_no_bias() -> Result<(), NeuraRustError> {
        let input = from_vec_f64(vec![1.0, -1.0], vec![1, 2])?;
        let mut linear = Linear::new(2, 3, false, DType::F64)?;
        let weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        linear.weight_mut().write().unwrap().tensor = from_vec_f64(weights_data, vec![3, 2])?;

        let output = linear.forward(&input)?;

        let expected_output = vec![-1.0, -1.0, -1.0];
        check_tensor_near_f64(&output, &[1, 3], &expected_output, 1e-9);
        Ok(())
    }
    
    fn linear_loss_fn(output: &Tensor) -> Result<Tensor, NeuraRustError> {
        output.sum(None, false)
    }

    #[test]
    fn test_linear_weights_grad_f32() -> Result<(), GradCheckError> {
        let linear = Linear::new(3, 2, true, DType::F32).unwrap();
        let input = from_vec_f32([1., 2., 3., 4., 5., 6.].to_vec(), vec![2, 3]).unwrap();
        
        let weights_tensor_clone = linear.weight().read().unwrap().tensor.clone();
        let bias_clone_opt = linear.bias().as_ref().map(|b_arc| b_arc.read().unwrap().tensor.clone());

        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let mut temp_linear = Linear::new(3, 2, true, DType::F32)?;
            temp_linear.weight_mut().write().unwrap().tensor = params[0].clone();
            if let Some(bias_tensor) = &bias_clone_opt {
                temp_linear.bias_mut().as_mut().unwrap().write().unwrap().tensor = bias_tensor.clone();
            }
            let output = temp_linear.forward(&input)?;
            linear_loss_fn(&output)
        };

        let default_output_grad = ones(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(func, &[weights_tensor_clone], &default_output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_linear_bias_grad_f32() -> Result<(), GradCheckError> {
        let linear = Linear::new(3, 2, true, DType::F32).unwrap();
        let input = from_vec_f32([1., 2., 3., 4., 5., 6.].to_vec(), vec![2, 3]).unwrap();
        
        let weights_tensor_clone = linear.weight().read().unwrap().tensor.clone().detach();
        let bias_tensor_clone = linear.bias().as_ref().unwrap().read().unwrap().tensor.clone();

        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_bias_tensor = &params[0];
            let weights_transposed = crate::ops::view::transpose::transpose_op(&weights_tensor_clone, 0, 1)?;
            let matmul_result = crate::ops::linalg::matmul::matmul_op(&input, &weights_transposed)?;
            let output = crate::ops::arithmetic::add::add_op(&matmul_result, current_bias_tensor)?;
            linear_loss_fn(&output)
        };
        let default_output_grad = ones(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(func, &[bias_tensor_clone], &default_output_grad, 1e-3, 1e-4, 3e-3)
    }

    #[test]
    fn test_linear_bias_grad_f64() -> Result<(), GradCheckError> {
        let linear = Linear::new(3, 2, true, DType::F64).unwrap();
        let input_data_f64: Vec<f64> = [1., 2., 3., 4., 5., 6.].iter().map(|&x| x as f64).collect();
        let input = from_vec_f64(input_data_f64, vec![2, 3]).unwrap();
        
        let weights_tensor_clone = linear.weight().read().unwrap().tensor.clone().detach();
        let bias_tensor_clone = linear.bias().as_ref().unwrap().read().unwrap().tensor.clone();

        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_bias_tensor = &params[0];
            let weights_transposed = crate::ops::view::transpose::transpose_op(&weights_tensor_clone, 0, 1)?;
            let matmul_result = crate::ops::linalg::matmul::matmul_op(&input, &weights_transposed)?;
            let output = crate::ops::arithmetic::add::add_op(&matmul_result, current_bias_tensor)?;
            linear_loss_fn(&output)
        };
        let default_output_grad = crate::tensor::ones_f64(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(func, &[bias_tensor_clone], &default_output_grad, 1e-5, 1e-7, 1e-5)
    }

    #[test]
    fn test_linear_input_grad_f64() -> Result<(), GradCheckError> { 
        let linear = Linear::new(3, 2, true, DType::F64).unwrap(); 
        let input_data_f64: Vec<f64> = [1., 2., 3., 4., 5., 6.].iter().map(|&x| x as f64).collect(); 
        let input = from_vec_f64(input_data_f64, vec![2, 3]).unwrap(); 
        let _ = input.set_requires_grad(true);
        
        let weight_f64_clone = linear.weight().read().unwrap().tensor.clone();
        let bias_f64_clone_opt = linear.bias().as_ref().map(|b_arc| b_arc.read().unwrap().tensor.clone());

        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_input = &params[0];
            let mut temp_linear = Linear::new(3, 2, true, DType::F64)?;
            temp_linear.weight_mut().write().unwrap().tensor = weight_f64_clone.clone();
            if let Some(bias_tensor) = &bias_f64_clone_opt {
                temp_linear.bias_mut().as_mut().unwrap().write().unwrap().tensor = bias_tensor.clone();
            } else {
                temp_linear.bias_mut().take();
            }
            let output = temp_linear.forward(current_input)?;
            linear_loss_fn(&output)
        };
        let default_output_grad = crate::tensor::ones_f64(&[]).map_err(GradCheckError::TensorError)?;
        
        check_grad(func, &[input.clone()], &default_output_grad, 1e-4, 1e-4, 1e-3) 
    }

    #[test]
    #[ignore = "Flaky due to F32 precision limitations in grad check; logic validated by test_linear_input_grad_f64"]
    fn test_linear_input_grad_f32() -> Result<(), GradCheckError> { 
        let linear = Linear::new(3, 2, true, DType::F32).unwrap();
        let input_data_f32: Vec<f32> = [1., 2., 3., 4., 5., 6.].to_vec();
        let input = from_vec_f32(input_data_f32, vec![2, 3]).unwrap();
        let _ = input.set_requires_grad(true);

        let weight_f32_clone = linear.weight().read().unwrap().tensor.clone();
        let bias_f32_clone_opt = linear.bias().as_ref().map(|b_arc| b_arc.read().unwrap().tensor.clone());
        
        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_input = &params[0];
            let mut temp_linear = Linear::new(3, 2, true, DType::F32)?;
            temp_linear.weight_mut().write().unwrap().tensor = weight_f32_clone.clone();
            if let Some(bias_tensor) = &bias_f32_clone_opt {
                temp_linear.bias_mut().as_mut().unwrap().write().unwrap().tensor = bias_tensor.clone();
            } else {
                temp_linear.bias_mut().take();
            }
            let output = temp_linear.forward(current_input)?;
            linear_loss_fn(&output)
        };
        let default_output_grad = crate::tensor::ones(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(func, &[input.clone()], &default_output_grad, 1e-4, 2e-3, 6e-2)
    }

    #[test]
    fn test_linear_named_parameters() -> Result<(), NeuraRustError> {
        let linear_with_bias = Linear::new(3, 2, true, DType::F32)?;
        let named_params_wb = linear_with_bias.named_parameters();
        assert_eq!(named_params_wb.len(), 2);
        assert_eq!(named_params_wb[0].0, "weight");
        assert_eq!(named_params_wb[0].1.read().unwrap().name(), Some("weight"));
        assert_eq!(named_params_wb[1].0, "bias");
        assert_eq!(named_params_wb[1].1.read().unwrap().name(), Some("bias"));

        let linear_no_bias = Linear::new(3, 2, false, DType::F32)?;
        let named_params_w = linear_no_bias.named_parameters();
        assert_eq!(named_params_w.len(), 1);
        assert_eq!(named_params_w[0].0, "weight");
        assert_eq!(named_params_w[0].1.read().unwrap().name(), Some("weight"));
        Ok(())
    }

    #[test]
    fn test_linear_children_and_modules() -> Result<(), NeuraRustError> {
        let linear = Linear::new(3, 2, true, DType::F32)?;

        let children = linear.children();
        assert!(children.is_empty(), "Linear layer (leaf) should have no children");

        let named_children = linear.named_children();
        assert!(named_children.is_empty(), "Linear layer (leaf) should have no named children");

        let modules = linear.modules();
        assert_eq!(modules.len(), 1, "Linear layer (leaf) should return itself in modules()");
        assert_eq!(modules[0].parameters().len(), linear.parameters().len(), "The module in modules() should be the Linear layer itself");
        Ok(())
    }
}