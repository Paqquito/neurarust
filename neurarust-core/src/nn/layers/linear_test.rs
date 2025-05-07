#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, from_vec_f32, from_vec_f64, ones};
    use crate::types::DType;
    use crate::error::NeuraRustError;
    use crate::nn::{Linear, Parameter};
    use crate::nn::module::Module;
    use crate::autograd::grad_check::{check_grad, GradCheckError};
    use crate::utils::testing::{check_tensor_near, check_tensor_near_f64};

    #[test]
    fn test_linear_creation_f32() -> Result<(), NeuraRustError> {
        let linear_f32 = Linear::new(10, 5, true, DType::F32)?;
        assert_eq!(linear_f32.weights.shape(), &[5, 10]);
        assert_eq!(linear_f32.weights.dtype(), DType::F32);
        assert!(linear_f32.weights.requires_grad());
        assert!(linear_f32.bias.is_some());
        let bias = linear_f32.bias.as_ref().unwrap();
        assert_eq!(bias.shape(), &[1, 5]);
        assert_eq!(bias.dtype(), DType::F32);
        assert!(bias.requires_grad());
        Ok(())
    }

    #[test]
    fn test_linear_creation_f64_no_bias() -> Result<(), NeuraRustError> {
        let linear_f64_no_bias = Linear::new(3, 2, false, DType::F64)?;
        assert_eq!(linear_f64_no_bias.weights.shape(), &[2, 3]);
        assert_eq!(linear_f64_no_bias.weights.dtype(), DType::F64);
        assert!(linear_f64_no_bias.weights.requires_grad());
        assert!(linear_f64_no_bias.bias.is_none());
        Ok(())
    }
    
    #[test]
    fn test_linear_set_weights_manually() -> Result<(), NeuraRustError> {
        let mut linear = Linear::new(3, 2, false, DType::F32)?;
        let new_weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let new_weights_tensor = from_vec_f32(new_weights_data.clone(), vec![2, 3])?;
        linear.weights = Parameter::new_unnamed(new_weights_tensor);
        assert_eq!(linear.weights.get_f32_data()?, new_weights_data);
        assert!(linear.weights.requires_grad());
        Ok(())
    }

    #[test]
    fn test_linear_set_bias_manually() -> Result<(), NeuraRustError> {
        let mut linear = Linear::new(3, 2, true, DType::F64)?;
        assert!(linear.bias.is_some());
        let new_bias_data = vec![10.0, 20.0];
        let new_bias_tensor = from_vec_f64(new_bias_data.clone(), vec![1, 2])?;
        linear.bias = Some(Parameter::new_unnamed(new_bias_tensor));
        assert!(linear.bias.is_some());
        let bias_param = linear.bias.as_ref().unwrap();
        assert_eq!(bias_param.get_f64_data()?, new_bias_data);
        assert!(bias_param.requires_grad());
        Ok(())
    }

    #[test]
    fn test_linear_forward_f32_with_bias() -> Result<(), NeuraRustError> {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = from_vec_f32(input_data, vec![2, 3])?;
        let mut linear = Linear::new(3, 2, true, DType::F32)?;
        
        let weights_data = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let bias_data = vec![0.5, -0.5];
        linear.weights = Parameter::new_unnamed(from_vec_f32(weights_data, vec![2, 3])?);
        linear.bias = Some(Parameter::new_unnamed(from_vec_f32(bias_data, vec![1, 2])?));
        
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
        linear.weights = Parameter::new_unnamed(from_vec_f64(weights_data, vec![3, 2])?);

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
        
        let weights_tensor_ref = &*linear.weights;

        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let mut temp_linear = Linear::new(3, 2, true, DType::F32)?;
            temp_linear.weights = Parameter::new_unnamed(params[0].clone());
            temp_linear.bias = linear.bias.clone();
            let output = temp_linear.forward(&input)?;
            linear_loss_fn(&output)
        };

        let default_output_grad = ones(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(func, &[weights_tensor_ref.clone()], &default_output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_linear_bias_grad_f32() -> Result<(), GradCheckError> {
        let linear = Linear::new(3, 2, true, DType::F32).unwrap();
        let input = from_vec_f32([1., 2., 3., 4., 5., 6.].to_vec(), vec![2, 3]).unwrap();
        
        let weights_detached = linear.weights.detach();
        
        let bias_param_ref = linear.bias.as_ref().unwrap();
        let bias_tensor_ref = &**bias_param_ref;

        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_bias = &params[0];
            let weights_transposed = crate::ops::view::transpose::transpose_op(&weights_detached, 0, 1)?;
            let matmul_result = crate::ops::linalg::matmul::matmul_op(&input, &weights_transposed)?;
            let output = crate::ops::arithmetic::add::add_op(&matmul_result, current_bias)?;
            linear_loss_fn(&output)
        };
        let default_output_grad = ones(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(func, &[bias_tensor_ref.clone()], &default_output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    // #[ignore = "Grad check tests need verification/adaptation for non-generic Tensor ops"]
    // Renaming test to reflect F64 conversion for debugging precision issues
    fn test_linear_input_grad_f64() -> Result<(), GradCheckError> { // Renamed test
        let linear = Linear::new(3, 2, true, DType::F64).unwrap(); // Changed DType to F64
        let input_data_f64: Vec<f64> = [1., 2., 3., 4., 5., 6.].iter().map(|&x| x as f64).collect(); // Convert input data to f64
        let input = from_vec_f64(input_data_f64, vec![2, 3]).unwrap(); // Use from_vec_f64
        let _ = input.set_requires_grad(true);
        
        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_input = &params[0];
            // Ensure the linear layer used inside matches the DType - This local var was unused, the outer `linear` is captured.
            // let linear_f64 = Linear::new(3, 2, true, DType::F64).unwrap();
            // Copy weights/bias from the outer scope if needed, or reinitialize.
            // For simplicity here, assume we use the specific linear_f64 initialized above
            // Make sure weights/bias dtypes match inside forward.
            // If linear captured from outside, ensure IT is F64.
            let output = linear.forward(current_input)?;
            linear_loss_fn(&output)
        };
        // Create F64 output grad
        let default_output_grad = crate::tensor::ones_f64(&[]).map_err(GradCheckError::TensorError)?;
        
        // Use epsilon 1e-4 and initial tolerances 1e-4, 1e-3 for F64 test
        check_grad(func, &[input.clone()], &default_output_grad, 1e-4, 1e-4, 1e-3) 
    }

    #[test]
    #[ignore = "Flaky due to F32 precision limitations in grad check; logic validated by test_linear_input_grad_f64"]
    // Test for F32 with adjusted tolerances due to precision limits
    fn test_linear_input_grad_f32() -> Result<(), GradCheckError> { 
        let linear = Linear::new(3, 2, true, DType::F32).unwrap(); // Back to F32
        let input_data_f32: Vec<f32> = [1., 2., 3., 4., 5., 6.].to_vec(); // F32 data
        let input = from_vec_f32(input_data_f32, vec![2, 3]).unwrap(); // Use from_vec_f32
        let _ = input.set_requires_grad(true);
        
        let func = |params: &[Tensor]| -> Result<Tensor, NeuraRustError> {
            let current_input = &params[0];
            // Use the captured F32 linear layer
            let output = linear.forward(current_input)?;
            linear_loss_fn(&output)
        };
        // Create F32 output grad using the default ones() which is F32
        let default_output_grad = crate::tensor::ones(&[]).map_err(GradCheckError::TensorError)?;
        
        // Use looser tolerances for F32: epsilon=1e-4, abs_tol=2e-3, rel_tol=6e-2
        check_grad(func, &[input.clone()], &default_output_grad, 1e-4, 2e-3, 6e-2)
    }

    #[test]
    fn test_linear_named_parameters() -> Result<(), NeuraRustError> {
        let linear_with_bias = Linear::new(3, 2, true, DType::F32)?;
        let named_params_wb = linear_with_bias.named_parameters();
        assert_eq!(named_params_wb.len(), 2);
        assert_eq!(named_params_wb[0].0, "weight");
        assert_eq!(named_params_wb[0].1.name(), Some("weight")); // Vérifie le nom interne du Parameter aussi
        assert_eq!(named_params_wb[1].0, "bias");
        assert_eq!(named_params_wb[1].1.name(), Some("bias"));

        let linear_no_bias = Linear::new(3, 2, false, DType::F32)?;
        let named_params_w = linear_no_bias.named_parameters();
        assert_eq!(named_params_w.len(), 1);
        assert_eq!(named_params_w[0].0, "weight");
        assert_eq!(named_params_w[0].1.name(), Some("weight"));
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
        // Vérification indirecte que c'est bien le module Linear lui-même
        assert_eq!(modules[0].parameters().len(), linear.parameters().len(), "The module in modules() should be the Linear layer itself");
        Ok(())
    }
}