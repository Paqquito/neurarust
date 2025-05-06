#[cfg(test)]
mod tests {
    use crate::nn::losses::mse::MSELoss;
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::tensor::from_vec_f32;
    use approx::assert_relative_eq;
    use crate::autograd::grad_check::{check_grad, GradCheckError};

    // Helper pour créer des tenseurs F32
    fn create_test_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        from_vec_f32(data, shape).expect("Test tensor f32 creation failed")
    }
    fn create_grad_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let tensor = from_vec_f32(data, shape).expect("Test grad tensor f32 creation failed");
        let _ = tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_mse_loss_creation() -> Result<(), NeuraRustError> {
        let _mse_mean = MSELoss::new("mean");
        let _mse_sum = MSELoss::new("sum");
        // TODO: Vérifier les champs internes si possible, ou au moins que la création ne panique pas
        // Pour l'instant, on vérifie juste que `new` ne panique pas avec des réductions valides.
        Ok(())
    }

    #[test]
    fn test_mse_loss_forward_basic() -> Result<(), NeuraRustError> {
        let mse = MSELoss::new("mean");
        let input = create_test_tensor_f32(vec![1.0, 2.0], vec![2]);
        let target = create_test_tensor_f32(vec![1.5, 1.0], vec![2]);
        let loss = mse.calculate(&input, &target)?;
        assert_eq!(loss.shape(), &[] as &[usize]); 
        assert!(!loss.requires_grad(), "Loss should not require grad if inputs dont");
        assert_relative_eq!(loss.item_f32()?, 0.625f32, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_forward_mismatched_shapes() {
        let mse = MSELoss::new("mean");
        let input = create_test_tensor_f32(vec![1.0, 2.0], vec![2]);
        let target = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let result = mse.calculate(&input, &target);
        assert!(matches!(result.err().unwrap(), NeuraRustError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_mse_loss_forward_requires_grad() -> Result<(), NeuraRustError> {
        let mse = MSELoss::new("mean");
        let input = create_grad_tensor_f32(vec![1.0, 2.0], vec![2]); 
        let target = create_test_tensor_f32(vec![1.5, 1.0], vec![2]); 
        let loss = mse.calculate(&input, &target)?;
        assert_eq!(loss.shape(), &[] as &[usize]);
        assert!(loss.requires_grad(), "Loss should require grad if input does");
        assert!(loss.grad_fn().is_some(), "Loss should have grad_fn");
        Ok(())
    }

    #[test]
    fn test_mse_loss_backward_input_grad() -> Result<(), NeuraRustError> {
        let mse = MSELoss::new("mean");
        let input = create_grad_tensor_f32(vec![1.0, 2.0], vec![2]);
        let target = create_test_tensor_f32(vec![1.5, 1.0], vec![2]);
        let loss = mse.calculate(&input, &target)?;
        loss.backward(None)?; 

        let grad_input = input.grad().expect("Input grad missing");
        let grad_data = grad_input.get_f32_data()?; 
        let expected_grad = vec![-0.5, 1.0];
        assert_relative_eq!(grad_data.as_slice(), expected_grad.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_backward_target_grad() -> Result<(), NeuraRustError> {
        let mse = MSELoss::new("mean");
        let input = create_test_tensor_f32(vec![1.0, 2.0], vec![2]);
        let target = create_grad_tensor_f32(vec![1.5, 1.0], vec![2]); 
        let loss = mse.calculate(&input, &target)?;
        loss.backward(None)?;

        let grad_target = target.grad().expect("Target grad missing");
        let grad_data = grad_target.get_f32_data()?; 
        let expected_grad = vec![0.5, -1.0];
        assert_relative_eq!(grad_data.as_slice(), expected_grad.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_grad_check_f32() -> Result<(), GradCheckError> {
        let input_data = vec![1.0, 2.0];
        let target_data = vec![1.5, 1.0];
        let shape = vec![2];

        let input = create_grad_tensor_f32(input_data, shape.clone());
        let target = create_grad_tensor_f32(target_data, shape.clone());
        
        let mse = MSELoss::new("mean");

        let mse_func = |inputs: &[Tensor]| -> Result<Tensor, NeuraRustError> {
             if inputs.len() != 2 {
                 return Err(NeuraRustError::UnsupportedOperation("Expected 2 inputs for MSE check_grad func".to_string()));
             }
            mse.calculate(&inputs[0], &inputs[1])
        };
        
        let default_output_grad = crate::tensor::ones(&[]).map_err(GradCheckError::TensorError)?;
        check_grad(mse_func, &[input, target], &default_output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_mse_loss_calculate_mean() -> Result<(), NeuraRustError> { 
        let mse = MSELoss::new("mean");
        let input = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_test_tensor_f32(vec![1.5, 1.0, 3.5, 3.0], vec![2, 2]);
        let result = mse.calculate(&input, &target)?;
        assert_eq!(result.item_f32()?, 0.625_f32);
        assert!(!result.requires_grad());
        Ok(())
    }

    #[test]
    fn test_mse_loss_calculate_sum() -> Result<(), NeuraRustError> { 
        let mse = MSELoss::new("sum");
        let input = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_test_tensor_f32(vec![1.5, 1.0, 3.5, 3.0], vec![2, 2]);
        let result = mse.calculate(&input, &target)?;
        assert_eq!(result.item_f32()?, 2.5_f32);
        assert!(!result.requires_grad());
        Ok(())
    }

    #[test]
    fn test_mse_loss_calculate_backward_mean() -> Result<(), NeuraRustError> {
        let mse = MSELoss::new("mean");
        let input = create_grad_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_test_tensor_f32(vec![1.5, 1.0, 3.5, 3.0], vec![2, 2]);
        let loss = mse.calculate(&input, &target)?;
        assert!(loss.requires_grad());
        loss.backward(None)?;
        let grad = input.grad().unwrap();
        let expected_grad = vec![-0.25, 0.5, -0.25, 0.5];
        let grad_data = grad.get_f32_data()?;
        assert_relative_eq!(grad_data.as_slice(), expected_grad.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_calculate_backward_sum() -> Result<(), NeuraRustError> {
        let mse = MSELoss::new("sum");
        let input = create_grad_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_test_tensor_f32(vec![1.5, 1.0, 3.5, 3.0], vec![2, 2]);
        let loss = mse.calculate(&input, &target)?;
        assert!(loss.requires_grad());
        loss.backward(None)?;
        let grad = input.grad().unwrap();
        let expected_grad = vec![-1.0, 2.0, -1.0, 2.0];
        let grad_data = grad.get_f32_data()?;
        assert_relative_eq!(grad_data.as_slice(), expected_grad.as_slice(), epsilon = 1e-6);
        Ok(())
    }
}
