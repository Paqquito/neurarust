#[cfg(test)]
mod tests {
    use super::*; // Référence mse.rs
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    use crate::types::{DType, NeuraNumeric}; // Pour check_grad helper
    use crate::autograd::grad_check::{check_grad, GradCheckError};
    use crate::ops;
    use std::fmt::Debug;
    use std::ops::{AddAssign, Mul, Neg, Sub, Add, Div};
    use std::iter::Sum;
    use num_traits::{Zero, One, FromPrimitive};
    
    // Helper functions (gardées ici car spécifiques aux tests mse)
    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape, DType::F32).expect("Test tensor creation failed")
    }
    fn create_tensor_f32_grad(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        // Crée un Tensor qui nécessite un gradient
        let mut tensor = Tensor::new(data, shape, DType::F32).expect("Test grad tensor creation failed");
        tensor.set_requires_grad(true);
        tensor
    }
    fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "assertion failed: `(left ≈ right)` (left: `{}`, right: `{}`)", a, b);
    }

    #[test]
    fn test_mse_loss_forward() -> Result<(), NeuraRustError> {
        let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5], vec![3]);
        let loss = mse.forward(&input, &target)?;
        assert_eq!(loss.shape(), &Vec::<usize>::new()); // Check shape is scalar
        assert_approx_eq(loss.get_f32_data().unwrap()[0], 0.25, 1e-6);
        assert!(!loss.requires_grad());
        Ok(())
    }
    
     #[test]
    fn test_mse_loss_forward_shape_mismatch() {
        let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32(vec![1.5, 2.5], vec![2]);
        let result = mse.forward(&input, &target);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::IncompatibleShapes { .. }));
    }

    #[test]
    fn test_mse_loss_backward() -> Result<(), NeuraRustError> {
        let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5], vec![3]);
        let loss = mse.forward(&input, &target)?;
        assert!(loss.requires_grad());
        assert_eq!(loss.shape(), &Vec::<usize>::new());
        assert_approx_eq(loss.get_f32_data().unwrap()[0], 0.25, 1e-6);
        
        loss.backward(None).expect("Backward pass failed"); 
        
        let grad_input_opt = input.grad();
        assert!(grad_input_opt.is_some());
        let grad_input = grad_input_opt.unwrap(); // Use unwrap after is_some check
        assert_eq!(grad_input.shape(), &vec![3]);
        let grad_data = grad_input.get_f32_data().unwrap();
        // Expected: 2/n * (input - target)
        // 2/3 * (1.0 - 1.5) = 2/3 * -0.5 = -1/3
        // 2/3 * (2.0 - 2.5) = 2/3 * -0.5 = -1/3
        // 2/3 * (3.0 - 3.5) = 2/3 * -0.5 = -1/3
        assert_approx_eq(grad_data[0], -1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_data[1], -1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_data[2], -1.0 / 3.0, 1e-6);
        assert!(target.grad().is_none());
        Ok(())
    }
    
    #[test]
    fn test_mse_loss_backward_target_grad() -> Result<(), NeuraRustError> {
         let mut mse = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let target = create_tensor_f32_grad(vec![1.5, 2.5, 3.5], vec![3]); 
        let loss = mse.forward(&input, &target)?;
        assert!(loss.requires_grad());
        
        loss.backward(None).expect("Backward pass failed");
        
        assert!(input.grad().is_none());
        let grad_target_opt = target.grad();
        assert!(grad_target_opt.is_some());
        let grad_target = grad_target_opt.unwrap();
        assert_eq!(grad_target.shape(), &vec![3]);
        let grad_data = grad_target.get_f32_data().unwrap();
        // Expected: -2/n * (input - target) = 2/n * (target - input)
        // 2/3 * (1.5 - 1.0) = 2/3 * 0.5 = 1/3
        // 2/3 * (2.5 - 2.0) = 2/3 * 0.5 = 1/3
        // 2/3 * (3.5 - 3.0) = 2/3 * 0.5 = 1/3
        assert_approx_eq(grad_data[0], 1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_data[1], 1.0 / 3.0, 1e-6);
        assert_approx_eq(grad_data[2], 1.0 / 3.0, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_creation() {
        let mse_mean = MSELoss::new(Reduction::Mean);
        assert_eq!(mse_mean.reduction, Reduction::Mean);
        
        let mse_sum = MSELoss::new(Reduction::Sum);
        assert_eq!(mse_sum.reduction, Reduction::Sum);

        let mse_default = MSELoss::default();
        assert_eq!(mse_default.reduction, Reduction::Mean);
    }

    #[test]
    #[should_panic(expected = "not yet supported")]
    fn test_mse_loss_none_panic() {
        let _ = MSELoss::new(Reduction::None);
    }

    #[test]
    fn test_mse_loss_shape_mismatch() {
        let mut loss = MSELoss::default();
        let input = create_tensor_f32(vec![1., 2.], vec![2]);
        let target = create_tensor_f32(vec![1., 2., 3.], vec![3]);
        let result = loss.forward(&input, &target);
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::IncompatibleShapes { shape1, shape2 } => {
                assert_eq!(shape1, &vec![2]); // Comparaison avec référence
                assert_eq!(shape2, &vec![3]); // Comparaison avec référence
            }
            e => panic!("Expected IncompatibleShapes error, got {:?}", e),
        }
    }

    #[test]
    fn test_mse_loss_forward_mean() {
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        // sum(([-0.5, -0.5, -0.5, -0.5])^2) / 4 = sum([0.25, 0.25, 0.25, 0.25]) / 4 = 1.0 / 4 = 0.25
        assert_approx_eq(result.get_f32_data().unwrap()[0], 0.25, 1e-6);
        assert!(!result.requires_grad());
    }
    
    #[test]
    fn test_mse_loss_forward_sum() {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        // sum(([-0.5, -0.5, -0.5, -0.5])^2) = sum([0.25, 0.25, 0.25, 0.25]) = 1.0
        assert_approx_eq(result.get_f32_data().unwrap()[0], 1.0, 1e-6);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_mse_loss_backward_mean() {
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        assert!(result.requires_grad());
        
        result.backward(None).expect("Backward pass failed");

        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), &vec![2, 2]);
        let grad_data = grad.get_f32_data().unwrap();
        // Expected: 2/n * (input - target)
        // 2/4 * (input - target) = 0.5 * [-0.5, -0.5, -0.5, -0.5] = [-0.25, -0.25, -0.25, -0.25]
        assert_approx_eq(grad_data[0], -0.25, 1e-6);
        assert_approx_eq(grad_data[1], -0.25, 1e-6);
        assert_approx_eq(grad_data[2], -0.25, 1e-6);
        assert_approx_eq(grad_data[3], -0.25, 1e-6);
    }
    
    #[test]
    fn test_mse_loss_backward_sum() {
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target).expect("Forward failed");
        assert!(result.requires_grad());
        
        result.backward(None).expect("Backward pass failed");

        let grad = input.grad().expect("Input gradient missing");
        assert_eq!(grad.shape(), &vec![2, 2]);
        let grad_data = grad.get_f32_data().unwrap();
        // Expected: 2 * (input - target)
        // 2 * [-0.5, -0.5, -0.5, -0.5] = [-1.0, -1.0, -1.0, -1.0]
        assert_approx_eq(grad_data[0], -1.0, 1e-6);
        assert_approx_eq(grad_data[1], -1.0, 1e-6);
        assert_approx_eq(grad_data[2], -1.0, 1e-6);
        assert_approx_eq(grad_data[3], -1.0, 1e-6);
    }

    // --- Tests renommés/ajoutés --- 
    // Les tests suivants semblent redondants ou similaires à ceux ci-dessus, 
    // mais gardons-les pour l'instant pour voir s'ils passent.
    // Note: les noms originaux sont conservés s'ils existent dans l'outline fourni précédemment.

    #[test]
    fn test_mse_loss_reduction_mean() -> Result<(), NeuraRustError> {
        // Similaire à test_mse_loss_forward_mean
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result_tensor = loss.forward(&input, &target)?;
        assert_approx_eq(result_tensor.get_f32_data().unwrap()[0], 0.25, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_reduction_sum() -> Result<(), NeuraRustError> {
        // Similaire à test_mse_loss_forward_sum
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result_tensor = loss.forward(&input, &target)?;
        assert_approx_eq(result_tensor.get_f32_data().unwrap()[0], 1.0, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_backward_reduction_mean() -> Result<(), NeuraRustError> {
        // Similaire à test_mse_loss_backward_mean
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target)?;
        result.backward(None)?;
        let grad = input.grad().expect("Input gradient missing");
        let grad_data = grad.get_f32_data().unwrap();
        assert_approx_eq(grad_data[0], -0.25, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_backward_reduction_sum() -> Result<(), NeuraRustError> {
        // Similaire à test_mse_loss_backward_sum
        let mut loss = MSELoss::new(Reduction::Sum);
        let input = create_tensor_f32_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = create_tensor_f32(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2]);
        let result = loss.forward(&input, &target)?;
        result.backward(None)?;
        let grad = input.grad().expect("Input gradient missing");
        let grad_data = grad.get_f32_data().unwrap();
        assert_approx_eq(grad_data[0], -1.0, 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_no_grad_target_requires_grad() -> Result<(), NeuraRustError> {
        // Teste que si l'input n'a pas de grad mais la target oui, la loss a un grad.
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32(vec![1.0, 2.0], vec![2]); // input no grad
        let target = create_tensor_f32_grad(vec![1.5, 2.5], vec![2]); // target grad
        let result = loss.forward(&input, &target)?;
        assert!(result.requires_grad(), "Loss should require grad if target does");
        result.backward(None)?;
        assert!(input.grad().is_none(), "Input should not have grad");
        assert!(target.grad().is_some(), "Target should have grad");
        Ok(())
    }

    #[test]
    fn test_mse_loss_no_grad_input_requires_grad() -> Result<(), NeuraRustError> {
        // Teste que si la target n'a pas de grad mais l'input oui, la loss a un grad.
        let mut loss = MSELoss::new(Reduction::Mean);
        let input = create_tensor_f32_grad(vec![1.0, 2.0], vec![2]); // input grad
        let target = create_tensor_f32(vec![1.5, 2.5], vec![2]); // target no grad
        let result = loss.forward(&input, &target)?;
        assert!(result.requires_grad(), "Loss should require grad if input does");
        result.backward(None)?;
        assert!(input.grad().is_some(), "Input should have grad");
        assert!(target.grad().is_none(), "Target should not have grad");
        Ok(())
    }

    // --- Autograd Check Tests --- 
    // (Déjà ajouté dans l'edit précédent)

    // Helper pour check_grad: calcule la perte MSE pour un input donné
    fn mse_loss_for_grad_check<T>(
        input_arg: &Tensor<T>,
        target_fixed: &Tensor<T>,
        reduction: Reduction,
    ) -> Result<Tensor<T>, NeuraRustError>
    where
        T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + Neg<Output=T> 
         + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + PartialEq + FromPrimitive,
    {
        // Recalcul manuel basé sur la définition de MSE
        let diff = ops::arithmetic::sub_op(input_arg, target_fixed)?;
        let sq_diff = ops::arithmetic::mul_op(&diff, &diff)?;
        
        match reduction {
            Reduction::Sum => ops::reduction::sum::sum(&sq_diff),
            Reduction::Mean => {
                let sum_val = ops::reduction::sum::sum(&sq_diff)?;
                let n = input_arg.numel();
                let n_t = T::from_usize(n).ok_or_else(|| 
                    NeuraRustError::InternalError("Failed numel conversion".to_string())
                )?;
                let n_tensor = Tensor::scalar(n_t);
                ops::arithmetic::div_op(&sum_val, &n_tensor)
            }
            Reduction::None => unimplemented!(), // Garder unimplemented pour l'instant
        }
    }

    #[test]
    fn test_mse_loss_input_grad_check_mean_f32() -> Result<(), GradCheckError<f32>> {
        let reduction = Reduction::Mean;
        let mut input = Tensor::<f32>::randn(vec![2, 3], DType::F32);
        input.set_requires_grad(true);
        let target = Tensor::<f32>::randn(vec![2, 3], DType::F32);

        let func = |input_for_check: &Tensor<f32>| -> Result<Tensor<f32>, NeuraRustError> {
            mse_loss_for_grad_check(input_for_check, &target, reduction)
        };

        check_grad(func, &input, 1e-3, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_mse_loss_input_grad_check_sum_f32() -> Result<(), GradCheckError<f32>> {
        let reduction = Reduction::Sum;
        let mut input = Tensor::<f32>::randn(vec![4, 1], DType::F32);
        input.set_requires_grad(true);
        let target = Tensor::<f32>::randn(vec![4, 1], DType::F32);

        let func = |input_for_check: &Tensor<f32>| -> Result<Tensor<f32>, NeuraRustError> {
            mse_loss_for_grad_check(input_for_check, &target, reduction)
        };

        check_grad(func, &input, 1e-3, 1e-5)?;
        Ok(())
    }
    
    // TODO: Ajouter check_grad pour target si nécessaire (devrait être l'opposé de input grad)
    // TODO: Ajouter check_grad pour f64 pour une meilleure précision
}
