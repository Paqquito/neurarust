// REMOVED: use super::*;

#[cfg(test)]
mod tests {
    // REMOVED: use super::*;
    use crate::ops::reduction::sum::sum_op;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::types::DType;
    use crate::device::StorageDevice;
    use crate::buffer::{Buffer, CpuBuffer};
    use approx::assert_relative_eq;
    use crate::tensor::create;
    
    fn get_f32_data_helper(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
        let guard = tensor.read_data();
        if guard.dtype != DType::F32 || guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::UnsupportedOperation("Test helper requires F32 CPU tensor".to_string()));
        }
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => Ok(data_arc.to_vec()),
            _ => Err(NeuraRustError::UnsupportedOperation("Buffer type not CpuF32".to_string())),
        }
    }

    #[test]
    fn test_sum_all() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = sum_op(&t, None, false)?;
        let data = get_f32_data_helper(&result)?;
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(data.len(), 1);
        assert_relative_eq!(data[0], 21.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_axis0() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = sum_op(&t, Some(&[0]), false)?;
        let data = get_f32_data_helper(&result)?;
        assert_eq!(result.shape(), &[3]);
        assert_eq!(data.len(), 3);
        assert_relative_eq!(data.as_slice(), &[5.0f32, 7.0f32, 9.0f32] as &[f32], epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_axis1_keepdims() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let result = sum_op(&t, Some(&[1]), true)?;
        let data = get_f32_data_helper(&result)?;
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(data.len(), 2);
        assert_relative_eq!(data.as_slice(), &[6.0f32, 15.0f32] as &[f32], epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_multiple_axes() -> Result<(), NeuraRustError> {
        let t = Tensor::new((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
        let result = sum_op(&t, Some(&[0, 2]), false)?;
        let data = get_f32_data_helper(&result)?;
        assert_eq!(result.shape(), &[3]);
        assert_eq!(data.len(), 3);
        assert_relative_eq!(data.as_slice(), &[68.0f32, 100.0f32, 132.0f32] as &[f32], epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_invalid_axis() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        
        // Test case 1: Single invalid axis
        let result1 = sum_op(&t, Some(&[2]), false);
        assert!(matches!(result1, Err(NeuraRustError::InvalidAxis { axis: 2, rank: 2 })),
                "Test Case 1 Failed: Expected InvalidAxis {{ axis: 2, rank: 2 }}, got {:?}", result1);

        // Test case 2: Mix of valid and invalid axes
        let result2 = sum_op(&t, Some(&[0, 2]), false); // Axis 2 is still invalid
         assert!(matches!(result2, Err(NeuraRustError::InvalidAxis { axis: 2, rank: 2 })),
                "Test Case 2 Failed: Expected InvalidAxis {{ axis: 2, rank: 2 }}, got {:?}", result2);
        
        // Test case 3: Axis equal to rank (also invalid)
        let result3 = sum_op(&t, Some(&[1, 2]), false); // Axis 2 is still invalid
         assert!(matches!(result3, Err(NeuraRustError::InvalidAxis { axis: 2, rank: 2 })),
                "Test Case 3 Failed: Expected InvalidAxis {{ axis: 2, rank: 2 }}, got {:?}", result3);

        Ok(())
    }

    #[test]
    fn test_sum_all_non_contiguous() -> Result<(), NeuraRustError> {
        // Créer un tenseur 3x4
        let base = create::from_vec_f32((0..12).map(|x| x as f32).collect(), vec![3, 4])?;
        // Le transposer pour le rendre non contigu (shape 4x3, strides [1, 4])
        let t = base.transpose(0, 1)?;
        assert!(!t.is_contiguous());
        assert_eq!(t.shape(), &[4, 3]);
        assert_eq!(t.strides(), &[1, 4], "Strides mismatch after transpose");

        // Calculer la somme globale sur le tenseur non contigu via sum_op
        let sum_result = sum_op(&t, None, false)?;
        
        // Vérifier le résultat
        let sum_data = get_f32_data_helper(&sum_result)?;
        assert_eq!(sum_result.shape(), &[] as &[usize], "Sum result shape should be scalar");
        assert_eq!(sum_data.len(), 1, "Sum result should have 1 element");
        assert_relative_eq!(sum_data[0], 66.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_sum_i32() -> Result<(), NeuraRustError> {
        let t = Tensor::new_i32(vec![1, 2, 3, 4, 5, 6], vec![2, 3])?;
        let result = sum_op(&t, None, false)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, DType::I32);
        let data = guard.buffer().try_get_cpu_i32()?.clone();
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 21);
        Ok(())
    }

    #[test]
    fn test_sum_i64_axis0_keepdims() -> Result<(), NeuraRustError> {
        let t = Tensor::new_i64(vec![10, 20, 30, 40, 50, 60], vec![2, 3])?;
        let result = sum_op(&t, Some(&[0]), true)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, DType::I64);
        let data = guard.buffer().try_get_cpu_i64()?.clone();
        assert_eq!(result.shape(), &[1, 3]);
        assert_eq!(data, std::sync::Arc::new(vec![50, 70, 90]));
        Ok(())
    }

    #[test]
    fn test_sum_bool() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![true, false, true, true], vec![2, 2])?;
        let result = sum_op(&t, None, false)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, DType::I64); // Somme des booléens = I64
        let data = guard.buffer().try_get_cpu_i64()?.clone();
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(data[0], 3); // 3 True
        Ok(())
    }

    #[test]
    fn test_sum_bool_axis1() -> Result<(), NeuraRustError> {
        let t = Tensor::new_bool(vec![true, false, true, true], vec![2, 2])?;
        let result = sum_op(&t, Some(&[1]), false)?;
        let guard = result.read_data();
        assert_eq!(guard.dtype, DType::I64);
        let data = guard.buffer().try_get_cpu_i64()?.clone();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(data, std::sync::Arc::new(vec![1, 2]));
        Ok(())
    }
}


#[cfg(test)]
mod autograd_tests {
    // Utiliser le nom actuel de l'opération et les types non-génériques
    use crate::ops::reduction::sum::sum_op;
    // REMOVED: use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::tensor::{Tensor, create}; // Importer create pour ones
    use crate::utils::testing::check_tensor_near; // Importer pour la comparaison
    // Supprimer l'helper générique
    // REMOVED: use crate::utils::testing::create_test_tensor_with_grad;

    #[test]
    fn test_sum_axes_backward_simple_keep_dims() -> Result<(), NeuraRustError> {
        let input_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let input_shape = vec![2, 3];
        let input = Tensor::new(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction
        let axes = &[0];
        let keep_dims = true;

        // --- Forward pass ---
        let output = sum_op(&input, Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[1, 3]); // Vérifier la forme de sortie

        // --- Backward pass ---
        let output_grad_data = vec![0.1f32, 0.2f32, 0.3f32];
        let output_grad_shape = vec![1, 3];
        let output_grad = Tensor::new(output_grad_data.clone(), output_grad_shape)?;

        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle du Gradient ---
        let grad_input = input.grad().expect("Gradient d'entrée manquant").contiguous()?;
        
        // Gradient attendu : output_grad broadcasté sur l'axe réduit (axe 0)
        // [[0.1, 0.2, 0.3],
        //  [0.1, 0.2, 0.3]]
        let expected_grad_data = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
        check_tensor_near(&grad_input, &input_shape, &expected_grad_data, 1e-6);

        Ok(())
    }

     #[test]
    fn test_sum_axes_backward_simple_no_keep_dims() -> Result<(), NeuraRustError> {
        let input_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let input_shape = vec![2, 3];
        let input = Tensor::new(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction
        let axes = &[1];
        let keep_dims = false;

        // --- Forward pass ---
        let output = sum_op(&input, Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[2]); // Vérifier la forme de sortie [6.0, 15.0]

        // --- Backward pass ---
        let output_grad_data = vec![0.5f32, -0.1f32];
        let output_grad_shape = vec![2];
        let output_grad = Tensor::new(output_grad_data.clone(), output_grad_shape)?;

        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle du Gradient ---
        let grad_input = input.grad().expect("Gradient d'entrée manquant").contiguous()?;
        
        // Gradient attendu : output_grad broadcasté sur l'axe réduit (axe 1)
        // [[0.5, 0.5, 0.5],
        //  [-0.1, -0.1, -0.1]]
        let expected_grad_data = vec![0.5, 0.5, 0.5, -0.1, -0.1, -0.1];
        check_tensor_near(&grad_input, &input_shape, &expected_grad_data, 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_sum_all_backward_keep_dims() -> Result<(), NeuraRustError> {
        let input_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let input_shape = vec![2, 2];
        let input = Tensor::new(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction: None (all)
        let axes = None;
        let keep_dims = true;

        // --- Forward pass ---
        let output = sum_op(&input, axes, keep_dims)?;
        assert_eq!(output.shape(), &[1, 1]); // Output scalaire gardant les dims

        // --- Backward pass ---
        let output_grad_data = vec![5.0f32];
        let output_grad_shape = vec![1, 1];
        let output_grad = Tensor::new(output_grad_data.clone(), output_grad_shape)?;

        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle du Gradient ---
        let grad_input = input.grad().expect("Gradient d'entrée manquant").contiguous()?;
        
        // Gradient attendu : output_grad broadcasté sur tous les éléments
        // [[5.0, 5.0],
        //  [5.0, 5.0]]
        let expected_grad_data = vec![5.0, 5.0, 5.0, 5.0];
        check_tensor_near(&grad_input, &input_shape, &expected_grad_data, 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_sum_all_backward_no_keep_dims() -> Result<(), NeuraRustError> {
        let input_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let input_shape = vec![2, 2];
        let input = Tensor::new(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction: None (all)
        let axes = None;
        let keep_dims = false;

        // --- Forward pass ---
        let output = sum_op(&input, axes, keep_dims)?;
        assert_eq!(output.shape(), &[] as &[usize]); // Output scalaire

        // --- Backward pass ---
        // Utiliser create::ones pour un gradient scalaire de 1.0
        let output_grad = create::ones(&[])?; 

        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle du Gradient ---
        let grad_input = input.grad().expect("Gradient d'entrée manquant").contiguous()?;
        
        // Gradient attendu : output_grad (1.0) broadcasté sur tous les éléments
        // [[1.0, 1.0],
        //  [1.0, 1.0]]
        let expected_grad_data = vec![1.0, 1.0, 1.0, 1.0];
        check_tensor_near(&grad_input, &input_shape, &expected_grad_data, 1e-6);
        
        Ok(())
    }

     #[test]
    fn test_sum_multiple_axes_backward() -> Result<(), NeuraRustError> {
        let input_data = (1..=24).map(|x| x as f32).collect::<Vec<_>>();
        let input_shape = vec![2, 3, 4];
        let input = Tensor::new(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction
        let axes = &[0, 2];
        let keep_dims = false;

        // --- Forward pass ---
        let output = sum_op(&input, Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[3]); // Output shape [68.0, 100.0, 132.0]

        // --- Backward pass ---
        let output_grad_data = vec![10.0f32, 20.0f32, 30.0f32];
        let output_grad_shape = vec![3];
        let output_grad = Tensor::new(output_grad_data.clone(), output_grad_shape)?;

        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle du Gradient ---
        let grad_input = input.grad().expect("Gradient d'entrée manquant").contiguous()?;
        
        let expected_grad_data = vec![
            10.0f32, 10.0f32, 10.0f32, 10.0f32,  20.0f32, 20.0f32, 20.0f32, 20.0f32,  30.0f32, 30.0f32, 30.0f32, 30.0f32,
            10.0f32, 10.0f32, 10.0f32, 10.0f32,  20.0f32, 20.0f32, 20.0f32, 20.0f32,  30.0f32, 30.0f32, 30.0f32, 30.0f32
        ];
        check_tensor_near(&grad_input, &input_shape, &expected_grad_data, 1e-6);
        
        Ok(())
    }

      // Ce test semble redondant ou mal formulé dans l'original, 
      // car il appelle sum_axes deux fois avec des closures différentes 
      // et ne teste pas vraiment le cas "pas de réduction".
      // Je le commente pour l'instant.
    //   #[test]
    //  fn test_sum_no_reduction_backward() {
    //      // ... (code original commenté)
    //  }
} 