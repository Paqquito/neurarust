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
    fn test_sum_all() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_op(&t, None, false).unwrap();
        assert_eq!(result.shape(), &[] as &[usize], "Result shape should be scalar");
        let result_val = get_f32_data_helper(&result).unwrap()[0];
        assert_relative_eq!(result_val, 21.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_axis_0() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_op(&t, Some(&[0]), false).unwrap();
        assert_eq!(result.shape(), &[3]);
        let expected_data = vec![5.0, 7.0, 9.0]; // [1+4, 2+5, 3+6]
        let res_data = get_f32_data_helper(&result).unwrap();
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_sum_axis_1() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_op(&t, Some(&[1]), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        let expected_data = vec![6.0, 15.0]; // [1+2+3, 4+5+6]
        let res_data = get_f32_data_helper(&result).unwrap();
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_sum_axes_multiple() {
        let t = Tensor::from_vec_f32((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4]).unwrap();
        let result = sum_op(&t, Some(&[0, 2]), false).unwrap();
        assert_eq!(result.shape(), &[3]);
        let expected_data = vec![68.0, 100.0, 132.0];
        let res_data = get_f32_data_helper(&result).unwrap();
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_sum_keep_dims() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        
        let result0 = sum_op(&t, Some(&[0]), true).unwrap();
        assert_eq!(result0.shape(), &[1, 3]);
        let expected_data0 = vec![5.0, 7.0, 9.0];
        let res_data0 = get_f32_data_helper(&result0).unwrap();
        assert_relative_eq!(res_data0.as_slice(), expected_data0.as_slice(), epsilon = 1e-6);

        let result1 = sum_op(&t, Some(&[1]), true).unwrap();
        assert_eq!(result1.shape(), &[2, 1]);
        let expected_data1 = vec![6.0, 15.0];
        let res_data1 = get_f32_data_helper(&result1).unwrap();
        assert_relative_eq!(res_data1.as_slice(), expected_data1.as_slice(), epsilon = 1e-6);

        let result_all = sum_op(&t, None, true).unwrap(); 
        assert_eq!(result_all.shape(), &[1, 1]);
        let expected_data_all = vec![21.0];
        let res_data_all = get_f32_data_helper(&result_all).unwrap();
        assert_relative_eq!(res_data_all.as_slice(), expected_data_all.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_sum_invalid_axis() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let result = sum_op(&t, Some(&[1]), false);
        assert!(matches!(result, Err(NeuraRustError::InvalidAxis { .. }) | Err(NeuraRustError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_sum_all_non_contiguous() -> Result<(), NeuraRustError> {
        // Créer un tenseur 2x3
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        // Le transposer pour le rendre non contigu (strides: [1, 2])
        let t_transposed = t.transpose(0, 1)?;
        assert!(!t_transposed.is_contiguous(), "Transposed tensor should not be contiguous");
        assert_eq!(t_transposed.shape(), &[3, 2]);
        assert_eq!(t_transposed.strides(), &[1, 2]); // Strides attendus pour transpose sur 2x3

        // Calculer la somme globale sur le tenseur non contigu
        // Utiliser sum_op directement pour tester le noyau
        let sum_result = crate::ops::reduction::sum::sum_op(&t_transposed, None, false)?;
        
        // Vérifier le résultat
        let sum_data = get_f32_data_helper(&sum_result)?;
        assert_eq!(sum_result.shape(), &[] as &[usize], "Sum result shape should be scalar");
        assert_eq!(sum_data.len(), 1, "Sum result should have 1 element");
        approx::assert_relative_eq!(sum_data[0], 21.0, epsilon = 1e-6);

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
        // Utiliser Tensor::from_vec_f32
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_shape = vec![2, 3];
        let input = Tensor::from_vec_f32(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction
        let axes = &[0];
        let keep_dims = true;

        // --- Forward pass ---
        let output = sum_op(&input, Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[1, 3]); // Vérifier la forme de sortie

        // --- Backward pass ---
        let output_grad_data = vec![0.1, 0.2, 0.3]; // Gradient arbitraire
        let output_grad_shape = vec![1, 3];
        let output_grad = Tensor::from_vec_f32(output_grad_data.clone(), output_grad_shape)?;

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
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_shape = vec![2, 3];
        let input = Tensor::from_vec_f32(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction
        let axes = &[1];
        let keep_dims = false;

        // --- Forward pass ---
        let output = sum_op(&input, Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[2]); // Vérifier la forme de sortie [6.0, 15.0]

        // --- Backward pass ---
        let output_grad_data = vec![0.5, -0.1]; // Gradient arbitraire
        let output_grad_shape = vec![2];
        let output_grad = Tensor::from_vec_f32(output_grad_data.clone(), output_grad_shape)?;

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
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_shape = vec![2, 2];
        let input = Tensor::from_vec_f32(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction: None (all)
        let axes = None;
        let keep_dims = true;

        // --- Forward pass ---
        let output = sum_op(&input, axes, keep_dims)?;
        assert_eq!(output.shape(), &[1, 1]); // Output scalaire gardant les dims

        // --- Backward pass ---
        let output_grad_data = vec![5.0]; // Gradient scalaire
        let output_grad_shape = vec![1, 1];
        let output_grad = Tensor::from_vec_f32(output_grad_data.clone(), output_grad_shape)?;

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
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_shape = vec![2, 2];
        let input = Tensor::from_vec_f32(input_data.clone(), input_shape.clone())?;
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
        let input = Tensor::from_vec_f32(input_data.clone(), input_shape.clone())?;
        input.set_requires_grad(true)?;

        // Axe de réduction
        let axes = &[0, 2];
        let keep_dims = false;

        // --- Forward pass ---
        let output = sum_op(&input, Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[3]); // Output shape [68.0, 100.0, 132.0]

        // --- Backward pass ---
        let output_grad_data = vec![0.1, 0.2, 0.3]; // Gradient arbitraire
        let output_grad_shape = vec![3];
        let output_grad = Tensor::from_vec_f32(output_grad_data.clone(), output_grad_shape)?;

        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle du Gradient ---
        let grad_input = input.grad().expect("Gradient d'entrée manquant").contiguous()?;
        
        // Gradient attendu : output_grad broadcasté sur les axes réduits (0 et 2)
        // Le gradient pour la dimension 1 (qui reste) est [0.1, 0.2, 0.3]
        // Ce gradient est appliqué à chaque élément le long des dimensions 0 et 2.
        // Bloc 0 (dim 0):
        //   Ligne 0 (dim 1): [0.1, 0.1, 0.1, 0.1]
        //   Ligne 1 (dim 1): [0.2, 0.2, 0.2, 0.2]
        //   Ligne 2 (dim 1): [0.3, 0.3, 0.3, 0.3]
        // Bloc 1 (dim 0):
        //   Ligne 0 (dim 1): [0.1, 0.1, 0.1, 0.1]
        //   Ligne 1 (dim 1): [0.2, 0.2, 0.2, 0.2]
        //   Ligne 2 (dim 1): [0.3, 0.3, 0.3, 0.3]
        let expected_grad_data = vec![
            0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, // Bloc 0
            0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3  // Bloc 1
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