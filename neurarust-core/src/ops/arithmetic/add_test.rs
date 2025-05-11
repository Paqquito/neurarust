#[cfg(test)]
mod tests {
    // Imports adjusted for external file
    use crate::ops::arithmetic::add_op;
    use crate::tensor::Tensor;
    use crate::device::StorageDevice;
    use crate::types::DType;
    use crate::error::NeuraRustError;
    use crate::buffer::{Buffer, CpuBuffer};
    // Importer la fonction de vérification
    use crate::utils::testing::check_tensor_near;
    // Remove numeric trait imports as check_grad is commented out
    // use std::ops::{Add, AddAssign};
    // use num_traits::{One, Zero}; 

    // Helper function - Seems correct
    fn get_f32_data(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
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
    fn test_add_tensors_ok() {
        let t1 = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = crate::tensor::from_vec_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let result = add_op(&t1, &t2).unwrap(); 
        let result_data = get_f32_data(&result).unwrap(); 
        assert_eq!(result_data, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.device(), StorageDevice::CPU);
    }

    #[test]
    fn test_add_tensors_shape_mismatch() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = add_op(&t1, &t2);
        assert!(matches!(result, Err(NeuraRustError::BroadcastError { .. })));
    }

    #[test]
    fn test_add_broadcasting() {
        let matrix = crate::tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let row_vector = crate::tensor::from_vec_f32(vec![10.0, 20.0], vec![1, 2]).unwrap();
        let result = add_op(&matrix, &row_vector).unwrap(); 
        let result_data = get_f32_data(&result).unwrap(); 
        assert_eq!(result_data, vec![11.0, 22.0, 13.0, 24.0]);
        assert_eq!(result.shape(), vec![2, 2]); 
    }

    // --- Autograd Tests ---
    #[test]
    // Ignoré enlevé, vérification manuelle
    // #[ignore = "check_grad needs update for non-generic Tensor"]
    fn test_add_backward_simple() -> Result<(), NeuraRustError> { // Ajout du type de retour
        let a_data = vec![1.0f32, 2.0, 3.0];
        let a_shape = vec![3];
        let b_data = vec![4.0f32, 5.0, 6.0];
        let b_shape = vec![3];

        // Pas besoin de func pour la vérif manuelle
        // let func = |inputs: &[Tensor]| add_op(&inputs[0], &inputs[1]);

        let a = crate::tensor::from_vec_f32(a_data.clone(), a_shape.clone())?;
        a.set_requires_grad(true)?;
        let b = crate::tensor::from_vec_f32(b_data.clone(), b_shape.clone())?;
        b.set_requires_grad(true)?;

        // --- Forward ---
        let output = add_op(&a, &b)?;

        // --- Backward ---
        let output_shape = output.shape();
        let output_grad = crate::tensor::create::ones(&output_shape)?;
        output.backward(Some(output_grad))?; // Déclencher le backward

        // --- Vérification Manuelle --- 
        let grad_a = a.grad().expect("Grad A manquant").contiguous()?;
        let grad_b = b.grad().expect("Grad B manquant").contiguous()?;

        // Gradient attendu = output_grad (pas de broadcast)
        let expected_grad_data = vec![1.0, 1.0, 1.0];
        
        check_tensor_near(&grad_a, &a_shape, &expected_grad_data, 1e-6);
        check_tensor_near(&grad_b, &b_shape, &expected_grad_data, 1e-6);

        // Supprimer les variables inutilisées liées à check_grad
        // let epsilon = 1e-4;
        // let tolerance = 1e-2; 

        Ok(())
    }

    #[test]
    // Ignoré enlevé, vérification manuelle
    // #[ignore = "check_grad needs update for non-generic Tensor"]
    fn test_add_backward_broadcast() -> Result<(), NeuraRustError> { // Ajout du type de retour
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_shape = vec![2, 2]; // Matrix
        let b_data = vec![10.0f32, 20.0];
        let b_shape = vec![1, 2]; // Row Vector

        // Pas besoin de func
        // let func = |inputs: &[Tensor]| add_op(&inputs[0], &inputs[1]);

        let a = crate::tensor::from_vec_f32(a_data.clone(), a_shape.clone())?;
        a.set_requires_grad(true)?;
        let b = crate::tensor::from_vec_f32(b_data.clone(), b_shape.clone())?;
        b.set_requires_grad(true)?;

        // --- Forward ---
        let output = add_op(&a, &b)?;
        assert_eq!(output.shape(), &[2, 2]); // Vérifier la forme broadcastée

        // --- Backward ---
        let output_shape = output.shape();
        let output_grad = crate::tensor::create::ones(&output_shape)?;
        output.backward(Some(output_grad))?;

        // --- Vérification Manuelle --- 
        let grad_a = a.grad().expect("Grad A manquant").contiguous()?;
        let grad_b = b.grad().expect("Grad B manquant").contiguous()?;

        // grad_a attendu: output_grad réduit à la forme de a ([2, 2]) -> [[1, 1], [1, 1]]
        let expected_grad_a_data = vec![1.0, 1.0, 1.0, 1.0];
        check_tensor_near(&grad_a, &a_shape, &expected_grad_a_data, 1e-6);

        // grad_b attendu: output_grad réduit à la forme de b ([1, 2]) -> somme sur l'axe 0 -> [1+1, 1+1] = [2, 2]
        let expected_grad_b_data = vec![2.0, 2.0];
        // La forme attendue pour grad_b est la forme originale de b
        let expected_grad_b_shape = vec![1, 2]; 
        check_tensor_near(&grad_b, &expected_grad_b_shape, &expected_grad_b_data, 1e-6);

        // Supprimer les variables inutilisées liées à check_grad
        // let epsilon = 1e-4;
        // let tolerance = 1e-2; 

        Ok(())
    }
}