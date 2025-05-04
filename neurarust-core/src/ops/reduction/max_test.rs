#[cfg(test)]
mod tests {
    // Importer explicitement ce qui est nécessaire
    // use crate::ops::reduction::max::max_op; // Supprimer, on utilise t.max()
    use crate::{Tensor, error::NeuraRustError, tensor::create};
    use crate::utils::testing::check_tensor_near;
    use approx::assert_relative_eq;
     // Pour get_f32_data local
    use crate::types::DType;
    use crate::device::StorageDevice;
    use crate::buffer::{Buffer, CpuBuffer};

    // Helper local pour obtenir f32 data car utils/testing n'est pas dans la portée ?
    // (Alternative: déplacer ce helper dans utils/testing si besoin ailleurs)
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

    // --- Forward Tests ---
    #[test]
    fn test_max_all() -> Result<(), NeuraRustError> {
        let t = Tensor::from_vec_f32(vec![1.0, -2.0, 3.0, 0.0, -5.0, 6.0], vec![2, 3])?;
        let result = t.max(None, false)?;
        let result_data = get_f32_data(&result)?;
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_relative_eq!(result_data[0], 6.0, epsilon = 1e-6);
        Ok(())
    }
    #[test]
    fn test_max_axis0() -> Result<(), NeuraRustError> {
         let t = Tensor::from_vec_f32(vec![1.0, -2.0, 3.0, 0.0, -5.0, 6.0], vec![2, 3])?;
         let result = t.max(Some(&[0]), false)?;
         let result_data = get_f32_data(&result)?;
         assert_eq!(result.shape(), &[3]);
         assert_relative_eq!(result_data.as_slice(), [1.0, -2.0, 6.0].as_slice(), epsilon = 1e-6);
         Ok(())
    }
    #[test]
    fn test_max_axis1_keepdims() -> Result<(), NeuraRustError> {
         let t = Tensor::from_vec_f32(vec![1.0, -2.0, 3.0, 0.0, -5.0, 6.0], vec![2, 3])?;
         let result = t.max(Some(&[1]), true)?;
         let result_data = get_f32_data(&result)?;
         assert_eq!(result.shape(), &[2, 1]);
         assert_relative_eq!(result_data.as_slice(), [3.0, 6.0].as_slice(), epsilon = 1e-6);
         Ok(())
    }
    #[test]
    fn test_max_multiple_axes() -> Result<(), NeuraRustError> {
         let t = Tensor::from_vec_f32((0..24).map(|x| x as f32).collect(), vec![2, 3, 4])?;
         let result = t.max(Some(&[0, 2]), false)?;
         let result_data = get_f32_data(&result)?;
         assert_eq!(result.shape(), &[3]);
         assert_relative_eq!(result_data.as_slice(), [15.0, 19.0, 23.0].as_slice(), epsilon = 1e-6);
         Ok(())
    }
    #[test]
    fn test_max_invalid_axis() -> Result<(), NeuraRustError> { 
         let t = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2])?; // Shape [2]
         let result = t.max(Some(&[1]), false);
         assert!(matches!(result, Err(NeuraRustError::InvalidAxis { axis: 1, rank: 1 })));
         Ok(())
    }

    // --- Backward Tests --- 
    #[test]
    fn test_max_all_backward() -> Result<(), NeuraRustError> {
        println!("Running test_max_all_backward");
        let t_data = vec![1.0, -2.0, 3.0, 0.0, -5.0, 6.0];
        let t_shape = vec![2, 3];
        let t = Tensor::from_vec_f32(t_data, t_shape.clone())?;
        t.set_requires_grad(true)?;

        let output = t.max(None, false)?;
        assert!(output.requires_grad(), "Output should require grad");
        assert!(output.grad_fn().is_some(), "Output should have grad_fn");

        let grad_output = create::ones(&[])?; // Utiliser create::ones

        let backward_result = output.backward(Some(grad_output));
        println!("Backward result: {:?}", backward_result);
        assert!(backward_result.is_ok(), "Backward pass failed");

        let input_grad = t.grad().expect("Input grad should exist").contiguous()?;

        let expected_data = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        check_tensor_near(&input_grad, &t_shape, &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_max_axis_backward() -> Result<(), NeuraRustError> {
        println!("Running test_max_axis_backward");
        let t_data = vec![1.0, 6.0, 3.0, 4.0, 5.0, 2.0];
        let t_shape = vec![2, 3];
        let t = Tensor::from_vec_f32(t_data, t_shape.clone())?;
        t.set_requires_grad(true)?;

        let axis = [0];
        let keep_dims = false;
        let output = t.max(Some(&axis), keep_dims)?;
        assert!(output.requires_grad(), "Output should require grad");
        assert!(output.grad_fn().is_some(), "Output should have grad_fn");
        let output_shape = output.shape().to_vec(); // Convert slice to Vec

        let grad_output_data = vec![0.1, 0.2, 0.3];
        let grad_output = Tensor::from_vec_f32(grad_output_data.clone(), output_shape)?;

        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist").contiguous()?;

        let expected_data = vec![0.0, 0.2, 0.3, 0.1, 0.0, 0.0];
        check_tensor_near(&input_grad, &t_shape, &expected_data, 1e-6);
        Ok(())
    }

    // --- Nouveaux Tests Backward ---

    #[test]
    fn test_max_backward_keep_dims() -> Result<(), NeuraRustError> {
        println!("Running test_max_backward_keep_dims");
        let t_data = vec![1.0, 6.0, 3.0, 4.0, 5.0, 2.0];
        let t_shape = vec![2, 3];
        let t = Tensor::from_vec_f32(t_data, t_shape.clone())?;
        t.set_requires_grad(true)?;

        let axis = [0];
        let keep_dims = true;
        let output = t.max(Some(&axis), keep_dims)?;
        assert_eq!(output.shape(), &[1, 3]);
        assert!(output.requires_grad());
        assert!(output.grad_fn().is_some());

        let grad_output_data = vec![0.1, 0.2, 0.3];
        let grad_output = Tensor::from_vec_f32(grad_output_data.clone(), vec![1, 3])?; // Shape avec keep_dims

        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist").contiguous()?;
        
        // Gradient attendu (idem que sans keep_dims car le broadcast gère la dim ajoutée)
        // Maxima sont à indices [1,0], [0,1], [0,2]
        // t = [[1.0, 6.0, 3.0],
        //      [4.0, 5.0, 2.0]]
        // max(0) = [4.0, 6.0, 3.0]
        // Grad = [[0.0, 0.2, 0.3],
        //         [0.1, 0.0, 0.0]]
        let expected_data = vec![0.0, 0.2, 0.3, 0.1, 0.0, 0.0];
        check_tensor_near(&input_grad, &t_shape, &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_max_backward_multiple_axes() -> Result<(), NeuraRustError> {
        println!("Running test_max_backward_multiple_axes");
        let t_data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
        let t_shape = vec![2, 3, 4];
        let t = Tensor::from_vec_f32(t_data, t_shape.clone())?;
        t.set_requires_grad(true)?;

        let axes = &[0, 2]; // Réduire sur dim 0 et 2
        let keep_dims = false;
        let output = t.max(Some(axes), keep_dims)?;
        assert_eq!(output.shape(), &[3]); // Maxima par dim 1: [15.0, 19.0, 23.0]
        assert!(output.requires_grad());
        assert!(output.grad_fn().is_some());

        let grad_output_data = vec![0.1, 0.2, 0.3];
        let grad_output = Tensor::from_vec_f32(grad_output_data.clone(), vec![3])?;

        output.backward(Some(grad_output))?;

        let input_grad = t.grad().expect("Input grad should exist").contiguous()?;
        
        // Gradient attendu : Le grad_output est placé aux positions des maxima
        // Maxima: 15 (1,3,3), 19 (1,1,3), 23 (1,2,3)
        let mut expected_data = vec![0.0f32; 24];
        expected_data[1*3*4 + 1*4 + 3] = 0.2; // Indice pour 19 (dim 1 = 1)
        expected_data[1*3*4 + 2*4 + 3] = 0.3; // Indice pour 23 (dim 1 = 2)
        expected_data[1*3*4 + 0*4 + 3] = 0.1; // Indice pour 15 (dim 1 = 0)
        
        check_tensor_near(&input_grad, &t_shape, &expected_data, 1e-6);
        Ok(())
    }

    // TODO: Add test for max backward with keep_dims=true
    // TODO: Add test for max backward with multiple axes
} 