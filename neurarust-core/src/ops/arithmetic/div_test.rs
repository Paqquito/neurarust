use super::*;
// use crate::autograd::grad_check::check_grad;
// use crate::tensor::create;
use crate::tensor::Tensor;
use approx::assert_relative_eq;
use crate::error::NeuraRustError;
use crate::utils::testing::check_tensor_near;
use crate::autograd::grad_check::check_grad;

// Helper (non-generic) pour obtenir les données f32
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

#[cfg(test)]
mod tests {
    use super::*;
    // Imports inutilisés commentés
    // use crate::autograd::grad_check::check_grad;
    // use crate::tensor::create; 
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;
    use crate::error::NeuraRustError;
    use crate::utils::testing::check_tensor_near; // Utiliser pour comparer f32
    use crate::autograd::grad_check::check_grad; // Réactiver l'import de check_grad

    // SUPPRIMER les anciens helpers create_test_tensor*<f64>

    #[test]
    fn test_div_tensors_ok() -> Result<(), NeuraRustError> {
        let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2])?; // Utiliser f32
        let b = Tensor::from_vec_f32(vec![2.0, 5.0], vec![2])?;   // Utiliser f32
        let result = div_op(&a, &b)?;
        let expected_data = vec![5.0, 4.0];
        assert_eq!(result.shape(), &[2]);
        let res_data = get_f32_data(&result)?;
        // Utiliser assert_relative_eq pour f32
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_div_by_zero() -> Result<(), NeuraRustError> {
        let a = Tensor::from_vec_f32(vec![10.0], vec![1])?; // Utiliser f32
        let b = Tensor::from_vec_f32(vec![0.0], vec![1])?;   // Utiliser f32
        let result = div_op(&a, &b);
        assert!(matches!(result, Err(NeuraRustError::DivisionByZero)));
        Ok(())
    }

    #[test]
    fn test_div_broadcasting() -> Result<(), NeuraRustError> {
        let matrix = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?; // f32
        let row_vector = Tensor::from_vec_f32(vec![2.0, 5.0], vec![1, 2])?;           // f32
        let result = div_op(&matrix, &row_vector)?;
        let expected_data = vec![5.0, 4.0, 15.0, 8.0];
        assert_eq!(result.shape(), &[2, 2]);
        let res_data = get_f32_data(&result)?;
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    // --- Autograd Tests ---

    #[test]
    fn test_div_backward_simple() -> Result<(), GradCheckError> { // Retourne GradCheckError
        // Utiliser Tensor::from_vec_f32 et set_requires_grad
        let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2])?;
        a.set_requires_grad(true)?;
        let b = Tensor::from_vec_f32(vec![2.0, 5.0], vec![2])?;
        b.set_requires_grad(true)?;

        // La closure attend &[Tensor]
        let func = |inputs: &[Tensor]| div_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2];
        // output_grad doit être f32
        let output_grad = Tensor::from_vec_f32(vec![1.0, 1.0], output_shape)?;
        
        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        // check_grad(func, &[a, b], &output_grad, epsilon, tolerance)
        check_grad(func, &[a, b], &output_grad, epsilon, abs_tol, rel_tol)
        // Le test réussit si check_grad retourne Ok(())
    }

    #[test]
    fn test_div_backward_broadcast() -> Result<(), GradCheckError> { // Retourne GradCheckError
        // Utiliser Tensor::from_vec_f32 et set_requires_grad
        let matrix = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        matrix.set_requires_grad(true)?;
        let row_vector = Tensor::from_vec_f32(vec![2.0, 5.0], vec![1, 2])?;
        row_vector.set_requires_grad(true)?;

        // La closure attend &[Tensor]
        let func = |inputs: &[Tensor]| div_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        // output_grad doit être f32
        let output_grad = Tensor::from_vec_f32(vec![1.0, 1.0, 1.0, 1.0], output_shape)?;

        let epsilon = 1e-5;
        let abs_tol = 1e-7;
        let rel_tol = 1e-5;

        // check_grad(func, &[matrix, row_vector], &output_grad, epsilon, tolerance)
        check_grad(func, &[matrix, row_vector], &output_grad, epsilon, abs_tol, rel_tol)
        // Le test réussit si check_grad retourne Ok(())
    }

    #[test]
    fn test_div_backward_with_zero_divisor() -> Result<(), NeuraRustError>{
        // Utiliser f32
        let a = Tensor::from_vec_f32(vec![10.0], vec![1])?;
        a.set_requires_grad(true)?;
        let b = Tensor::from_vec_f32(vec![0.0001], vec![1])?; // Small divisor
        b.set_requires_grad(true)?;
        let output_grad = Tensor::from_vec_f32(vec![1.0], vec![1])?;

        // The forward pass should work
        let c = div_op(&a, &b)?;

        // The backward pass should also work
        let backward_result = c.backward(Some(output_grad));
        assert!(backward_result.is_ok(), "Backward failed for small divisor: {:?}", backward_result.err());

        // Check gradients are large but finite (check f32 data)
        let a_grad_val = a.grad().unwrap().get_f32_data()?[0];
        let b_grad_val = b.grad().unwrap().get_f32_data()?[0];
        assert!(a_grad_val.is_finite());
        assert!(b_grad_val.is_finite());
        assert_relative_eq!(a_grad_val, 1.0 / 0.0001, epsilon = 1e-3); // Rel tol for f32
        assert_relative_eq!(b_grad_val, -10.0 / (0.0001 * 0.0001), epsilon = 1e-3); // Rel tol for f32
        Ok(())
    }
} 