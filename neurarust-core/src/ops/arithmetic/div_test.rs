use crate::autograd::grad_check::{check_grad, GradCheckError};
use crate::error::NeuraRustError;
use crate::ops::arithmetic::div::div_op;

// Helper (non-generic) pour obtenir les données f32
// fn get_f32_data(tensor: &Tensor) -> Result<Vec<f32>, NeuraRustError> {
//     let guard = tensor.read_data();
//     if guard.dtype != crate::types::DType::F32 || guard.device != crate::device::StorageDevice::CPU {
//         return Err(NeuraRustError::UnsupportedOperation("Test helper requires F32 CPU tensor".to_string()));
//     }
//     match &*guard.buffer {
//         crate::buffer::Buffer::Cpu(crate::buffer::CpuBuffer::F32(data_arc)) => Ok(data_arc.to_vec()),
//         _ => Err(NeuraRustError::UnsupportedOperation("Buffer type not CpuF32".to_string())),
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    // Imports inutilisés commentés
    // use crate::autograd::grad_check::check_grad;
    // use crate::tensor::create; 
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;
    use crate::utils::testing::check_tensor_near; // Utiliser pour comparer f32

    // SUPPRIMER les anciens helpers create_test_tensor*<f64>

    #[test]
    fn test_div_tensors_ok() -> Result<(), NeuraRustError> {
        let a = Tensor::new(vec![10.0f32, 20.0f32, -30.0f32], vec![3])?;
        let b = Tensor::new(vec![2.0f32, 5.0f32, 10.0f32], vec![3])?;
        let result = div_op(&a, &b)?;
        let result_data = result.get_f32_data().expect("Failed to get result data");
        assert_eq!(result.shape(), &[3]);
        assert_relative_eq!(result_data.as_slice(), &[5.0f32, 4.0f32, -3.0f32] as &[f32], epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_div_by_zero() -> Result<(), NeuraRustError> {
        let a = Tensor::new(vec![1.0f32], vec![1])?;
        let b = Tensor::new(vec![0.0f32], vec![1])?;
        // L'opération doit maintenant réussir (Ok) car la division flottante ne panique pas
        let result_tensor = div_op(&a, &b)?;

        // Vérifier que la valeur résultante est infinie
        let data = result_tensor.get_f32_data().expect("Failed to get data");
        assert_eq!(data.len(), 1);
        assert!(data[0].is_infinite(), "Result should be infinite");
        assert!(data[0].is_sign_positive(), "Result should be positive infinity");

        // Test avec -1.0 / 0.0
        let a_neg = Tensor::new(vec![-1.0f32], vec![1])?;
        let result_tensor_neg = div_op(&a_neg, &b)?;
        let data_neg = result_tensor_neg.get_f32_data().expect("Failed to get data");
        assert_eq!(data_neg.len(), 1);
        assert!(data_neg[0].is_infinite(), "Result should be infinite");
        assert!(data_neg[0].is_sign_negative(), "Result should be negative infinity");

        // Test avec 0.0 / 0.0 (devrait être NaN)
        let a_zero = Tensor::new(vec![0.0f32], vec![1])?;
        let result_tensor_nan = div_op(&a_zero, &b)?;
        let data_nan = result_tensor_nan.get_f32_data().expect("Failed to get data");
        assert_eq!(data_nan.len(), 1);
        assert!(data_nan[0].is_nan(), "Result should be NaN");

        Ok(())
    }

    #[test]
    fn test_div_broadcasting() -> Result<(), NeuraRustError> {
        let a = Tensor::new(vec![10.0f32, 20.0f32], vec![2, 1])?;
        let b = Tensor::new(vec![2.0f32, 4.0f32], vec![2])?;
        let result = div_op(&a, &b)?;
        let result_data = result.get_f32_data().expect("Failed to get result data");
        assert_eq!(result.shape(), &[2, 2]);
        let expected_data = vec![5.0f32, 2.5f32, 10.0f32, 5.0f32];
        assert_relative_eq!(result_data.as_slice(), expected_data.as_slice() as &[f32], epsilon = 1e-6);
        Ok(())
    }

    // --- Autograd Tests ---

    #[test]
    fn test_div_backward_simple() -> Result<(), GradCheckError> {
        let a = Tensor::new(vec![10.0f32, 20.0f32], vec![2])?;
        a.set_requires_grad(true)?;
        let b = Tensor::new(vec![2.0f32, 5.0f32], vec![2])?;
        b.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| div_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2];
        let output_grad = Tensor::new(vec![1.0f32, 1.0f32], output_shape)?;
        
        let epsilon = 1e-3;
        let abs_tol = 1e-4;
        let rel_tol = 1e-3;

        check_grad(func, &[a, b], &output_grad, epsilon, abs_tol, rel_tol)
    }

    #[test]
    fn test_div_backward_broadcast() -> Result<(), GradCheckError> {
        let matrix = Tensor::new(vec![10.0f32, 20.0f32, 30.0f32, 40.0f32], vec![2, 2])?;
        matrix.set_requires_grad(true)?;
        let row_vector = Tensor::new(vec![2.0f32, 5.0f32], vec![1, 2])?;
        row_vector.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| div_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::new(vec![1.0f32, 1.0f32, 1.0f32, 1.0f32], output_shape)?;

        let epsilon = 1e-3;
        let abs_tol = 1e-3;
        let rel_tol = 1e-2;

        check_grad(func, &[matrix, row_vector], &output_grad, epsilon, abs_tol, rel_tol)
    }

    #[test]
    fn test_div_backward_with_zero_divisor() -> Result<(), NeuraRustError>{
        let a = Tensor::new(vec![10.0f32], vec![1])?;
        a.set_requires_grad(true)?;
        let b = Tensor::new(vec![0.0001f32], vec![1])?;
        b.set_requires_grad(true)?;
        let output_grad = Tensor::new(vec![1.0f32], vec![1])?;

        let c = div_op(&a, &b)?;

        let backward_result = c.backward(Some(output_grad));
        assert!(backward_result.is_ok(), "Backward failed for small divisor: {:?}", backward_result.err());

        let a_grad_val = a.grad().unwrap().get_f32_data()?[0];
        let b_grad_val = b.grad().unwrap().get_f32_data()?[0];
        assert!(a_grad_val.is_finite());
        assert!(b_grad_val.is_finite());
        assert_relative_eq!(a_grad_val, 1.0 / 0.0001, epsilon = 1e-3);
        assert_relative_eq!(b_grad_val, -10.0 / (0.0001 * 0.0001), epsilon = 1e-3);
        Ok(())
    }

    // --- Tests Backward (Calcul Manuel - intégrés depuis div.rs) --- 

    #[test]
    fn test_div_backward_simple_manual() -> Result<(), NeuraRustError> {
        let a_data = vec![6.0f32, 10.0f32];
        let b_data = vec![2.0f32, 5.0f32];
        let shape = vec![2];
        let a = Tensor::new(a_data.clone(), shape.clone())?;
        let b = Tensor::new(b_data.clone(), shape.clone())?;
        a.set_requires_grad(true)?;
        b.set_requires_grad(true)?;

        let output = div_op(&a, &b)?;
        
        let grad_output_data = vec![0.1f32, 0.2f32];
        let grad_output = Tensor::new(grad_output_data.clone(), shape.clone())?;

        output.backward(Some(grad_output)).expect("Backward pass failed");

        let grad_a_contig = a.grad().unwrap().contiguous()?;
        let expected_grad_a: Vec<f32> = grad_output_data.iter().zip(b_data.iter()).map(|(&g, &bi)| g / bi).collect();
        check_tensor_near(&grad_a_contig, &shape, &expected_grad_a, 1e-6);

        let grad_b_contig = b.grad().unwrap().contiguous()?;
        let expected_grad_b: Vec<f32> = grad_output_data.iter().zip(a_data.iter()).zip(b_data.iter())
            .map(|((&g, &ai), &bi)| g * (-ai / (bi * bi)))
            .collect(); 
        check_tensor_near(&grad_b_contig, &shape, &expected_grad_b, 1e-6);
        Ok(())
    }

    #[test]
    fn test_div_backward_broadcast_manual() -> Result<(), NeuraRustError> {
        let a_data = vec![6.0f32, 10.0f32];
        let b_data = vec![2.0f32, 5.0f32];
        let a_shape = vec![2, 1];
        let b_shape = vec![2];
        let output_shape = vec![2, 2];

        let a = Tensor::new(a_data.clone(), a_shape.clone())?;
        let b = Tensor::new(b_data.clone(), b_shape.clone())?;
        a.set_requires_grad(true)?;
        b.set_requires_grad(true)?;

        let output = div_op(&a, &b)?;
        
        let grad_output_data = vec![0.1f32, 0.2f32, 0.3f32, 0.4f32];
        let grad_output = Tensor::new(grad_output_data.clone(), output_shape.clone())?;

        output.backward(Some(grad_output)).expect("Backward pass failed");

        let grad_a_contig = a.grad().unwrap().contiguous()?;
        let expected_grad_a = vec![0.09f32, 0.23f32];
        check_tensor_near(&grad_a_contig, &a_shape, &expected_grad_a, 1e-6);

        let grad_b_contig = b.grad().unwrap().contiguous()?;
        let expected_grad_b = vec![-0.9f32, -0.208f32];
        check_tensor_near(&grad_b_contig, &b_shape, &expected_grad_b, 1e-6);
        Ok(())
    }

    #[test]
    fn test_div_tensors_i32() {
        let t1 = crate::tensor::from_vec_i32(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let t2 = crate::tensor::from_vec_i32(vec![2, 4, 5, 8], vec![2, 2]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&t1, &t2).unwrap();
        let result_data = result.get_i32_data().unwrap();
        assert_eq!(result_data, vec![5, 5, 6, 5]);
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.dtype(), crate::DType::I32);
    }

    #[test]
    fn test_div_tensors_i64() {
        let t1 = crate::tensor::from_vec_i64(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let t2 = crate::tensor::from_vec_i64(vec![2, 4, 5, 8], vec![2, 2]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&t1, &t2).unwrap();
        let result_data = result.get_i64_data().unwrap();
        assert_eq!(result_data, vec![5, 5, 6, 5]);
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.dtype(), crate::DType::I64);
    }

    #[test]
    fn test_div_broadcasting_i32() {
        let matrix = crate::tensor::from_vec_i32(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let row_vector = crate::tensor::from_vec_i32(vec![2, 4], vec![1, 2]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&matrix, &row_vector).unwrap();
        let result_data = result.get_i32_data().unwrap();
        assert_eq!(result_data, vec![5, 5, 15, 10]);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_div_broadcasting_i64() {
        let matrix = crate::tensor::from_vec_i64(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let row_vector = crate::tensor::from_vec_i64(vec![2, 4], vec![1, 2]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&matrix, &row_vector).unwrap();
        let result_data = result.get_i64_data().unwrap();
        assert_eq!(result_data, vec![5, 5, 15, 10]);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_div_by_zero_i32() {
        let t1 = crate::tensor::from_vec_i32(vec![1, -1, 0], vec![3]).unwrap();
        let t2 = crate::tensor::from_vec_i32(vec![0, 0, 0], vec![3]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&t1, &t2);
        // Division entière par zéro : comportement Rust = panic, mais ici on attend une erreur gérée
        assert!(result.is_err());
    }

    #[test]
    fn test_div_by_zero_i64() {
        let t1 = crate::tensor::from_vec_i64(vec![1, -1, 0], vec![3]).unwrap();
        let t2 = crate::tensor::from_vec_i64(vec![0, 0, 0], vec![3]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&t1, &t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_tensors_shape_mismatch_i32() {
        let t1 = crate::tensor::from_vec_i32(vec![1, 2], vec![2]).unwrap();
        let t2 = crate::tensor::from_vec_i32(vec![1, 2, 3], vec![3]).unwrap();
        let result = crate::ops::arithmetic::div::div_op(&t1, &t2);
        assert!(matches!(result, Err(crate::NeuraRustError::BroadcastError { .. })));
    }
} 