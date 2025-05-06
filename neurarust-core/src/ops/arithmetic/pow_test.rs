#[cfg(test)]
mod tests {
    use crate::ops::arithmetic::pow::pow_op;
    use crate::error::NeuraRustError;
    use crate::tensor::{self, Tensor};
    use crate::utils::testing::check_tensor_near;
    use crate::autograd::grad_check::{check_grad, GradCheckError};

    #[test]
    fn test_pow_forward() -> Result<(), NeuraRustError> {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let exponent = Tensor::new(vec![2.0, 3.0, 0.5], vec![3])?; // Use 0.5 for sqrt
        let result = pow_op(&base, &exponent)?;
        let expected_data = vec![1.0, 8.0, 3.0_f32.sqrt()]; // 1^2, 2^3, 3^0.5
        check_tensor_near(&result, &[3], &expected_data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_pow_forward_broadcast() -> Result<(), NeuraRustError> {
        let base = Tensor::new(vec![1.0, 2.0], vec![1, 2])?; // [[1.0, 2.0]]
        let exponent = Tensor::new(vec![2.0, 3.0], vec![2, 1])?; // [[2.0], [3.0]]
        // Expected result shape: [2, 2]
        // [[1^2, 2^2], 
        //  [1^3, 2^3]] = [[1.0, 4.0], [1.0, 8.0]]
        let result = pow_op(&base, &exponent)?;
        let expected_data = vec![1.0, 4.0, 1.0, 8.0]; 
        check_tensor_near(&result, &[2, 2], &expected_data, 1e-6);
        Ok(())
    }

    // --- Autograd Tests ---

    #[test]
    // No ignore needed for F32 simple case if tolerances adjusted
    fn test_pow_backward_simple() -> Result<(), GradCheckError> {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let exponent = Tensor::new(vec![2.0, 3.0, 2.0], vec![3])?;
        base.set_requires_grad(true)?;
        exponent.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = tensor::full(&base.shape(), 1.0)?; // Use tensor::full
        
        // Slightly looser tolerance for F32 pow may be needed
        check_grad(func, &vec![base, exponent], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_pow_backward_only_base_grad() -> Result<(), GradCheckError> {
        let base = Tensor::new(vec![2.0, 3.0], vec![2])?;
        base.set_requires_grad(true)?;
        let exponent = Tensor::new(vec![2.0, 1.0], vec![2])?;
        // exponent does not require grad

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = tensor::full(&base.shape(), 1.0)?;

        check_grad(func, &vec![base, exponent], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    // Ignore still valid as grad for exponent might be tricky/unstable
    fn test_pow_backward_only_exponent_grad() -> Result<(), GradCheckError> {
        let base = Tensor::new(vec![2.0, 3.0], vec![2])?; 
        // base does not require grad
        let exponent = Tensor::new(vec![2.0, 1.0], vec![2])?;
        exponent.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_grad = tensor::full(&base.shape(), 1.0)?;

        check_grad(func, &vec![base, exponent], &output_grad, 1e-3, 1e-4, 1e-3)
    }

    #[test]
    fn test_pow_backward_broadcast_base() -> Result<(), GradCheckError> {
        let base = Tensor::new(vec![2.0, 3.0], vec![1, 2])?; // Shape [1, 2]
        let exponent = Tensor::new(vec![2.0, 1.0], vec![2, 1])?; // Shape [2, 1] -> Output [2, 2]
        base.set_requires_grad(true)?;
        // exponent does not require grad

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], output_shape)?; // Use Tensor::new

        check_grad(func, &vec![base, exponent], &output_grad, 1e-3, 1e-4, 1e-3)
    }
    
    #[test]
    fn test_pow_backward_broadcast_exponent() -> Result<(), GradCheckError> {
        let base = Tensor::new(vec![2.0, 3.0], vec![1, 2])?; // Shape [1, 2]
        // base does not require grad
        let exponent = Tensor::new(vec![2.0, 1.0], vec![2, 1])?; // Shape [2, 1] -> Output [2, 2]
        exponent.set_requires_grad(true)?;

        let func = |inputs: &[Tensor]| pow_op(&inputs[0], &inputs[1]);

        let output_shape = vec![2, 2];
        let output_grad = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], output_shape)?;

        check_grad(func, &vec![base, exponent], &output_grad, 1e-3, 1e-4, 1e-3)
    }
} 