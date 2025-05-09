#[cfg(test)]
mod tests {
    use crate::device::StorageDevice;
    use crate::nn::parameter::Parameter;
    use crate::optim::grad_clipping::clip_grad_value_;
    use crate::tensor::Tensor;
    use crate::types::DType;
    use crate::NeuraRustError;
    use approx::assert_relative_eq;
    // Removed unused Arc, RwLock as Parameter::new and direct grad manipulation is simpler for tests.

    fn create_param_with_grad(
        shape: &[usize],
        dtype: DType,
        grad_data_opt: Option<Vec<f32>>, // Using f32 for simplicity in test data
        requires_grad: bool,
    ) -> Parameter {
        let mut tensor = Tensor::zeros(shape, dtype, StorageDevice::Cpu);
        if requires_grad {
            tensor.set_requires_grad(true); 
        }

        if let Some(grad_data_vec) = grad_data_opt {
            let grad_tensor_val: Tensor = match dtype {
                DType::F32 => Tensor::from_vec_f32(grad_data_vec, shape, StorageDevice::Cpu),
                DType::F64 => Tensor::from_vec_f64(
                    grad_data_vec.into_iter().map(|x| x as f64).collect(),
                    shape,
                    StorageDevice::Cpu,
                ),
                _ => panic!("Test setup error: grad_data provided for non-float type for clipping test"),
            };
            // Access TensorData to set the grad field
            tensor.data.write().unwrap().grad = Some(grad_tensor_val);
        }
        Parameter::new(tensor, Some("test_param".to_string()))
    }

    #[test]
    fn test_clip_grad_value_f32() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2, 2], DType::F32, Some(vec![-5.0, 2.0, 3.0, -0.5]), true);
        let mut params_vec = vec![param1];

        clip_grad_value_(params_vec.iter_mut(), 1.0)?;

        let grad_tensor_data = params_vec[0].data.read().unwrap();
        let grad_tensor = grad_tensor_data.grad.as_ref().unwrap();
        let grad_data = grad_tensor.get_f32_data()?;
        assert_eq!(grad_data, vec![-1.0, 1.0, 1.0, -0.5]);
        Ok(())
    }

    #[test]
    fn test_clip_grad_value_f64() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2, 2], DType::F64, Some(vec![-5.0, 2.0, 3.0, -0.5]), true);
        let mut params_vec = vec![param1];

        clip_grad_value_(params_vec.iter_mut(), 1.5)?;

        let grad_tensor_data = params_vec[0].data.read().unwrap();
        let grad_tensor = grad_tensor_data.grad.as_ref().unwrap();
        let grad_data = grad_tensor.get_f64_data()?;
        assert_eq!(grad_data, vec![-1.5, 1.5, 1.5, -0.5]);
        Ok(())
    }

    #[test]
    fn test_clip_grad_value_no_grad() -> Result<(), NeuraRustError> {
        let param_no_grad = create_param_with_grad(&[1], DType::F32, None, true);
        let param_with_grad = create_param_with_grad(&[1], DType::F32, Some(vec![10.0]), true);
        let mut params_vec = vec![param_no_grad, param_with_grad];

        clip_grad_value_(params_vec.iter_mut(), 1.0)?;

        assert!(params_vec[0].data.read().unwrap().grad.is_none());
        
        let grad_tensor_data_with_grad = params_vec[1].data.read().unwrap();
        let grad_tensor_with_grad = grad_tensor_data_with_grad.grad.as_ref().unwrap();
        assert_eq!(grad_tensor_with_grad.get_f32_data()?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn test_clip_grad_value_already_in_bounds() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![-0.5, 0.7]), true);
        let mut params_vec = vec![param1];

        clip_grad_value_(params_vec.iter_mut(), 1.0)?;
        
        let grad_tensor_data = params_vec[0].data.read().unwrap();
        let grad_tensor = grad_tensor_data.grad.as_ref().unwrap();
        let grad_data = grad_tensor.get_f32_data()?;
        assert_eq!(grad_data, vec![-0.5, 0.7]);
        Ok(())
    }
    
    #[test]
    fn test_clip_grad_value_negative_clip_value() {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![-0.5, 0.7]), true);
        let mut params_vec = vec![param1]; 

        let result = clip_grad_value_(params_vec.iter_mut(), -1.0);
        assert!(matches!(result, Err(NeuraRustError::ConfigurationError(msg)) if msg == "clip_value must be non-negative"));
    }

    #[test]
    fn test_clip_grad_value_non_float_grad_error_setup() -> Result<(), NeuraRustError> {
        let mut tensor_for_param = Tensor::zeros(&[1], DType::I32, StorageDevice::Cpu);
        tensor_for_param.set_requires_grad(true);

        let grad_i32 = Tensor::from_vec_i32(vec![10], &[1], StorageDevice::Cpu);
        
        // Directly assign the I32 gradient to the TensorData of the tensor
        tensor_for_param.data.write().unwrap().grad = Some(grad_i32);
        
        let mut param_i32 = Parameter::new(tensor_for_param, Some("param_i32".to_string()));
        let mut params_vec = vec![param_i32];

        let result = clip_grad_value_(params_vec.iter_mut(), 1.0);
        
        assert!(
            matches!(result, Err(NeuraRustError::DataTypeMismatch { expected, actual, operation }) 
                if expected == DType::F32 && actual == DType::I32 && operation == "clip_grad_value_")
        );
        Ok(())
    }

    // --- Tests for clip_grad_norm_ ---\n

    #[test]
    fn test_clip_grad_norm_basic_clipping_l2() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![3.0, 4.0]), true); // norm = 5.0
        let mut param2 = create_param_with_grad(&[1], DType::F32, Some(vec![12.0]), true);    // norm = 12.0
        // total_norm_pow_2 = 3^2 + 4^2 + 12^2 = 9 + 16 + 144 = 25 + 144 = 169
        // total_norm_l2 = sqrt(169) = 13.0

        let mut params_vec = vec![param1, param2];
        let max_norm = 6.5; // Moitié de 13.0
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;

        assert_relative_eq!(calculated_total_norm, 13.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6); // Environ 0.5

        let grad1_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        let grad2_data = params_vec[1].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;

        assert_relative_eq!(grad1_data[0], 3.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 4.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad2_data[0], 12.0 * clip_coef, epsilon = 1e-6);
        
        // Vérifier la nouvelle norme totale
        let mut new_total_norm_pow_p: f32 = 0.0;
        for val in &grad1_data { new_total_norm_pow_p += val.abs().powf(norm_type); }
        for val in &grad2_data { new_total_norm_pow_p += val.abs().powf(norm_type); }
        let new_total_norm = new_total_norm_pow_p.powf(1.0/norm_type);
        assert_relative_eq!(new_total_norm, max_norm, epsilon = 1e-5); // Epsilon un peu plus large à cause des calculs

        Ok(());
    }

    #[test]
    fn test_clip_grad_norm_no_clipping_needed_l2() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![1.0, 2.0]), true);
        // total_norm_l2 = sqrt(1^2 + 2^2) = sqrt(5) approx 2.236
        let mut params_vec = vec![param1];
        let max_norm = 3.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;
        
        assert_relative_eq!(calculated_total_norm, 5.0_f32.sqrt(), epsilon = 1e-6);

        let grad1_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 2.0, epsilon = 1e-6);
        Ok(());
    }

    #[test]
    fn test_clip_grad_norm_max_norm_zero() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![3.0, 4.0]), true);
        let mut params_vec = vec![param1];
        let max_norm = 0.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 5.0, epsilon = 1e-6);

        let grad1_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 0.0, epsilon = 1e-6);
        Ok(());
    }

    #[test]
    fn test_clip_grad_norm_l1_norm() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![-2.0, 3.0]), true);
        // L1 norm = |-2| + |3| = 5.0
        let mut params_vec = vec![param1];
        let max_norm = 2.5;
        let norm_type = 1.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 5.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6); // 2.5 / 5.0 = 0.5

        let grad1_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], -2.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 3.0 * clip_coef, epsilon = 1e-6);
        
        let mut new_total_norm_l1: f32 = 0.0;
        for val in &grad1_data { new_total_norm_l1 += val.abs(); }
        assert_relative_eq!(new_total_norm_l1, max_norm, epsilon = 1e-5);

        Ok(());
    }

    // Note: Testing with norm_type = f32::INFINITY (L_inf norm) is more complex due to how powf behaves with INFINITY.
    // The current implementation sums val.abs().powf(norm_type). If norm_type is Inf, this will likely lead to Inf or NaN quickly.
    // A specialized L_inf norm calculation would be `max(abs(g_i))`. Our generic formula isn't ideal for L_inf directly.
    // For now, we skip direct L_inf test with the current generic norm_type approach.

    #[test]
    fn test_clip_grad_norm_with_no_grad_param() -> Result<(), NeuraRustError> {
        let mut param_with_grad = create_param_with_grad(&[2], DType::F32, Some(vec![6.0, 8.0]), true); // norm 10
        let mut param_no_grad = create_param_with_grad(&[1], DType::F32, None, true);
        let mut params_vec = vec![param_with_grad, param_no_grad];
        let max_norm = 5.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 10.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6); // 0.5

        let grad1_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 6.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 8.0 * clip_coef, epsilon = 1e-6);

        assert!(params_vec[1].data.read().unwrap().grad.is_none());
        Ok(());
    }

    #[test]
    fn test_clip_grad_norm_all_grads_zero() -> Result<(), NeuraRustError> {
        let mut param1 = create_param_with_grad(&[2], DType::F32, Some(vec![0.0, 0.0]), true);
        let mut params_vec = vec![param1];
        let max_norm = 5.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 0.0, epsilon = 1e-6);

        // Grads should remain 0.0
        let grad1_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 0.0, epsilon = 1e-6);
        Ok(());
    }

    #[test]
    fn test_clip_grad_norm_mixed_dtypes_f32_f64() -> Result<(), NeuraRustError> {
        let mut param_f32 = create_param_with_grad(&[2], DType::F32, Some(vec![3.0, 4.0]), true); // norm_f32 = 5.0
        let mut param_f64 = create_param_with_grad(&[1], DType::F64, Some(vec![12.0]), true);    // norm_f64 = 12.0
        // total_norm_pow_2 = (3.0_f32).powi(2) + (4.0_f32).powi(2) + (12.0_f64 as f32).powi(2)
        // = 9.0 + 16.0 + 144.0 = 169.0
        // total_norm_l2 = sqrt(169.0) = 13.0 (as f32)

        let mut params_vec = vec![param_f32, param_f64];
        let max_norm = 6.5; 
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_vec.iter_mut(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 13.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6); // Approx 0.5

        let grad_f32_data = params_vec[0].data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad_f32_data[0], 3.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad_f32_data[1], 4.0 * clip_coef, epsilon = 1e-6);

        let grad_f64_data = params_vec[1].data.read().unwrap().grad.as_ref().unwrap().get_f64_data()?;
        assert_relative_eq!(grad_f64_data[0], 12.0 * (clip_coef as f64), epsilon = 1e-6); // Compare with f64 coeff
        Ok(());
    }

    #[test]
    fn test_clip_grad_norm_invalid_max_norm() {
        let mut param1 = create_param_with_grad(&[1], DType::F32, Some(vec![1.0]), true);
        let mut params_vec = vec![param1];
        let result = clip_grad_norm_(params_vec.iter_mut(), -1.0, 2.0);
        assert!(matches!(result, Err(NeuraRustError::ConfigurationError(msg)) if msg == "max_norm must be non-negative"));
    }

    #[test]
    fn test_clip_grad_norm_invalid_norm_type() {
        let mut param1 = create_param_with_grad(&[1], DType::F32, Some(vec![1.0]), true);
        let mut params_vec = vec![param1];
        let result_zero = clip_grad_norm_(params_vec.iter_mut(), 1.0, 0.0);
        assert!(matches!(result_zero, Err(NeuraRustError::ConfigurationError(msg)) if msg == "norm_type must be positive"));
        let result_neg = clip_grad_norm_(params_vec.iter_mut(), 1.0, -1.0);
        assert!(matches!(result_neg, Err(NeuraRustError::ConfigurationError(msg)) if msg == "norm_type must be positive"));
    }

    #[test]
    fn test_clip_grad_norm_non_float_grad_type_error() -> Result<(), NeuraRustError> {
        let mut tensor_for_param = Tensor::zeros(&[1], DType::I32, StorageDevice::Cpu);
        tensor_for_param.set_requires_grad(true);
        let grad_i32 = Tensor::from_vec_i32(vec![10], &[1], StorageDevice::Cpu);
        tensor_for_param.data.write().unwrap().grad = Some(grad_i32);
        
        let mut param_i32 = Parameter::new(tensor_for_param, Some("param_i32".to_string()));
        let mut params_vec = vec![param_i32];

        let result = clip_grad_norm_(params_vec.iter_mut(), 1.0, 2.0);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch { .. })));
        Ok(());
    }
} 