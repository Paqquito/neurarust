#[cfg(test)]
mod tests {
    use crate::device::StorageDevice;
    use crate::nn::parameter::Parameter;
    use crate::optim::grad_clipping::{clip_grad_value_, clip_grad_norm_};
    use crate::tensor::Tensor;
    use crate::types::DType;
    use crate::NeuraRustError;
    use std::sync::{Arc, RwLock};
    use approx::assert_relative_eq;

    // Helper pour créer un Parameter enveloppé pour les tests
    fn create_arc_param_with_grad(
        shape: &[usize],
        dtype: DType,
        grad_data_opt: Option<Vec<f32>>,
        requires_grad: bool,
    ) -> Arc<RwLock<Parameter>> {
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
            tensor.data.write().unwrap().grad = Some(grad_tensor_val);
        }
        Arc::new(RwLock::new(Parameter::new(tensor, Some("test_param".to_string()))))
    }

    // Helper pour créer un Parameter I32 pour les tests d'erreur de type
    fn create_arc_param_i32_grad() -> Arc<RwLock<Parameter>> {
        let mut tensor_for_param = Tensor::zeros(&[1], DType::I32, StorageDevice::Cpu);
        tensor_for_param.set_requires_grad(true);
        let grad_i32 = Tensor::from_vec_i32(vec![10], &[1], StorageDevice::Cpu);
        tensor_for_param.data.write().unwrap().grad = Some(grad_i32);
        Arc::new(RwLock::new(Parameter::new(tensor_for_param, Some("param_i32".to_string()))))
    }

    #[test]
    fn test_clip_grad_value_f32() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2, 2], DType::F32, Some(vec![-5.0, 2.0, 3.0, -0.5]), true);
        let params_arcs = vec![param_arc1.clone()];

        clip_grad_value_(params_arcs.into_iter(), 1.0)?;

        let p_guard = param_arc1.read().unwrap();
        let td_guard = p_guard.tensor.data.read().unwrap();
        let grad_tensor = td_guard.grad.as_ref().unwrap();
        let grad_data = grad_tensor.get_f32_data()?;
        assert_eq!(grad_data, vec![-1.0, 1.0, 1.0, -0.5]);
        Ok(())
    }

    #[test]
    fn test_clip_grad_value_f64() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2, 2], DType::F64, Some(vec![-5.0, 2.0, 3.0, -0.5]), true);
        let params_arcs = vec![param_arc1.clone()];

        clip_grad_value_(params_arcs.into_iter(), 1.5)?;
        
        let p_guard = param_arc1.read().unwrap();
        let td_guard = p_guard.tensor.data.read().unwrap();
        let grad_tensor = td_guard.grad.as_ref().unwrap();
        let grad_data = grad_tensor.get_f64_data()?;
        assert_eq!(grad_data, vec![-1.5, 1.5, 1.5, -0.5]);
        Ok(())
    }

    #[test]
    fn test_clip_grad_value_no_grad() -> Result<(), NeuraRustError> {
        let param_no_grad_arc = create_arc_param_with_grad(&[1], DType::F32, None, true);
        let param_with_grad_arc = create_arc_param_with_grad(&[1], DType::F32, Some(vec![10.0]), true);
        let params_arcs = vec![param_no_grad_arc.clone(), param_with_grad_arc.clone()];

        clip_grad_value_(params_arcs.into_iter(), 1.0)?;

        assert!(param_no_grad_arc.read().unwrap().tensor.data.read().unwrap().grad.is_none());
        
        let p_guard_with = param_with_grad_arc.read().unwrap();
        let td_guard_with = p_guard_with.tensor.data.read().unwrap();
        assert_eq!(td_guard_with.grad.as_ref().unwrap().get_f32_data()?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn test_clip_grad_value_already_in_bounds() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![-0.5, 0.7]), true);
        let params_arcs = vec![param_arc1.clone()];

        clip_grad_value_(params_arcs.into_iter(), 1.0)?;
        
        let p_guard = param_arc1.read().unwrap();
        let td_guard = p_guard.tensor.data.read().unwrap();
        let grad_data = td_guard.grad.as_ref().unwrap().get_f32_data()?;
        assert_eq!(grad_data, vec![-0.5, 0.7]);
        Ok(())
    }
    
    #[test]
    fn test_clip_grad_value_negative_clip_value() {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![-0.5, 0.7]), true);
        let params_arcs = vec![param_arc1]; 

        let result = clip_grad_value_(params_arcs.into_iter(), -1.0);
        assert!(matches!(result, Err(NeuraRustError::ConfigurationError(msg)) if msg == "clip_value must be non-negative"));
    }

    #[test]
    fn test_clip_grad_value_non_float_grad_error_setup() -> Result<(), NeuraRustError> {
        let param_i32_arc = create_arc_param_i32_grad();
        let params_arcs = vec![param_i32_arc];

        let result = clip_grad_value_(params_arcs.into_iter(), 1.0);
        
        assert!(
            matches!(result, Err(NeuraRustError::DataTypeMismatch { expected, actual, operation }) 
                if expected == DType::F32 && actual == DType::I32 && operation == "clip_grad_value_")
        );
        Ok(())
    }

    // --- Tests for clip_grad_norm_ ---

    #[test]
    fn test_clip_grad_norm_basic_clipping_l2() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![3.0, 4.0]), true); 
        let param_arc2 = create_arc_param_with_grad(&[1], DType::F32, Some(vec![12.0]), true);   
        let params_arcs = vec![param_arc1.clone(), param_arc2.clone()];
        let max_norm = 6.5; 
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.clone().into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 13.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6); 

        let grad1_data = param_arc1.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        let grad2_data = param_arc2.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;

        assert_relative_eq!(grad1_data[0], 3.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 4.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad2_data[0], 12.0 * clip_coef, epsilon = 1e-6);
        
        let mut new_total_norm_pow_p: f32 = 0.0;
        for val in &grad1_data { new_total_norm_pow_p += val.abs().powf(norm_type); }
        for val in &grad2_data { new_total_norm_pow_p += val.abs().powf(norm_type); }
        let new_total_norm = new_total_norm_pow_p.powf(1.0/norm_type);
        assert_relative_eq!(new_total_norm, max_norm, epsilon = 1e-5); 
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_no_clipping_needed_l2() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![1.0, 2.0]), true);
        let params_arcs = vec![param_arc1.clone()];
        let max_norm = 3.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 5.0_f32.sqrt(), epsilon = 1e-6);

        let grad1_data = param_arc1.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 2.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_max_norm_zero() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![3.0, 4.0]), true);
        let params_arcs = vec![param_arc1.clone()];
        let max_norm = 0.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 5.0, epsilon = 1e-6);

        let grad1_data = param_arc1.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_l1_norm() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![-2.0, 3.0]), true);
        let params_arcs = vec![param_arc1.clone()];
        let max_norm = 2.5;
        let norm_type = 1.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.clone().into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 5.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6); 

        let grad1_data = param_arc1.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], -2.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 3.0 * clip_coef, epsilon = 1e-6);
        
        let mut new_total_norm_l1: f32 = 0.0;
        for val in &grad1_data { new_total_norm_l1 += val.abs(); }
        assert_relative_eq!(new_total_norm_l1, max_norm, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_with_no_grad_param() -> Result<(), NeuraRustError> {
        let param_with_grad_arc = create_arc_param_with_grad(&[2], DType::F32, Some(vec![6.0, 8.0]), true);
        let param_no_grad_arc = create_arc_param_with_grad(&[1], DType::F32, None, true);
        let params_arcs = vec![param_with_grad_arc.clone(), param_no_grad_arc.clone()];
        let max_norm = 5.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.clone().into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 10.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6);

        let grad1_data = param_with_grad_arc.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 6.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 8.0 * clip_coef, epsilon = 1e-6);

        assert!(param_no_grad_arc.read().unwrap().tensor.data.read().unwrap().grad.is_none());
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_all_grads_zero() -> Result<(), NeuraRustError> {
        let param_arc1 = create_arc_param_with_grad(&[2], DType::F32, Some(vec![0.0, 0.0]), true);
        let params_arcs = vec![param_arc1.clone()];
        let max_norm = 5.0;
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 0.0, epsilon = 1e-6);

        let grad1_data = param_arc1.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad1_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(grad1_data[1], 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_mixed_dtypes_f32_f64() -> Result<(), NeuraRustError> {
        let param_f32_arc = create_arc_param_with_grad(&[2], DType::F32, Some(vec![3.0, 4.0]), true);
        let param_f64_arc = create_arc_param_with_grad(&[1], DType::F64, Some(vec![12.0]), true);   
        let params_arcs = vec![param_f32_arc.clone(), param_f64_arc.clone()];
        let max_norm = 6.5; 
        let norm_type = 2.0;

        let calculated_total_norm = clip_grad_norm_(params_arcs.clone().into_iter(), max_norm, norm_type)?;
        assert_relative_eq!(calculated_total_norm, 13.0, epsilon = 1e-6);

        let clip_coef = max_norm / (calculated_total_norm + 1e-6);

        let grad_f32_data = param_f32_arc.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f32_data()?;
        assert_relative_eq!(grad_f32_data[0], 3.0 * clip_coef, epsilon = 1e-6);
        assert_relative_eq!(grad_f32_data[1], 4.0 * clip_coef, epsilon = 1e-6);

        let grad_f64_data = param_f64_arc.read().unwrap().tensor.data.read().unwrap().grad.as_ref().unwrap().get_f64_data()?;
        assert_relative_eq!(grad_f64_data[0], 12.0 * (clip_coef as f64), epsilon = 1e-6); 
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm_invalid_max_norm() {
        let param_arc1 = create_arc_param_with_grad(&[1], DType::F32, Some(vec![1.0]), true);
        let params_arcs = vec![param_arc1];
        let result = clip_grad_norm_(params_arcs.into_iter(), -1.0, 2.0);
        assert!(matches!(result, Err(NeuraRustError::ConfigurationError(msg)) if msg == "max_norm must be non-negative"));
    }

    #[test]
    fn test_clip_grad_norm_invalid_norm_type() {
        let param_arc1 = create_arc_param_with_grad(&[1], DType::F32, Some(vec![1.0]), true);
        let params_arcs_zero = vec![param_arc1.clone()];
        let result_zero = clip_grad_norm_(params_arcs_zero.into_iter(), 1.0, 0.0);
        assert!(matches!(result_zero, Err(NeuraRustError::ConfigurationError(msg)) if msg == "norm_type must be positive"));
        
        let params_arcs_neg = vec![param_arc1];
        let result_neg = clip_grad_norm_(params_arcs_neg.into_iter(), 1.0, -1.0);
        assert!(matches!(result_neg, Err(NeuraRustError::ConfigurationError(msg)) if msg == "norm_type must be positive"));
    }

    #[test]
    fn test_clip_grad_norm_non_float_grad_type_error() -> Result<(), NeuraRustError> {
        let param_i32_arc = create_arc_param_i32_grad();
        let params_arcs = vec![param_i32_arc];

        let result = clip_grad_norm_(params_arcs.into_iter(), 1.0, 2.0);
        assert!(matches!(result, Err(NeuraRustError::DataTypeMismatch { .. })));
        Ok(())
    }
} 