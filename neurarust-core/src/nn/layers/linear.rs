use crate::tensor::Tensor;
use crate::ops::linalg::matmul::matmul;
use crate::ops::arithmetic::add_op;
use crate::nn::Parameter;
use crate::error::NeuraRustError;
use crate::nn::module::Module;
use crate::nn::init::{kaiming_uniform_, zeros_};
use crate::types::{DType, NeuraNumeric};

use std::fmt::Debug;
use std::ops::{AddAssign, Mul, Neg};
use std::iter::Sum;
use crate::autograd::grad_check::{check_grad, GradCheckError};
use crate::ops::reduction::sum_op;

pub trait DefaultDType {
    fn default_dtype() -> DType;
}

impl DefaultDType for f32 {
    fn default_dtype() -> DType { DType::F32 }
}

impl DefaultDType for f64 {
    fn default_dtype() -> DType { DType::F64 }
}

/// A fully connected linear layer: y = xA^T + b
#[derive(Debug)]
pub struct Linear<T: NeuraNumeric + DefaultDType> {
    in_features: usize,
    out_features: usize,
    pub weights: Parameter<T>,
    pub bias: Option<Parameter<T>>,
}

impl<T> Linear<T>
where
    T: NeuraNumeric + DefaultDType + Copy + Debug + Default + AddAssign + Mul<Output=T> + Neg<Output=T> + Sum + PartialEq + 'static,
{
    pub fn new(in_features: usize, out_features: usize, bias_flag: bool) -> Result<Self, NeuraRustError> {
        let mut weights_tensor = Tensor::<T>::zeros(vec![out_features, in_features], T::default_dtype());
        kaiming_uniform_(&mut weights_tensor)?;
        let weights = Parameter::new(weights_tensor);

        let bias = if bias_flag {
            let mut bias_tensor = Tensor::<T>::zeros(vec![1, out_features], T::default_dtype());
            zeros_(&mut bias_tensor)?;
            Some(Parameter::new(bias_tensor))
        } else {
            None
        };

        Ok(Linear {
            weights,
            bias,
            in_features,
            out_features,
        })
    }
}

impl<T> Module<T> for Linear<T>
where
    T: NeuraNumeric + DefaultDType + Copy + Debug + Default + AddAssign + Mul<Output=T> + Neg<Output=T> + Sum + PartialEq + 'static,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError> {
        let weight_tensor = &self.weights;
        let transposed_weight = weight_tensor.transpose(1, 0)?;
        
        let mut output = matmul(input, &transposed_weight)?;

        if let Some(bias_param) = &self.bias {
            let bias_tensor = &bias_param;
            output = add_op(&output, bias_tensor)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<T>> {
        let mut params = Vec::new();
        params.push(self.weights.clone());
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::types::DType;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer_creation() -> Result<(), NeuraRustError> {
        let linear_f32 = Linear::<f32>::new(10, 5, true)?;
        assert_eq!(linear_f32.in_features, 10);
        assert_eq!(linear_f32.out_features, 5);
        assert_eq!(linear_f32.weights.shape(), &vec![5, 10]);
        assert!(linear_f32.weights.requires_grad());
        
        assert!(linear_f32.bias.is_some());
        let bias_f32 = linear_f32.bias.as_ref().unwrap();
        assert_eq!(bias_f32.shape(), &vec![1, 5]);
        assert!(bias_f32.requires_grad());
        
        let params_f32 = Module::<f32>::parameters(&linear_f32);
        assert_eq!(params_f32.len(), 2);

        let linear_f64_no_bias = Linear::<f64>::new(3, 2, false)?;
        assert_eq!(linear_f64_no_bias.in_features, 3);
        assert_eq!(linear_f64_no_bias.out_features, 2);
        assert!(linear_f64_no_bias.bias.is_none());
        assert_eq!(linear_f64_no_bias.weights.shape(), &vec![2, 3]);
        assert!(linear_f64_no_bias.weights.requires_grad());

        let params_f64 = Module::<f64>::parameters(&linear_f64_no_bias);
        assert_eq!(params_f64.len(), 1);
        Ok(())
    }

    #[test]
    fn test_linear_forward_f32_no_bias() -> Result<(), NeuraRustError> {
        let mut linear = Linear::<f32>::new(3, 2, false)?;
        
        let weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let new_weights_tensor = Tensor::<f32>::new(weights_data, vec![2, 3], DType::F32);
        linear.weights = Parameter::new(new_weights_tensor);

        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::<f32>::new(input_data, vec![1, 3], DType::F32);

        let expected_output_data = vec![14.0, 32.0];
        let output = Module::<f32>::forward(&linear, &input)?;

        assert_eq!(output.shape(), &vec![1, 2]);
        let output_data = output.get_f32_data().unwrap();
        assert_relative_eq!(output_data.as_slice(), expected_output_data.as_slice(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_forward_f64_with_bias() -> Result<(), NeuraRustError> {
        let mut linear = Linear::<f64>::new(3, 2, true)?;

        let weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let new_weights_tensor = Tensor::<f64>::new(weights_data, vec![2, 3], DType::F64);
        linear.weights = Parameter::new(new_weights_tensor);

        let bias_data = vec![0.5, -0.5];
        let new_bias_tensor = Tensor::<f64>::new(bias_data, vec![1, 2], DType::F64);
        linear.bias = Some(Parameter::new(new_bias_tensor));
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::<f64>::new(input_data, vec![2, 3], DType::F64);
        
        let expected_output_data = vec![14.5, 31.5, 32.5, 76.5];
        let output = Module::<f64>::forward(&linear, &input)?;

        assert_eq!(output.shape(), &vec![2, 2]);
        let output_data = output.get_f64_data().unwrap();
        assert_relative_eq!(output_data.as_slice(), expected_output_data.as_slice(), epsilon = 1e-9);

        let input_no_grad = Tensor::<f64>::new(vec![1.0, 2.0, 3.0], vec![1, 3], DType::F64);
        let output_grad_params = Module::<f64>::forward(&linear, &input_no_grad)?;
        assert!(output_grad_params.requires_grad(), "Output should require grad if params do");

        let mut input_grad = Tensor::<f64>::new(vec![1.0, 2.0, 3.0], vec![1, 3], DType::F64);
        input_grad.set_requires_grad(true);
        let old_weights_req_grad = linear.weights.requires_grad();
        let old_bias_req_grad = linear.bias.as_ref().map_or(false, |b| b.requires_grad());
        linear.weights.set_requires_grad(false);
        if let Some(b) = linear.bias.as_mut() { b.set_requires_grad(false); }
        
        let output_grad_input = Module::<f64>::forward(&linear, &input_grad)?;
        assert!(output_grad_input.requires_grad(), "Output should require grad if input does");

        linear.weights.set_requires_grad(old_weights_req_grad);
        if let Some(b) = linear.bias.as_mut() { b.set_requires_grad(old_bias_req_grad); }

        Ok(())
    }

    #[test]
    fn test_linear_parameters() -> Result<(), NeuraRustError> {
        let linear_with_bias = Linear::<f32>::new(5, 3, true)?;
        let params_with_bias = Module::<f32>::parameters(&linear_with_bias);
        assert_eq!(params_with_bias.len(), 2);
        assert_eq!(params_with_bias[0].shape(), &vec![3, 5]);
        assert_eq!(params_with_bias[1].shape(), &vec![1, 3]);

        let linear_no_bias = Linear::<f32>::new(5, 3, false)?;
        let params_no_bias = Module::<f32>::parameters(&linear_no_bias);
        assert_eq!(params_no_bias.len(), 1);
        assert_eq!(params_no_bias[0].shape(), &vec![3, 5]);
        Ok(())
    }

    fn linear_loss_fn<T>(output: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
    where
        T: NeuraNumeric + DefaultDType + Copy + Debug + Default + AddAssign + Mul<Output=T> + Neg<Output=T> + Sum + PartialEq + 'static,
    {
        sum_op(output, None, false)
    }

    #[test]
    fn test_linear_weights_grad_f32() -> Result<(), GradCheckError<f32>> {
        let mut linear = Linear::<f32>::new(3, 2, true).unwrap();
        linear.weights.set_requires_grad(true);
        if let Some(b) = linear.bias.as_mut() {
            b.set_requires_grad(true);
        }

        let input = Tensor::<f32>::randn(vec![4, 3], DType::F32);

        let func_for_weights = |weights_tensor: &Tensor<f32>| -> Result<Tensor<f32>, NeuraRustError> {
            let transposed_weight = weights_tensor.transpose(1, 0)?;
            let mut output = matmul(&input, &transposed_weight)?;

            if let Some(bias_param) = &linear.bias {
                let bias_tensor = &bias_param;
                output = add_op(&output, bias_tensor)?;
            }
            linear_loss_fn(&output)
        };
        
        check_grad(func_for_weights, &linear.weights, 1e-3, 1e-5)?;

        Ok(())
    }

    #[test]
    fn test_linear_bias_grad_f32() -> Result<(), GradCheckError<f32>> {
        let mut linear = Linear::<f32>::new(3, 2, true).unwrap();
        linear.weights.set_requires_grad(true);
        let bias_present = linear.bias.is_some();
        assert!(bias_present, "Bias must be present for this test");

        if let Some(b) = linear.bias.as_mut() {
            b.set_requires_grad(true);
        }
        
        let input = Tensor::<f32>::randn(vec![4, 3], DType::F32);

        let func_for_bias = |bias_tensor_arg: &Tensor<f32>| -> Result<Tensor<f32>, NeuraRustError> {
            let weight_tensor = &linear.weights;
            let transposed_weight = weight_tensor.transpose(1, 0)?;
            let mut output = matmul(&input, &transposed_weight)?;
            
            output = add_op(&output, bias_tensor_arg)?;
            linear_loss_fn(&output)
        };

        if let Some(ref bias_param) = linear.bias {
             check_grad(func_for_bias, bias_param, 1e-3, 1e-5)?;
        } else {
            panic!("Bias was expected for test_linear_bias_grad_f32");
        }
        Ok(())
    }

    #[test]
    fn test_linear_input_grad_f32() -> Result<(), GradCheckError<f32>> {
        let mut linear = Linear::<f32>::new(3, 2, true).unwrap();
        let mut input = Tensor::<f32>::randn(vec![4, 3], DType::F32);
        input.set_requires_grad(true);

        let func_for_input = |current_input: &Tensor<f32>| -> Result<Tensor<f32>, NeuraRustError> {
            let weight_tensor = &linear.weights;
            let transposed_weight = weight_tensor.transpose(1, 0)?;
            let mut output = matmul(current_input, &transposed_weight)?;

            if let Some(bias_param) = &linear.bias {
                output = add_op(&output, bias_param)?;
            }
            linear_loss_fn(&output)
        };

        check_grad(func_for_input, &input, 1e-3, 1e-5)?;

        Ok(())
    }
}
 