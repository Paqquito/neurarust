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
#[path = "linear_test.rs"]
mod tests;
 