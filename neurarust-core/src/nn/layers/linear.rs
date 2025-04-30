use crate::tensor::Tensor;
use crate::nn::{Module, Parameter};
use crate::ops::linalg::matmul;
use crate::ops::arithmetic::add;
// Import necessary traits
use num_traits::{Zero, One}; 
use std::fmt::Debug;
use std::ops::{AddAssign, Add, Mul}; // Added missing ops traits
use std::iter::Sum;
use crate::error::NeuraRustError; // Keep only one import
// Remove external crate imports for now
// use rand::distributions::{Distribution, Uniform};
// use rand::SeedableRng; 
// use rand_chacha::ChaCha8Rng;

/// A fully connected linear layer: y = xA^T + b
#[derive(Debug, Clone)]
pub struct Linear<T> {
    #[allow(dead_code)]
    in_features: usize,
    #[allow(dead_code)]
    out_features: usize,
    pub weight: Parameter<T>,
    pub bias: Option<Parameter<T>>,
}

impl<T> Linear<T>
where
    // Simplified bounds for basic initialization (Zero/One)
    T: Zero + One + Clone + Debug + 'static,
{
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self, NeuraRustError> {
        // Simplified initialization (e.g., with ones) - Remove Kaiming init for now
        let weight_data = vec![T::one(); out_features * in_features];
        let weight_tensor = Tensor::new(weight_data, vec![out_features, in_features])?; 
        let weight = Parameter::new(weight_tensor);

        let bias_param = if bias {
            let bias_data = vec![T::zero(); out_features];
            let bias_tensor = Tensor::new(bias_data, vec![out_features])?;
            Some(Parameter::new(bias_tensor))
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias: bias_param,
            in_features,
            out_features,
        })
    }
}

impl<T> Module<T> for Linear<T>
where 
    // Update bounds based on actual usage below (transpose, matmul, add)
    T: Copy + Clone + Debug + 'static + AddAssign + One + Zero + PartialEq + Default + Sum + Add<Output=T> + Mul<Output=T>,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError> {
        let transposed_weight = self.weight.transpose();
        let mut output = matmul(input, &transposed_weight)?;
        
        if let Some(ref bias_param) = self.bias {
            let bias_tensor_ref = &**bias_param;
            let output_shape = output.shape(); 
            let bias_shape = bias_tensor_ref.shape();

             if output_shape.len() > 1 && bias_shape.len() == 1 && output_shape.last() == bias_shape.last() {
                 output = add(&output, bias_tensor_ref)?;
             } else if output_shape == bias_shape {
                 output = add(&output, bias_tensor_ref)?;
             } else {
                 return Err(NeuraRustError::IncompatibleShapes { 
                     shape1: output_shape.clone(), 
                     shape2: bias_shape.clone(),
                 });
             }
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::nn::{Module, Parameter}; // Ensure Parameter is imported
    // Remove approx import
    // use approx::assert_relative_eq;
    use crate::error::NeuraRustError;
    // Import traits needed for test helpers and asserts
    use num_traits::{Zero, One};
    use std::ops::AddAssign;
    use std::iter::Sum;
    use std::fmt::Debug;

    // Helper to create tensor (ensure bounds cover usage in tests)
    fn create_tensor<T: Clone + Debug + Default + PartialEq + Zero + One + AddAssign + Copy + Sum + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    
    // Helper to create tensor with grad
    fn create_grad_tensor<T: Clone + Debug + Default + PartialEq + Zero + One + AddAssign + Copy + Sum + 'static>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }

    #[test]
    fn test_linear_layer_creation() -> Result<(), NeuraRustError> {
        let linear_layer = Linear::<f32>::new(10, 5, true)?;
        assert_eq!(linear_layer.in_features, 10);
        assert_eq!(linear_layer.out_features, 5);
        assert_eq!(linear_layer.weight.shape(), vec![5, 10]);
        assert!(linear_layer.bias.is_some());
        assert_eq!(linear_layer.bias.as_ref().unwrap().shape(), vec![5]);
        assert_eq!(linear_layer.parameters().len(), 2);
        
        let linear_layer_no_bias = Linear::<f64>::new(3, 2, false)?;
        assert_eq!(linear_layer_no_bias.in_features, 3);
        assert_eq!(linear_layer_no_bias.out_features, 2);
        assert!(linear_layer_no_bias.bias.is_none());
        assert_eq!(linear_layer_no_bias.parameters().len(), 1);
        Ok(())
    }

    #[test]
    fn test_linear_forward_no_bias() -> Result<(), NeuraRustError> {
        let mut linear = Linear::<f32>::new(3, 2, false)?;
        let weights = create_tensor(vec![1.0; 6], vec![2, 3]);
        linear.weight = Parameter::new(weights);
        // Explicitly disable grad for the weight *after* creating the Parameter
        linear.weight.set_requires_grad(false);

        let input = create_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let output = linear.forward(&input)?; 

        assert_eq!(output.shape(), vec![1, 2]);
        assert_eq!(output.data()[0], 6.0);
        assert_eq!(output.data()[1], 6.0);
        assert!(!output.requires_grad());
        Ok(())
    }

    #[test]
    fn test_linear_forward_with_bias() -> Result<(), NeuraRustError> {
        let mut linear = Linear::<f32>::new(3, 2, true)?; 
        let weights = create_tensor(vec![1.0; 6], vec![2, 3]);
        linear.weight = Parameter::new(weights);
        
        let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let output = linear.forward(&input)?; 

        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.data()[0], 6.0);
        assert_eq!(output.data()[1], 6.0);
        assert_eq!(output.data()[2], 15.0);
        assert_eq!(output.data()[3], 15.0);
        Ok(())
    }
    
    #[test]
    fn test_linear_forward_grad_propagation() -> Result<(), NeuraRustError> {
        let linear = Linear::<f32>::new(3, 2, true)?;
        let input_grad = create_grad_tensor::<f32>(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let input_no_grad = create_tensor::<f32>(vec![1.0, 2.0, 3.0], vec![1, 3]);

        let output1 = linear.forward(&input_grad)?; 
        assert!(output1.requires_grad());
        assert!(output1.grad_fn().is_some());
        
        let output2 = linear.forward(&input_no_grad)?; 
        assert!(output2.requires_grad()); 
        assert!(output2.grad_fn().is_some());
        
        for param in linear.parameters() {
             param.set_requires_grad(false);
        }
        let output3 = linear.forward(&input_no_grad)?; 
        assert!(!output3.requires_grad());
        assert!(output3.grad_fn().is_none());
        Ok(())
    }

     // TODO: Add backward tests for Linear layer
}
 