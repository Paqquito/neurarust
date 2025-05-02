use crate::tensor::Tensor;
use crate::ops::linalg::matmul::matmul;
use crate::ops::arithmetic::add_op; 
// use crate::ops::linalg::transpose::transpose;
use crate::nn::Parameter; // Corrected import path
use crate::error::NeuraRustError;
use std::fmt::Debug;
use num_traits::{Float, Zero, One}; 
 // Removed Normal
use std::ops::{AddAssign, Mul, Neg};
use std::iter::Sum;

/// A fully connected linear layer: y = xA^T + b
#[derive(Debug, Clone)]
pub struct Linear<T> {
    #[allow(dead_code)]
    in_features: usize,
    #[allow(dead_code)]
    out_features: usize,
    pub weights: Parameter<T>,
    pub bias: Option<Parameter<T>>,
}

impl<T> Linear<T>
where
    T: Float + Debug + Copy + AddAssign + Mul<Output=T> + Default + Zero + One + Sum + PartialEq + Neg<Output=T> + 'static,
{
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self, NeuraRustError> {
        // Add underscore to unused variable
        let _name = format!("linear_{}_{}", in_features, out_features);

        // Placeholder for weight init
        let weight_data = vec![T::zero(); in_features * out_features];
        let weight_shape = vec![out_features, in_features]; // Shape [out, in]
        let weights = Parameter::new(Tensor::new(weight_data, weight_shape)?);

        let bias_param = if bias {
            let bias_data = vec![T::zero(); out_features];
            let bias_shape = vec![1, out_features]; // Shape [1, out]
            Some(Parameter::new(Tensor::new(bias_data, bias_shape)?))
        } else {
            None
        };

        Ok(Linear {
            weights,
            bias: bias_param,
            in_features,
            out_features,
        })
    }

    // Keep forward as an inherent method
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError> {
        let weight_tensor = &self.weights.0; 
        let transposed_weight = weight_tensor.transpose()?;
        let mut output = matmul(input, &transposed_weight)?;

        if let Some(bias_param) = &self.bias {
            let bias_tensor_ref = &bias_param.0; 
            let out_features = self.out_features;
            let expected_bias_shape = vec![1, out_features];
            
            if bias_tensor_ref.shape() != expected_bias_shape {
                if bias_tensor_ref.numel() == out_features {
                    let reshaped_bias = bias_tensor_ref.reshape(expected_bias_shape.clone());
                    output = add_op(&output, &reshaped_bias)?;
                } else {
                    return Err(NeuraRustError::ShapeMismatch { 
                        expected: expected_bias_shape, 
                        actual: bias_tensor_ref.shape().clone() 
                    });
                }
            } else {
                 output = add_op(&output, bias_tensor_ref)?;
            }
        }
        Ok(output)
    }

    // Keep parameters as an inherent method
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weights];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

// Assuming it should implement some trait, maybe Module or a new Layer trait?
// Let's implement forward and parameters as inherent methods for now.
// impl<T> Layer<T> for Linear<T> // Commented out trait impl
// where
//    T: Float + Debug + Copy + AddAssign + Mul<Output=T> + Default + Zero + One + Sum + PartialEq + Neg<Output=T> + 'static,
// {
//     fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError> {
//         let transposed_weight = transpose(&self.weights.tensor)?;
//         let mut output = matmul(input, &transposed_weight)?;
//
//         if let Some(bias_param) = &self.bias {
//             let bias_tensor_ref = &bias_param.tensor;
//             let out_features = self.out_features;
//             let expected_bias_shape = vec![1, out_features];
//             
//             if bias_tensor_ref.shape() != expected_bias_shape {
//                 if bias_tensor_ref.numel() == out_features {
//                     // Assuming reshape exists and works
//                     // TODO: Check reshape implementation for Result and Rc handling
//                     let reshaped_bias = bias_tensor_ref.reshape(&expected_bias_shape);
//                     output = add_op(&output, &reshaped_bias)?;
//                 } else {
//                     return Err(NeuraRustError::ShapeMismatch { 
//                         expected: expected_bias_shape, 
//                         actual: bias_tensor_ref.shape().clone() 
//                     });
//                 }
//             } else {
//                  output = add_op(&output, bias_tensor_ref)?;
//             }
//         }
//         Ok(output)
//     }
//
//     fn parameters(&self) -> Vec<&Parameter<T>> {
//         let mut params = vec![&self.weights];
//         if let Some(ref bias) = self.bias {
//             params.push(bias);
//         }
//         params
//     }
//
//     fn name(&self) -> &str {
//         &self.name
//     }
// }

// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::nn::Parameter; // Removed unused Module import
    use approx::assert_relative_eq;
    use crate::error::NeuraRustError;
    use num_traits::{Zero, One};
    use std::ops::AddAssign;
    use std::iter::Sum;
    use std::fmt::Debug;

    // Helper to create tensors for tests
    fn create_tensor<T: Clone + Debug + PartialEq + Zero + One + Copy + AddAssign + 'static + Sum + Default>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> {
        Tensor::new(data, shape).unwrap()
    }
    fn create_grad_tensor<T: Clone + Debug + PartialEq + Zero + One + Copy + AddAssign + 'static + Sum + Default>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> {
        Tensor::new_with_grad(data, shape).unwrap()
    }

    #[test]
    fn test_linear_layer_creation() -> Result<(), NeuraRustError> {
        let linear = Linear::<f32>::new(10, 5, true)?;
        assert_eq!(linear.in_features, 10);
        assert_eq!(linear.out_features, 5);
        assert_eq!(linear.weights.0.shape(), vec![5, 10]);
        assert!(linear.bias.is_some());
        assert_eq!(linear.bias.as_ref().unwrap().0.shape(), vec![1, 5]);
        assert_eq!(linear.parameters().len(), 2);
        
        let linear_no_bias = Linear::<f64>::new(3, 2, false)?;
        assert_eq!(linear_no_bias.in_features, 3);
        assert_eq!(linear_no_bias.out_features, 2);
        assert!(linear_no_bias.bias.is_none());
        assert_eq!(linear_no_bias.parameters().len(), 1);
        Ok(())
    }

    #[test]
    fn test_linear_forward_no_bias() -> Result<(), NeuraRustError> {
         let mut linear = Linear::<f32>::new(3, 2, false)?;
         // Set specific weights for predictable output
         let weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
         let weights = create_tensor(weights_data, vec![2, 3]);
         linear.weights = Parameter::new(weights);
         // Ensure requires_grad is false for this forward test
         linear.weights.0.set_requires_grad(false);

         let input_data = vec![1.0, 2.0, 3.0]; // shape [1, 3]
         let input = create_tensor(input_data, vec![1, 3]);

         let expected_output_data = vec![14.0, 32.0];
         let expected_output_shape = vec![1, 2];

         let output = linear.forward(&input)?;
         assert_eq!(output.shape(), expected_output_shape);
         let output_data = output.borrow_data_buffer();
         assert_relative_eq!(output_data[0], expected_output_data[0]);
         assert_relative_eq!(output_data[1], expected_output_data[1]);
         Ok(())
    }

    #[test]
    fn test_linear_forward_with_bias() -> Result<(), NeuraRustError> {
         let mut linear = Linear::<f32>::new(3, 2, true)?;
         let weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
         let weights = create_tensor(weights_data, vec![2, 3]);
         linear.weights = Parameter::new(weights);
         linear.weights.0.set_requires_grad(false); // Access via .0
 
         let bias_data = vec![0.5, -0.5]; // shape [1, 2]
         let bias = create_tensor(bias_data, vec![1, 2]);
         linear.bias = Some(Parameter::new(bias));
         linear.bias.as_ref().unwrap().0.set_requires_grad(false); // Access via .0
 
         let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
         let input = create_tensor(input_data, vec![2, 3]);
 
         let expected_output_data = vec![14.5, 31.5, 32.5, 76.5];
         // Add underscore to unused variable
         let _expected_output_shape = vec![2, 2]; 
 
         let output = linear.forward(&input)?;
         assert_eq!(output.shape(), vec![2, 2]);
         // Use borrow_data_buffer
         let output_data = output.borrow_data_buffer();
         assert_relative_eq!(output_data[0], expected_output_data[0]);
         assert_relative_eq!(output_data[1], expected_output_data[1]);
         assert_relative_eq!(output_data[2], expected_output_data[2]);
         assert_relative_eq!(output_data[3], expected_output_data[3]);
 
         // Test requires_grad propagation
         linear.weights.0.set_requires_grad(true); // Make weights require grad
         linear.bias.as_mut().unwrap().0.set_requires_grad(true); // Make bias require grad
         let input_no_grad = create_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
         let output_grad_params = linear.forward(&input_no_grad)?;
         assert!(output_grad_params.requires_grad()); // Should require grad if params do
         linear.weights.0.set_requires_grad(false);
         linear.bias.as_mut().unwrap().0.set_requires_grad(false);
 
         let input_grad = create_grad_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
         let output_grad_input = linear.forward(&input_grad)?;
         assert!(output_grad_input.requires_grad()); // Should require grad if input does
 
         Ok(())
    }

    #[test]
    fn test_linear_parameters() -> Result<(), NeuraRustError> {
        let linear_bias = Linear::<f32>::new(5, 2, true)?;
        assert_eq!(linear_bias.parameters().len(), 2);
        let linear_no_bias = Linear::<f32>::new(5, 2, false)?;
        assert_eq!(linear_no_bias.parameters().len(), 1);
        Ok(())
    }

     // TODO: Add backward tests for Linear layer
}
 