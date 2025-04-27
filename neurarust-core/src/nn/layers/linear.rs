use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use std::fmt::{self, Debug};
use num_traits::{Zero, One}; // Need Zero/One for data initialization and ops
use std::ops::{Add, Mul, AddAssign, Neg}; // Ops for forward/backward
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::marker::PhantomData;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;

// Placeholder for random initialization
fn simple_uniform_init<T: Zero + One + Copy>(rows: usize, cols: usize) -> Vec<T> {
    // Temporary: Fill with 0.1 for reproducibility
    let val = T::one() / (T::one() + T::one() + T::one() + T::one() + T::one() +
                           T::one() + T::one() + T::one() + T::one() + T::one());
    vec![val; rows * cols]
}
    
    
/// Applies a linear transformation to the incoming data: y = xA^T + b
#[derive(Debug)] // Automatically derive Debug
pub struct Linear<T> {
    // Store weight and bias as Parameters
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    // Store feature counts for clarity
    in_features: usize,
    out_features: usize,
}

impl<T> Linear<T>
where 
    T: Zero + One + Copy + Clone + 'static + Debug + PartialEq 
       + AddAssign + Mul<Output=T> + Add<Output=T> // Ops needed for matmul/add
       + Neg<Output=T> // Needed for Add bound for &Tensor + &Tensor
{
    /// Creates a new Linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Size of each input sample.
    /// * `out_features` - Size of each output sample.
    /// * `has_bias` - If `true`, the layer will learn an additive bias.
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        // Weight tensor shape: [out_features, in_features]
        let weight_data = simple_uniform_init(out_features, in_features);
        let weight_tensor = Tensor::new(weight_data, vec![out_features, in_features]);
        let weight = Parameter::new(weight_tensor); // Wrap in Parameter
        
        // Bias tensor shape: [out_features]
        let bias_param = if has_bias {
            let bias_data = vec![T::zero(); out_features];
            let bias_tensor = Tensor::new(bias_data, vec![out_features]);
            Some(Parameter::new(bias_tensor)) // Wrap in Parameter
        } else {
            None
        };
        Linear {
            weight,
            bias: bias_param,
            in_features,
            out_features,
        }
    }
}


// --- Backward Operation for Linear Layer --- 

struct LinearBackward<T> {
    input: Tensor<T>, // Clone of the input tensor from the forward pass
    weight: Parameter<T>, // Clone of the weight parameter
    bias: Option<Parameter<T>>, // Clone of the bias parameter (if it exists)
    input_ref: Weak<RefCell<TensorData<T>>>, // Weak reference to the input's data
    weight_ref: Weak<RefCell<TensorData<T>>>, // Weak reference to the weight's data
    bias_ref: Option<Weak<RefCell<TensorData<T>>>>, // Weak reference to the bias's data
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for LinearBackward<T>
where
    T: Copy + Clone + 'static + Debug + PartialEq + AddAssign + Mul<Output = T> + Add<Output = T> + Zero + One + Neg<Output = T>,
{
    fn backward(&self, upstream_grad: &Tensor<T>) {
        // Calculate gradient with respect to the input: grad_input = upstream_grad @ weight
        // Note: We use the original weight matrix, not its transpose used in forward
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                let grad_input = upstream_grad.matmul(&self.weight);
                let mut input_td = input_rc.borrow_mut();
                if let Some(ref mut grad) = input_td.grad {
                    *grad += &grad_input;
                } else {
                    input_td.grad = Some(grad_input);
                }
            }
        }

        // Calculate gradient with respect to the weight: grad_weight = input.T @ upstream_grad
        if let Some(weight_rc) = self.weight_ref.upgrade() {
             // Parameters always require grad, no need to check input_rc.borrow().requires_grad
             let input_transposed = self.input.transpose();
             let grad_weight = input_transposed.matmul(upstream_grad);
             let mut weight_td = weight_rc.borrow_mut();
             if let Some(ref mut grad) = weight_td.grad {
                 *grad += &grad_weight;
             } else {
                 weight_td.grad = Some(grad_weight);
             }
        }

        // Calculate gradient with respect to the bias: grad_bias = upstream_grad.sum(axis=0) (or appropriate dims)
        if let Some(ref bias_param) = self.bias {
            if let Some(bias_rc) = self.bias_ref.as_ref().unwrap().upgrade() {
                // Parameters always require grad
                // Summing the upstream gradient across the batch dimension(s)
                // If upstream_grad is [B, N] and bias is [N], sum along axis 0.
                // If upstream_grad is [B, C, N] and bias is [N], sum along axes 0 and 1.
                let mut axes_to_sum = Vec::new();
                let upstream_rank = upstream_grad.shape().len();
                let bias_rank = bias_param.shape().len(); // Should be 1
                if upstream_rank > bias_rank {
                    for i in 0..(upstream_rank - bias_rank) {
                        axes_to_sum.push(i);
                    }
                }
                
                let grad_bias = if !axes_to_sum.is_empty() {
                    upstream_grad.sum_keep_dims(&axes_to_sum)
                } else {
                    // If ranks match (e.g., both are 1D), clone directly
                    upstream_grad.clone()
                }; 
                
                // Reshape grad_bias if needed after sum_keep_dims (e.g., from [1, N] to [N])
                let final_grad_bias = grad_bias.reshape(bias_param.shape());
                
                let mut bias_td = bias_rc.borrow_mut();
                if let Some(ref mut grad) = bias_td.grad {
                    *grad += &final_grad_bias;
                } else {
                    bias_td.grad = Some(final_grad_bias);
                }
            }
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        let mut inputs = Vec::with_capacity(3);
        inputs.push(self.input_ref.clone());
        inputs.push(self.weight_ref.clone());
        if let Some(ref bias_ref) = self.bias_ref {
            inputs.push(bias_ref.clone());
        }
        inputs
    }
}


impl<T> Module<T> for Linear<T>
where
    // Bounds needed for forward AND backward (matmul, transpose, add, sum, etc.)
    T: Zero + One + Copy + Clone + 'static + Debug + PartialEq 
       + Add<Output=T> + Mul<Output=T> + AddAssign 
       + Neg<Output=T>,
{
    fn forward(&self, input: &Tensor<T>) -> Tensor<T> {
        let input_shape = input.shape();
        assert!(input_shape.len() >= 1, "Input tensor must have at least one dimension.");
        assert_eq!(input_shape[input_shape.len() - 1], self.in_features, "Input feature dimension mismatch");

        // Perform matrix multiplication: input @ weight.T
        // Use clone of weight parameter for potential backward pass
        let weight_for_matmul = self.weight.clone(); 
        let output = input.matmul(&weight_for_matmul.transpose());
        
        let final_output = if let Some(ref bias_param) = self.bias {
             // --- Bias Addition Logic (modified slightly to return owned tensor) ---
             let output_shape = output.shape();
             let bias_shape = bias_param.shape(); // Bias is [out_features]
             
             if output_shape.len() > 1 && bias_shape.len() == 1 && output_shape[output_shape.len()-1] == bias_shape[0] {
                 let batch_size = output_shape[0..output_shape.len()-1].iter().product::<usize>();
                 if batch_size > 1 {
                     let bias_data = bias_param.data();
                     let mut broadcasted_bias_data = Vec::with_capacity(batch_size * self.out_features);
                     for _ in 0..batch_size {
                         broadcasted_bias_data.extend_from_slice(&bias_data);
                     }
                     let broadcasted_bias = Tensor::new(broadcasted_bias_data, output_shape.clone());
                      // Addition returns an owned Tensor
                     &output + &broadcasted_bias 
                 } else {
                     &output + bias_param
                 }
             } else if output_shape == bias_shape {
                  &output + bias_param
             } else {
                 panic!("Cannot broadcast bias shape {:?} to output shape {:?}", bias_shape, output_shape);
             }
        } else {
             output // Return the matmul result directly if no bias
        };

        // Set up backward pass if needed
        let requires_grad = input.requires_grad() || self.weight.requires_grad() || self.bias.as_ref().map_or(false, |b| b.requires_grad());
        if requires_grad {
            final_output.set_requires_grad(true);
            let grad_fn = LinearBackward {
                input: input.clone(), // Clone input for backward
                weight: self.weight.clone(), // Clone weight parameter
                bias: self.bias.clone(), // Clone bias parameter
                input_ref: input.get_weak_ref(),
                weight_ref: self.weight.get_weak_ref(),
                bias_ref: self.bias.as_ref().map(|p| p.get_weak_ref()),
                _phantom: PhantomData,
            };
            final_output.0.borrow_mut().grad_fn = Some(Rc::new(grad_fn));
        }

        final_output
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        let mut params = Vec::with_capacity(2);
        params.push(self.weight.0.clone()); // Clone inner tensor
        if let Some(ref bias_param) = self.bias {
            params.push(bias_param.0.clone()); // Clone inner tensor
        }
        params
    }
}
    
    
// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*; // Import Linear, etc.
    use crate::tensor::Tensor;
    use crate::utils::testing::check_tensor_near; // Assuming a helper for float comparison exists
    
    // Helper to check tensor properties
    fn check_tensor<T: PartialEq + Debug>(tensor: &Tensor<T>, expected_shape: &[usize], requires_grad: bool) {
        assert_eq!(tensor.shape(), expected_shape, "Shape mismatch");
        assert_eq!(tensor.requires_grad(), requires_grad, "requires_grad mismatch");
    }

    #[test]
    fn test_linear_creation() {
        let linear = Linear::<f32>::new(10, 5, true); // With bias
        check_tensor(&linear.weight, &[5, 10], true);
        assert!(linear.bias.is_some());
        let bias_tensor = linear.bias.as_ref().unwrap();
        check_tensor(&bias_tensor.0, &[5], true); // Check inner tensor
        assert_eq!(linear.in_features, 10);
        assert_eq!(linear.out_features, 5);
        
        let linear_no_bias = Linear::<f32>::new(20, 30, false); // Without bias
        check_tensor(&linear_no_bias.weight, &[30, 20], true);
        assert!(linear_no_bias.bias.is_none());
    }
    
    #[test]
    fn test_linear_parameters() {
        let linear = Linear::<f32>::new(3, 2, true);
        let params = linear.parameters();
        assert_eq!(params.len(), 2);
        check_tensor(&params[0], &[2, 3], true); // Weight
        check_tensor(&params[1], &[2], true);    // Bias
        
        let linear_no_bias = Linear::<f32>::new(5, 4, false);
        let params_no_bias = linear_no_bias.parameters();
        assert_eq!(params_no_bias.len(), 1);
        check_tensor(&params_no_bias[0], &[4, 5], true); // Weight only
    }
    
    #[test]
    fn test_linear_forward_no_bias() {
        let mut linear = Linear::<f32>::new(3, 2, false);
        // Manually set weights - access inner tensor of Parameter
        linear.weight.0.borrow_tensor_data_mut().data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let output = linear.forward(&input);
        
        // Expected: input @ weight.T = [10, 20, 30] @ [[1, 4], [2, 5], [3, 6]]
        // = [10*1+20*2+30*3, 10*4+20*5+30*6] = [140, 320]
        let expected_output_data = vec![140.0_f32, 320.0];
        let expected_output_shape = vec![1, 2];
        
        check_tensor(&output, &expected_output_shape, true); // Requires grad because weight does
        assert_eq!(output.data(), expected_output_data);
    }
    
    #[test]
    fn test_linear_forward_with_bias() {
        let mut linear = Linear::<f32>::new(3, 2, true);
        linear.weight.0.borrow_tensor_data_mut().data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        linear.bias.as_mut().unwrap().0.borrow_tensor_data_mut().data = vec![0.1, 0.2];
        let input = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let output = linear.forward(&input);

        // Expected: [140, 320] + [0.1, 0.2] = [140.1, 320.2]
        let expected_output_data = vec![140.1_f32, 320.2];
        let expected_output_shape = vec![1, 2];
        
        check_tensor(&output, &expected_output_shape, true); // Requires grad from params
        assert_eq!(output.data(), expected_output_data);
    }
    
    #[test]
    fn test_linear_forward_with_bias_batch() {
        let mut linear = Linear::<f32>::new(3, 2, true);
        linear.weight.0.borrow_tensor_data_mut().data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // Simple weight [[1,0,0],[0,1,0]]
        linear.bias.as_mut().unwrap().0.borrow_tensor_data_mut().data = vec![0.1, 0.2]; // Bias [0.1, 0.2]
        
        // Input shape [2, 3]
        let input = Tensor::new(vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let output = linear.forward(&input);
        
        // Expected: input @ weight.T + bias
        // weight.T = [[1,0],[0,1],[0,0]]
        // matmul = [[10, 20], [1, 2]] -> Shape [2, 2]
        // Add broadcasted bias [[0.1, 0.2], [0.1, 0.2]]
        // = [[10.1, 20.2], [1.1, 2.2]]
        let expected_output_data = vec![10.1_f32, 20.2, 1.1, 2.2];
        let expected_output_shape = vec![2, 2];
        
        check_tensor(&output, &expected_output_shape, true);
        assert_eq!(output.data(), expected_output_data);
    }
    
    #[test]
    fn test_linear_backward_simple() {
        // Input: [1, 2], Weight: [[3], [4]] (2x1), Bias: [0.1] (1x1)
        let mut linear = Linear::<f32>::new(2, 1, true);
        linear.weight.0.borrow_tensor_data_mut().data = vec![3.0, 4.0]; // Shape [1, 2]
        linear.bias.as_mut().unwrap().0.borrow_tensor_data_mut().data = vec![0.1]; // Shape [1]
        
        let input = Tensor::new(vec![10.0, 20.0], vec![1, 2]);

        // Forward pass: 
        // input @ weight.T + bias
        // [10, 20] @ [[3], [4]] + [0.1]
        // [10*3 + 20*4] + [0.1] = [110] + [0.1] = [110.1]
        let output = linear.forward(&input);
        check_tensor_near(&output, &[1, 1], true, &vec![110.1], 1e-6);
        assert!(output.grad_fn().is_some());

        // Backward pass
        output.backward(); // dOutput/dOutput = [[1.0]]

        // --- Check Gradients ---

        // Expected grad_input = upstream_grad @ weight
        // = [[1.0]] @ [[3.0, 4.0]] = [[3.0, 4.0]]
        let grad_input = input.grad().unwrap();
        check_tensor_near(&grad_input, &[1, 2], true, &vec![3.0, 4.0], 1e-6);
        
        // Expected grad_weight = input.T @ upstream_grad
        // = [[10], [20]] @ [[1.0]] = [[10.0], [20.0]] -> shape [2, 1] -> reshaped to weight's [1, 2]? No, grad shape matches weight shape.
        // weight shape is [1, 2]. grad_weight should be [1, 2]
        // input.T shape [2, 1], upstream_grad shape [1, 1]
        // matmul result shape [2, 1]. Wait, this doesn't match weight shape [1, 2].
        // Let's re-derive: dL/dW = dL/dY * dY/dW = upstream_grad * d(XW^T+b)/dW
        // d(XW^T)/dW_ij = ? This is complex. Let Y = XW^T. dY/dW_kl = X_k * delta_jl ? 
        // Let's use the formula: grad_W = X.T @ grad_Y
        // Input X: [1, 2], Weight W: [1, 2], Output Y: [1, 1]
        // Upstream Grad dY: [1, 1]
        // Grad W: X.T @ dY = [2, 1] @ [1, 1] = [2, 1]
        // This still doesn't match W's shape [1, 2]. Something is wrong.
        // Ah, the forward uses W.T! Forward: Y = X @ W.T + b
        // Weight W has shape [out, in] = [1, 2]
        // W.T has shape [in, out] = [2, 1]
        // Input X has shape [batch, in] = [1, 2]
        // Y = X @ W.T => [1, 2] @ [2, 1] = [1, 1]
        // Now gradients:
        // grad_X = dY @ W = [1, 1] @ [1, 2] = [1, 2] (Matches input shape) - CORRECT
        // grad_W.T = X.T @ dY = [2, 1] @ [1, 1] = [2, 1] (Matches W.T shape)
        // So grad_W = (X.T @ dY).T = [1, 2] (Matches W shape) - CORRECT
        // X.T = [[10], [20]]
        // dY = [[1.0]]
        // X.T @ dY = [[10], [20]]
        // (X.T @ dY).T = [[10, 20]]
        let grad_weight = linear.weight.grad().unwrap();
        check_tensor_near(&grad_weight, &[1, 2], true, &vec![10.0, 20.0], 1e-6);

        // Expected grad_bias = upstream_grad.sum(axes) = [[1.0]].sum() = [[1.0]] -> shape [1] after reshape
        let grad_bias = linear.bias.as_ref().unwrap().grad().unwrap();
         check_tensor_near(&grad_bias, &[1], true, &vec![1.0], 1e-6);
    }

    #[test]
    fn test_linear_backward_batch() {
        // Input: [2, 3], Weight: [4, 3], Bias: [4]
        let mut linear = Linear::<f32>::new(3, 4, true);
        // Set predictable weights/bias
        let w_data = (1..=12).map(|x| x as f32 * 0.1).collect::<Vec<_>>(); // [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...] shape [4, 3]
        let b_data = (1..=4).map(|x| x as f32 * 0.01).collect::<Vec<_>>(); // [0.01, 0.02, 0.03, 0.04] shape [4]
        linear.weight.0.borrow_tensor_data_mut().data = w_data;
        linear.bias.as_mut().unwrap().0.borrow_tensor_data_mut().data = b_data;
        
        let input_data = (1..=6).map(|x| x as f32).collect::<Vec<_>>(); // [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] shape [2, 3]
        let input = Tensor::new(input_data, vec![2, 3]);

        // Forward pass (Calculations omitted for brevity)
        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[2, 4]);
        assert!(output.requires_grad());

        // Use sum as loss for backward
        let loss = output.sum();
        loss.backward(); // Upstream grad is [[1, 1, 1, 1], [1, 1, 1, 1]] essentially before matmul

        // --- Check Gradients (Shapes and existence) ---
        assert!(input.grad().is_some());
        assert_eq!(input.grad().unwrap().shape(), &[2, 3]);

        assert!(linear.weight.grad().is_some());
        assert_eq!(linear.weight.grad().unwrap().shape(), &[4, 3]);
        
        assert!(linear.bias.is_some());
        assert!(linear.bias.as_ref().unwrap().grad().is_some());
        assert_eq!(linear.bias.as_ref().unwrap().grad().unwrap().shape(), &[4]);

        // TODO: Add numerical checks for gradient values if needed (complex to calculate manually)
    }
}
 