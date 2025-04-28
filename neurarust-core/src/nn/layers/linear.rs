use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use std::fmt::{Debug};
use num_traits::{Zero, One}; // Need Zero/One for data initialization and ops
use std::ops::{Add, Mul, AddAssign, Neg, Deref, Sub}; // Ops for forward/backward - Added Deref back
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::marker::PhantomData;
use crate::autograd::BackwardOp;
use crate::tensor_data::TensorData;
use std::iter::Sum as IterSum; // Added IterSum import
use std::collections::HashMap;

// Placeholder for random initialization
fn simple_uniform_init<T>(rows: usize, cols: usize) -> Vec<T> 
where
    T: Zero + Copy, // Removed One, Add, Div constraints - only Zero is needed now
{
    // Temporary: Fill with Zero for simplicity and to avoid Div constraint issues.
    // TODO: Implement proper random initialization (e.g., Kaiming, Xavier).
    vec![T::zero(); rows * cols]
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
       + AddAssign + Mul<Output=T> + Add<Output=T> 
       + Neg<Output=T> + Sub<Output=T> + IterSum, // Added missing bounds (Sub, Sum, Add was already there)
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

#[derive(Debug)] // Ajout de derive(Debug)
struct LinearBackward<T> {
    input: Tensor<T>,
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    input_ref: Weak<RefCell<TensorData<T>>>,
    weight_ref: Weak<RefCell<TensorData<T>>>,
    bias_ref: Option<Weak<RefCell<TensorData<T>>>>,
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for LinearBackward<T>
where
    T: Add<Output = T> + Mul<Output = T> + AddAssign + Neg<Output = T> + Sub<Output = T> + 
       Zero + One + Copy + Clone + 'static + Debug + PartialEq + IterSum,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        // Cloner et remodeler si scalaire [1] -> [1, 1] pour matmul
        let grad_clone = if upstream_grad.shape() == [1] {
            upstream_grad.clone().reshape(vec![1, 1])
        } else {
            upstream_grad.clone()
        };
        // grad_clone.set_requires_grad(false); // Déjà supprimé

        // --- 1. Calculate all local gradients FIRST ---
        
        // Calculate grad_input (if needed)
        let maybe_grad_input = if self.input_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad) {
            let grad_input = grad_clone.matmul(&self.weight);
            grad_input.set_requires_grad(false);
            Some(grad_input)
        } else {
            None
        };
        
        // Calculate grad_weight (always needed for parameter)
        let input_transposed = self.input.transpose();
        let grad_weight_t = input_transposed.matmul(&grad_clone);
        let grad_weight = grad_weight_t.transpose(); 
        grad_weight.set_requires_grad(false);

        // Calculate grad_bias (if bias exists)
        let maybe_grad_bias = if let Some(ref bias_param) = self.bias {
             // Check requires_grad on bias parameter itself
             if bias_param.requires_grad() { // Assumes Parameter has requires_grad()
                 let upstream_rank = grad_clone.shape().len();
                 let bias_rank = bias_param.shape().len();
                 
                 let grad_bias = if upstream_rank > bias_rank {
                     let axes_to_sum: Vec<usize> = (0..(upstream_rank - bias_rank)).collect();
                     grad_clone.sum_axes(&axes_to_sum, false)
                 } else {
                     grad_clone.clone() // Clone needed if grad_clone is used later
                 };
                 grad_bias.set_requires_grad(false); 
                 Some(grad_bias)
            } else {
                 None
            }
        } else {
            None
        };

        // --- 2. Accumulate gradients (take mutable borrows sequentially) ---
        
        // Accumulate grad_input
        if let Some(grad_input) = maybe_grad_input {
            if let Some(input_rc) = self.input_ref.upgrade() {
                let mut input_td = input_rc.borrow_mut();
                if input_td.requires_grad {
                    if let Some(existing_grad_tensor) = input_td.grad.as_mut() {
                        // Accumulate data directly
                        let mut existing_grad_data = existing_grad_tensor.borrow_tensor_data_mut();
                        let grad_input_data = grad_input.borrow_tensor_data();
                        assert_eq!(existing_grad_data.data.len(), grad_input_data.data.len());
                        for (existing, new) in existing_grad_data.data.iter_mut().zip(grad_input_data.data.iter()) {
                            *existing += *new;
                        }
                    } else {
                        input_td.grad = Some(grad_input);
                    }
                }
            }
        }

        // Accumulate grad_weight
        if let Some(weight_rc) = self.weight_ref.upgrade() {
             let mut weight_td = weight_rc.borrow_mut();
              // Weight always requires grad if we are in backward
             if let Some(existing_grad_tensor) = weight_td.grad.as_mut() {
                 // Accumulate data directly
                 let mut existing_grad_data = existing_grad_tensor.borrow_tensor_data_mut();
                 let grad_weight_data = grad_weight.borrow_tensor_data();
                 assert_eq!(existing_grad_data.data.len(), grad_weight_data.data.len());
                 for (existing, new) in existing_grad_data.data.iter_mut().zip(grad_weight_data.data.iter()) {
                     *existing += *new;
                 }
             } else {
                 weight_td.grad = Some(grad_weight);
             }
         }

        // Accumulate grad_bias
        if let Some(grad_bias) = maybe_grad_bias {
            // Make sure bias_ref exists before unwrapping
            if let Some(bias_ref_weak) = self.bias_ref.as_ref() {
                 if let Some(bias_rc) = bias_ref_weak.upgrade() {
                    let mut bias_td = bias_rc.borrow_mut(); 
                    // Bias requires grad (checked when creating grad_bias)
                    if let Some(existing_grad_tensor) = bias_td.grad.as_mut() {
                         // Accumulate data directly
                        let mut existing_grad_data = existing_grad_tensor.borrow_tensor_data_mut();
                        let grad_bias_data = grad_bias.borrow_tensor_data();
                         assert_eq!(existing_grad_data.data.len(), grad_bias_data.data.len());
                        for (existing, new) in existing_grad_data.data.iter_mut().zip(grad_bias_data.data.iter()) {
                            *existing += *new;
                        }
                    } else {
                        bias_td.grad = Some(grad_bias);
                    }
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
       + Neg<Output=T> + Sub<Output=T> + IterSum + Default, // Added Default
{
    fn forward(&self, input: &Tensor<T>) -> Tensor<T> 
        where T: Add<Output=T>
    {
        let input_shape = input.shape();
        assert!(input_shape.len() >= 1, "Input tensor must have at least one dimension.");
        assert_eq!(input_shape[input_shape.len() - 1], self.in_features, "Input feature dimension mismatch");

        // Perform matrix multiplication: input @ weight.T
        // Use clone of weight parameter for potential backward pass
        let weight_for_matmul = self.weight.clone(); 
        let output = input.matmul(&weight_for_matmul.transpose());
        
        let final_output = if let Some(ref bias_param) = self.bias {
             // --- Bias Addition Logic --- 
             let output_shape = output.shape();
             let bias_shape = bias_param.shape(); // Bias is [out_features] == [N]
             
             // Cas 1: Output est [B, ..., N] et Bias est [N]
             if output_shape.len() > 1 && bias_shape.len() == 1 && output_shape[output_shape.len()-1] == bias_shape[0] {
                 // Toujours broadcaster le biais à la forme de l'output dans ce cas
                 let batch_size = output_shape[0..output_shape.len()-1].iter().product::<usize>();
                 let bias_data = bias_param.data();
                 let mut broadcasted_bias_data = Vec::with_capacity(batch_size * self.out_features);
                 for _ in 0..batch_size {
                     broadcasted_bias_data.extend_from_slice(&bias_data);
                 }
                 let broadcasted_bias = Tensor::new(broadcasted_bias_data, output_shape.clone());
                 // On suppose que broadcasted_bias a requires_grad = true si bias_param l'a
                 // et que l'op Add propage requires_grad correctement.
                 if bias_param.requires_grad() { broadcasted_bias.set_requires_grad(true); }
                 
                 &output + &broadcasted_bias 
             
             // Cas 2: Output et Bias ont exactement la même forme (e.g., input était 1D, output est [N])
             } else if output_shape == bias_shape {
                  // Déréférencer Parameter vers &Tensor<T> pour l'addition
                  &output + &*bias_param 
             } else {
                 panic!("Cannot broadcast bias shape {:?} to output shape {:?}", bias_shape, output_shape);
             }
        } else {
             output // Return the matmul result directly if no bias
        };

        // Set up backward pass if needed
        let requires_grad = input.requires_grad() || self.weight.requires_grad() || self.bias.as_ref().map_or(false, |b| b.requires_grad());
        if requires_grad {
            final_output.data.borrow_mut().grad_fn = Some(Rc::new(LinearBackward {
                input: input.clone(), // Clone input for backward
                weight: self.weight.clone(), // Clone weight parameter
                bias: self.bias.clone(), // Clone bias parameter
                input_ref: input.get_weak_ref(),
                weight_ref: self.weight.get_weak_ref(),
                bias_ref: self.bias.as_ref().map(|p| p.get_weak_ref()),
                _phantom: PhantomData,
            }));
        }

        final_output
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        let mut params = Vec::with_capacity(2);
        // Clone the Tensor itself using Deref on Parameter
        params.push(self.weight.deref().clone()); 
        if let Some(ref bias_param) = self.bias {
            // Clone the Tensor itself using Deref on Parameter
            params.push(bias_param.deref().clone()); 
        }
        params
    }
}
    
    
// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*; // Import Linear, etc.
    use crate::tensor::Tensor;
    // use crate::utils::testing::check_tensor_near; // Assuming a helper for float comparison exists - Uncomment if used
     // Cleaned up ops
     // Keep Zero/One needed for Linear::new and potential ops
    use std::fmt::Debug; // Keep Debug
     // For test helpers
        // For test helpers
    
    // Helper to check tensor properties
    fn check_tensor<T: PartialEq + Debug>(tensor: &Tensor<T>, expected_shape: &[usize], requires_grad: bool) {
        assert_eq!(tensor.shape(), expected_shape, "Shape mismatch");
        assert_eq!(tensor.requires_grad(), requires_grad, "requires_grad mismatch");
    }

    #[test]
    fn test_linear_creation() {
        let linear = Linear::<f32>::new(10, 5, true); // With bias
        check_tensor(&linear.weight, &[5, 10], true); // Check Parameter directly
        assert!(linear.bias.is_some());
        let bias_param = linear.bias.as_ref().unwrap();
        check_tensor(bias_param, &[5], true); // Pass Parameter directly to check_tensor
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
        let linear = Linear::<f32>::new(3, 2, false);
        // Access data via Parameter's Deref to Tensor, then borrow
        linear.weight.borrow_tensor_data_mut().data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let output = linear.forward(&input);
        
        // Expected: input @ weight.T = [10, 20, 30] @ [[1, 4], [2, 5], [3, 6]]
        // = [10*1+20*2+30*3, 10*4+20*5+30*6] = [140, 320]
        let expected_output_data = vec![140.0_f32, 320.0];
        let expected_output_shape = vec![1, 2];
        
        check_tensor(&output, &expected_output_shape, true); // Requires grad because weight does
        assert_eq!(output.data().to_vec(), expected_output_data);
    }
    
    #[test]
    fn test_linear_forward_with_bias() {
        let mut linear = Linear::<f32>::new(3, 2, true);
        // Access data via Parameter's Deref to Tensor, then borrow
        linear.weight.borrow_tensor_data_mut().data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        linear.bias.as_mut().unwrap().borrow_tensor_data_mut().data = vec![0.1, 0.2];
        let input = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let output = linear.forward(&input);

        // Expected: [140, 320] + [0.1, 0.2] = [140.1, 320.2]
        let expected_output_data = vec![140.1_f32, 320.2];
        let expected_output_shape = vec![1, 2];
        
        check_tensor(&output, &expected_output_shape, true); // Requires grad from params
        assert_eq!(output.data().to_vec(), expected_output_data);
    }
    
    #[test]
    fn test_linear_forward_with_bias_batch() {
        let mut linear = Linear::<f32>::new(3, 2, true);
        // Access data via Parameter's Deref to Tensor, then borrow
        linear.weight.borrow_tensor_data_mut().data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        linear.bias.as_mut().unwrap().borrow_tensor_data_mut().data = vec![0.1, 0.2]; 
        
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
        assert_eq!(output.data().to_vec(), expected_output_data);
    }
    
    #[test]
    fn test_linear_backward_simple() {
        let mut linear = Linear::<f32>::new(2, 1, true);
        linear.weight.borrow_tensor_data_mut().data = vec![3.0, 4.0]; 
        linear.bias.as_mut().unwrap().borrow_tensor_data_mut().data = vec![0.1];
        
        let input = Tensor::new(vec![10.0, 20.0], vec![1, 2]);
        input.set_requires_grad(true);

        let output = linear.forward(&input);
        check_tensor(&output, &[1, 1], true);
        assert!((output.data()[0] - 110.1).abs() < 1e-6, "Output data mismatch");
        assert!(output.grad_fn().is_some());

        output.backward(None); 

        let grad_input = input.grad().expect("Input gradient should exist now");
        // Gradient tensors themselves usually don't require grad
        check_tensor(&grad_input, &[1, 2], false); // CORRECTED: requires_grad should be false
        assert!((grad_input.data()[0] - 3.0).abs() < 1e-6, "Input grad[0] mismatch");
        assert!((grad_input.data()[1] - 4.0).abs() < 1e-6, "Input grad[1] mismatch");
        
        let grad_weight = linear.weight.grad().unwrap();
        // Gradient tensors themselves usually don't require grad
        check_tensor(&grad_weight, &[1, 2], false); // CORRECTED: requires_grad should be false
        assert!((grad_weight.data()[0] - 10.0).abs() < 1e-6, "Weight grad[0] mismatch");
        assert!((grad_weight.data()[1] - 20.0).abs() < 1e-6, "Weight grad[1] mismatch");

        let bias_param = linear.bias.as_ref().unwrap();
        let grad_bias = bias_param.grad().unwrap();
        // Gradient tensors themselves usually don't require grad
        check_tensor(&grad_bias, &[1], false); // CORRECTED: requires_grad should be false
        assert!((grad_bias.data()[0] - 1.0).abs() < 1e-6, "Bias grad mismatch");
    }

    #[test]
    fn test_linear_backward_batch() {
        let mut linear = Linear::<f32>::new(3, 4, true);
        let w_data = (1..=12).map(|x| x as f32 * 0.1).collect::<Vec<_>>();
        let b_data = (1..=4).map(|x| x as f32 * 0.01).collect::<Vec<_>>();
        linear.weight.borrow_tensor_data_mut().data = w_data;
        linear.bias.as_mut().unwrap().borrow_tensor_data_mut().data = b_data;
        
        let input_data = (1..=6).map(|x| x as f32).collect::<Vec<_>>(); 
        let input = Tensor::new(input_data, vec![2, 3]);
        input.set_requires_grad(true);

        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[2, 4]);
        assert!(output.requires_grad());

        // Use a simple sum as the pseudo-loss for testing backward pass
        let loss = output.sum(); // Sum all elements of the output
        loss.backward(None); 

        assert!(input.grad().is_some(), "Input gradient should exist now");
        let grad_input = input.grad().unwrap();
        assert_eq!(grad_input.shape(), &[2, 3]);
        assert_eq!(grad_input.requires_grad(), false); // Gradient shouldn't require grad

        assert!(linear.weight.grad().is_some());
        let grad_weight = linear.weight.grad().unwrap();
        assert_eq!(grad_weight.shape(), &[4, 3]);
        assert_eq!(grad_weight.requires_grad(), false); // Gradient shouldn't require grad
        
        assert!(linear.bias.is_some());
        let bias_param = linear.bias.as_ref().unwrap();
        assert!(bias_param.grad().is_some());
        let grad_bias = bias_param.grad().unwrap();
        // The gradient of the bias should sum the gradients across the batch dimension.
        // If output was [2, 4], upstream grad (from sum()) is scalar 1.0 broadcasted.
        // Bias gradient should be upstream_grad summed over axis 0 -> [1.0, 1.0, 1.0, 1.0] * 2 ? 
        // Let's check the shape for now. It MUST be [4].
        assert_eq!(grad_bias.shape(), &[4]); // Bias shape is [out_features]
        assert_eq!(grad_bias.requires_grad(), false); // Gradient shouldn't require grad
        // TODO: Add precise value check for grad_bias if sum operation is confirmed
    }
}
 