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

#[derive(Debug)]
struct LinearBackward<T> {
    input: Tensor<T>, // Clone of input tensor from forward pass
    weight: Parameter<T>, // Clone of weight parameter from forward pass
    bias: Option<Parameter<T>>, // Keep clone of bias parameter (optional)
    
    // Weak references to original tensors/parameters for grad accumulation
    input_ref: Weak<RefCell<TensorData<T>>>,
    weight_ref: Weak<RefCell<TensorData<T>>>,
    bias_ref: Option<Weak<RefCell<TensorData<T>>>>, // Option because bias is optional
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for LinearBackward<T>
where
    T: Add<Output = T> + Mul<Output = T> + AddAssign + Neg<Output = T> + Sub<Output = T> + 
       Zero + One + Copy + Clone + 'static + Debug + PartialEq + IterSum + Default,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        // println!("\n--- DEBUG LinearBackward --- Upstream shape: {:?}", upstream_grad.shape());
        
        // 1. Gradient w.r.t. Layer Input (dL/dInput = dL/dOutput @ W)
        let needs_grad_input = self.input_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        // println!("[Input Grad] Needs grad? {}", needs_grad_input);
        if needs_grad_input {
            let weight_tensor = self.weight.deref();
            // println!("[Input Grad] Calc: upstream({:?}) @ W({:?})", upstream_grad.shape(), weight_tensor.shape());
            let grad_input = upstream_grad.matmul(weight_tensor); 
            grad_input.set_requires_grad(false);
            // println!("[Input Grad] Result shape: {:?}", grad_input.shape());
            crate::autograd::accumulate_gradient(gradients, &self.input_ref, grad_input);
            // println!("[Input Grad] Accumulated.");
        }

        // 2. Gradient w.r.t. Weight (dL/dW = (dL/dOutput)^T @ input)
        let needs_grad_weight = self.weight_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
        // println!("[Weight Grad] Needs grad? {}", needs_grad_weight);
        // Calculate gradient regardless, accumulation depends on requires_grad.
        // if needs_grad_weight { // Don't skip calculation based on needs_grad
            // println!("[Weight Grad] Calc: upstream.T({:?}) @ input({:?})", upstream_grad.shape(), self.input.shape());
            let grad_wrt_output_t = upstream_grad.transpose();
            // println!("[Weight Grad] upstream.T shape: {:?}", grad_wrt_output_t.shape());
            let grad_weight = grad_wrt_output_t.matmul(&self.input);
            grad_weight.set_requires_grad(false);
            // println!("[Weight Grad] Result shape: {:?}", grad_weight.shape());
            crate::autograd::accumulate_gradient(gradients, &self.weight_ref, grad_weight);
            // println!("[Weight Grad] Accumulated.");
        // }

        // 3. Gradient w.r.t. Bias (dL/db = sum(dL/dOutput, axis=0)) 
        if let Some(ref bias_weak_ref) = self.bias_ref {
            let needs_grad_bias = bias_weak_ref.upgrade().map_or(false, |rc| rc.borrow().requires_grad);
            // println!("[Bias Grad] Needs grad? {}", needs_grad_bias);
            if needs_grad_bias {
                let upstream_rank = upstream_grad.shape().len();
                // println!("[Bias Grad] Calc from upstream({:?})", upstream_grad.shape());
                let grad_bias = if upstream_rank > 1 {
                    let axes_to_sum: Vec<usize> = (0..upstream_rank - 1).collect();
                    let summed_grad = upstream_grad.sum_axes(&axes_to_sum, false);
                    summed_grad.set_requires_grad(false);
                    // println!("[Bias Grad] Result shape (summed): {:?}", summed_grad.shape());
                    summed_grad
                } else {
                     let cloned_grad = upstream_grad.clone();
                     cloned_grad.set_requires_grad(false);
                     // println!("[Bias Grad] Result shape (cloned): {:?}", cloned_grad.shape());
                     cloned_grad
                }; 
                 crate::autograd::accumulate_gradient(gradients, bias_weak_ref, grad_bias);
                 // println!("[Bias Grad] Accumulated.");
            }
        } else {
            // println!("[Bias Grad] No bias in this layer.");
        }
        // println!("--- DEBUG LinearBackward END ---");
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        // Return weak refs to all potential gradient sources: input, weight, bias
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

        // Create scalar loss
        let loss = output.sum();

        // println!("\nDEBUG test_linear_backward_simple: Running backward on loss...");
        loss.backward(None);
        // println!("DEBUG test_linear_backward_simple: Backward finished.");

        // println!("DEBUG test_linear_backward_simple: Checking input grad...");
        let grad_input_opt = input.grad();
        assert!(grad_input_opt.is_some(), "Input gradient missing");
        let grad_input = grad_input_opt.unwrap();
        check_tensor(&grad_input, &[1, 2], false); // Grads don't require grad
        assert!((grad_input.data()[0] - 3.0).abs() < 1e-6, "Input grad[0] mismatch");
        assert!((grad_input.data()[1] - 4.0).abs() < 1e-6, "Input grad[1] mismatch");
        
        // println!("DEBUG test_linear_backward_simple: Checking weight grad...");
        let grad_weight_opt = linear.weight.grad();
        assert!(grad_weight_opt.is_some(), "Weight gradient missing");
        let grad_weight = grad_weight_opt.unwrap();
        check_tensor(&grad_weight, &[1, 2], false); // Grads don't require grad
        assert!((grad_weight.data()[0] - 10.0).abs() < 1e-6, "Weight grad[0] mismatch");
        assert!((grad_weight.data()[1] - 20.0).abs() < 1e-6, "Weight grad[1] mismatch");

        // println!("DEBUG test_linear_backward_simple: Checking bias grad...");
        let bias_param = linear.bias.as_ref().unwrap();
        let grad_bias_opt = bias_param.grad();
        assert!(grad_bias_opt.is_some(), "Bias gradient missing");
        let grad_bias = grad_bias_opt.unwrap();
        check_tensor(&grad_bias, &[1], false); // Grads don't require grad
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

        let loss = output.sum();
        // println!("\nDEBUG test_linear_backward_batch: Running backward...");
        loss.backward(None);
        // println!("DEBUG test_linear_backward_batch: Backward finished.");

        // println!("DEBUG test_linear_backward_batch: Checking input grad...");
        let grad_input_opt = input.grad();
        assert!(grad_input_opt.is_some(), "Input gradient missing");
        let grad_input = grad_input_opt.unwrap();
        check_tensor(&grad_input, &[2, 3], false);

        // println!("DEBUG test_linear_backward_batch: Checking weight grad...");
        let grad_weight_opt = linear.weight.grad();
        assert!(grad_weight_opt.is_some(), "Weight gradient missing");
        let grad_weight = grad_weight_opt.unwrap();
        check_tensor(&grad_weight, &[4, 3], false);
        
        // println!("DEBUG test_linear_backward_batch: Checking bias grad...");
        let bias_param = linear.bias.as_ref().unwrap();
        let grad_bias_opt = bias_param.grad();
        assert!(grad_bias_opt.is_some(), "Bias gradient missing");
        let grad_bias = grad_bias_opt.unwrap();
        check_tensor(&grad_bias, &[4], false);
        // dLoss/dBias = Sum(dLoss/dOutput, axis=0)
        // dLoss/dOutput is 1.0 for every element because loss = output.sum()
        // So grad_bias should be [batch_size, batch_size, ..., batch_size]
        // Here batch_size is 2. So expected grad is [2.0, 2.0, 2.0, 2.0]
        assert_eq!(grad_bias.data().to_vec(), vec![2.0_f32, 2.0, 2.0, 2.0], "Bias grad data mismatch");
    }
}
 