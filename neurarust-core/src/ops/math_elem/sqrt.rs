// neurarust-core/src/ops/math_elem/sqrt.rs

use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::autograd::{self, BackwardOp};
use crate::error::NeuraRustError;
use num_traits::{Float, Zero, One, FromPrimitive};
use std::fmt::{Debug};
use std::cell::{RefCell};
use std::rc::{Rc, Weak};
use std::marker::PhantomData;
use std::collections::HashMap;
use std::iter::Sum;
use std::ops::{Mul, AddAssign};
use crate::ops::arithmetic::{mul, div}; // For backward pass
 // Import ones

// --- Backward Operation ---

#[derive(Debug)]
struct SqrtBackward<T> {
    input_ref: Weak<RefCell<TensorData<T>>>,
    output: Tensor<T>, // Store output for gradient calculation: grad = upstream_grad / (2 * sqrt(input)) = upstream_grad / (2 * output)
    _phantom: PhantomData<T>,
}

impl<T> BackwardOp<T> for SqrtBackward<T>
where
    // Bounds needed for calculation + accumulate_gradient
    T: Float + Debug + 'static + Mul<Output=T> + AddAssign + Default + Zero + One + Clone + Copy + PartialEq + Sum + FromPrimitive,
{
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>) {
        if let Some(input_rc) = self.input_ref.upgrade() {
            if input_rc.borrow().requires_grad {
                // local_grad = upstream_grad * (1 / (2 * output))
                let two = T::from_f64(2.0).expect("Cannot create 2.0 from f64");
                let two_tensor = Tensor::scalar(two);

                // Calculate denominator: 2 * output
                let denominator_res = mul(&two_tensor, &self.output);

                // Check for multiplication error before proceeding
                let denominator = match denominator_res {
                    Ok(den) => den,
                    Err(e) => {
                        eprintln!("Error during sqrt backward (multiplication): {:?}", e);
                        // Decide how to handle the error, e.g., panic or skip gradient accumulation
                        // Panic might be safer to highlight internal issues.
                        panic!("Internal error during sqrt backward (multiplication): {:?}", e);
                    }
                };

                // Calculate local gradient: upstream_grad / denominator
                let local_grad_res = div(upstream_grad, &denominator);

                 // Check for division error (e.g., division by zero) before proceeding
                 let local_grad = match local_grad_res {
                    Ok(grad) => grad,
                    Err(e) => {
                        eprintln!("Error during sqrt backward (division): {:?}", e);
                        // Handle division error (e.g., if output contained zero)
                        // Potentially substitute with a large number, zero, or panic.
                        panic!("Internal error during sqrt backward (division - possible zero output?): {:?}", e);
                    }
                };

                // Accumulate gradient
                autograd::accumulate_gradient(gradients, &self.input_ref, local_grad);
            }
        } else {
             eprintln!("SqrtBackward: Input tensor weak reference expired.");
        }
    }

    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
        vec![self.input_ref.clone()]
    }
}


// --- Forward Operation ---

/// Computes the element-wise square root of the tensor.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn sqrt_op<T>(input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    // Add FromPrimitive bound here for Rc<dyn BackwardOp> conversion
    T: Float + Debug + Clone + AddAssign + Default + Zero + One + Sum + 'static + Mul<Output=T> + Copy + PartialEq + PartialOrd + FromPrimitive,
{
    let input_td = input.borrow_tensor_data();
    // Ensure all elements are non-negative
    for &x in input_td.data.iter() {
        if x < T::zero() {
            // Use InternalError and format with Debug
            return Err(NeuraRustError::InternalError(
                format!("Cannot compute sqrt of negative number: {:?}", x) // Use {:?} for Debug
            ));
        }
    }

    let result_data: Vec<T> = input_td.data.iter().map(|&x| x.sqrt()).collect();
    let result_shape = input_td.shape.clone();

    let input_weak_ref = input.get_weak_ref();
    drop(input_td);

    let result = Tensor::new(result_data, result_shape)?;

    let requires_grad = input.requires_grad();
    if requires_grad {
        result.set_requires_grad(true);
        // Use SqrtBackward for grad_fn
        let grad_fn = SqrtBackward {
            input_ref: input_weak_ref,
            output: result.clone(),
            _phantom: PhantomData,
        };
        // Conversion Rc<SqrtBackward> -> Rc<dyn BackwardOp> requires T: Sum, Default, One etc.
        result.set_grad_fn(Some(Rc::new(grad_fn)));
    }

    Ok(result)
}

// --- Tensor Method (Calls fallible op) ---
// This block defining Tensor::sqrt is removed as it duplicates the one in tensor/mod.rs
// // This is the single, correct implementation
// impl<T> Tensor<T> {
//     /// Computes the element-wise square root of the tensor.
//     ///
//     /// # Panics
//     /// Panics if the operation fails (e.g., negative input).
//     /// Use `neurarust::ops::math_elem::sqrt_op` for fallible sqrt.
//     pub fn sqrt(&self) -> Tensor<T>
//     where
//         // Ensure bounds match sqrt_op
//         T: Float + Debug + Clone + AddAssign + Default + Zero + One + Sum + 'static + Mul<Output=T> + Copy + PartialEq + PartialOrd,
//     {
//         sqrt_op(self)
//             .unwrap_or_else(|e| panic!("Tensor sqrt failed: {:?}", e))
//     }
// }


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::error::NeuraRustError;
    use crate::tensor::ones; // Import ones
    use num_traits::{Float, Zero, One, FromPrimitive}; // Add FromPrimitive
    use std::iter::Sum;
    use std::ops::{AddAssign, Mul};
    use crate::autograd::BackwardOp;
    use std::collections::HashMap;
    use std::cell::RefCell;
    use crate::tensor_data::TensorData;


    // Helper to create tensors, add bounds needed by sqrt and backward
    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
         T: Float + Debug + Clone + AddAssign + Default + Zero + One + Sum + 'static + Mul<Output=T> + Copy + PartialEq + PartialOrd,
    {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
    where
        T: Float + Debug + Clone + AddAssign + Default + Zero + One + Sum + 'static + Mul<Output=T> + Copy + PartialEq + PartialOrd,
    {
        Tensor::new_with_grad(data, shape).expect("Test grad tensor creation failed")
    }

    fn assert_approx_eq_vec(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ");
        for (val_a, val_b) in a.iter().zip(b.iter()) {
            assert!((val_a - val_b).abs() < epsilon, "Value mismatch: {} vs {}", val_a, val_b);
        }
    }


    #[test]
    fn test_sqrt_op_forward() -> Result<(), NeuraRustError> {
        let t = create_test_tensor::<f32>(vec![1.0, 4.0, 9.0, 0.0], vec![2, 2]);
        let result = sqrt_op(&t)?;
        assert_eq!(result.shape(), vec![2, 2]);
        assert_approx_eq_vec(&result.data(), &[1.0, 2.0, 3.0, 0.0], 1e-6);
        assert!(!result.requires_grad());
        Ok(())
    }

    #[test]
    fn test_sqrt_op_forward_negative_input() {
        let t = create_test_tensor::<f32>(vec![1.0, -4.0], vec![2]);
        let result = sqrt_op(&t);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NeuraRustError::InternalError(_)));
    }
     #[test]
    fn test_tensor_sqrt_method() {
        let t = create_test_tensor::<f32>(vec![1.0, 4.0, 9.0], vec![3]);
        let result = t.sqrt(); // Uses unwrap internally
        assert_eq!(result.shape(), vec![3]);
         assert_approx_eq_vec(&result.data(), &[1.0, 2.0, 3.0], 1e-6);
    }

     #[test]
    #[should_panic(expected = "sqrt failed")]
    fn test_tensor_sqrt_method_panic() {
        let t = create_test_tensor::<f32>(vec![-1.0], vec![1]);
        let _ = t.sqrt(); // Should panic
    }


    #[test]
    fn test_sqrt_grad_propagation() -> Result<(), NeuraRustError> {
        let t_grad = create_test_tensor_with_grad::<f32>(vec![4.0, 9.0], vec![2]);
        let result_grad = sqrt_op(&t_grad)?;
        assert!(result_grad.requires_grad());
        assert!(result_grad.grad_fn().is_some());

        let t_no_grad = create_test_tensor::<f32>(vec![16.0], vec![1]);
        let result_no_grad = sqrt_op(&t_no_grad)?;
        assert!(!result_no_grad.requires_grad());
         assert!(result_no_grad.grad_fn().is_none());
        Ok(())
    }

    #[test]
    fn test_sqrt_backward() -> Result<(), NeuraRustError> {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 4.0, 9.0, 16.0], vec![4]);
        let result = sqrt_op(&t)?;

        // Provide upstream gradient of ones
        let upstream_grad = ones(result.shape()).expect("Failed to create upstream grad");
        result.backward(Some(&upstream_grad));

        let grad_t = t.grad().expect("Grad t missing");
         assert_eq!(grad_t.shape(), vec![4]);
        let expected_grad = vec![0.5, 0.25, 1.0/6.0, 0.125];
         assert_approx_eq_vec(&grad_t.data(), &expected_grad, 1e-6);
         Ok(())
    }

     #[test]
    fn test_sqrt_backward_with_upstream() -> Result<(), NeuraRustError> {
        let t = create_test_tensor_with_grad::<f32>(vec![4.0, 9.0], vec![2]);
        let result = sqrt_op(&t)?;
        let upstream_grad = Tensor::new(vec![10.0, 2.0], vec![2]).unwrap();

        result.backward(Some(&upstream_grad));

        // Grad = upstream / (2 * output)
        // Output = [2, 3]
        // Grad = [10/(2*2), 2/(2*3)] = [2.5, 1/3]
        let grad_t = t.grad().expect("Grad t missing");
        assert_eq!(grad_t.shape(), vec![2]);
        let expected_grad = vec![2.5, 1.0/3.0];
        assert_approx_eq_vec(&grad_t.data(), &expected_grad, 1e-6);
        Ok(())
    }

     #[test]
    #[should_panic(expected = "Internal error during sqrt backward (division - possible zero output?)")]
    fn test_sqrt_backward_division_by_zero() {
         let t = create_test_tensor_with_grad::<f32>(vec![0.0, 4.0], vec![2]);
         let result = sqrt_op(&t).expect("Forward pass failed");
         assert_approx_eq_vec(&result.data(), &[0.0, 2.0], 1e-6);

         // Provide upstream gradient of ones
         let upstream_grad = ones(result.shape()).expect("Failed to create upstream grad");
         // Backward pass should panic due to internal panic in SqrtBackward
         result.backward(Some(&upstream_grad)); 
    }
}
