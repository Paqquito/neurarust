use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
// Add Add trait needed for potential acc_grad, Send/Sync for BackwardOp
use std::ops::{Add, AddAssign, Neg};
use std::sync::{Arc, RwLock};

// --- Backward Operation Structure ---

/// Backward operation context for unary negation (-a).
/// Stores a cloned Tensor of the input to safely get its NodeId.
#[derive(Debug)]
struct NegBackward<T: 'static + Debug + Copy + Send + Sync> {
    // Store cloned tensor to satisfy Send + Sync and provide NodeId
    a: Tensor<T>,
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for NegBackward<T>
where
    T: Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + AddAssign
        + Add<Output = T>
        + Neg<Output = T> // Required for the backward calculation
        + Copy
        + Send
        + Sync
        + 'static
        + PartialEq
        + PartialOrd, // Keep consistent, although not strictly needed here
{
    /// Computes gradient for the negation operation z = -a.
    /// grad(a) = grad_output * (-1) = -grad_output
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = -grad_output
        // Use the existing neg_op for this calculation.
        let grad_a = neg_op(grad_output)?;

        // Ensure gradient is on the correct device (although currently CPU only)
        let expected_device = grad_output.device();
        if grad_a.device() != expected_device {
            return Err(NeuraRustError::BackwardError(format!(
                "NegBackward gradient device mismatch. Expected {:?}, got grad_a: {:?}",
                expected_device,
                grad_a.device()
            )));
        }

        // Negation is unary, so we return a Vec with one gradient tensor.
        Ok(vec![grad_a])
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData<T>>> {
        // Return NodeId from the stored tensor clone
        vec![self.a.get_node_id()]
    }
}

// --- Forward Operation ---

/// Performs unary negation for a Tensor.
/// Requires the tensor to be on CPU.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn neg_op<T>(a: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    // Add bounds required for BackwardOp and autograd linkage
    T: Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    // --- Autograd Setup ---
    let requires_grad = a.requires_grad();
    // Store a clone if grad is required
    let mut a_maybe_clone: Option<Tensor<T>> = None;

    if requires_grad {
        a_maybe_clone = Some(a.clone());
    }
    // --- End Autograd Setup ---

    // Acquire read lock
    let a_guard = a.read_data();

    // --- Device Check ---
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Negation is currently only supported on CPU, not {:?}",
            device
        )));
    }

    // --- Get CPU Data Buffer ---
    let a_data_arc = a_guard.data.cpu_data()?.clone();

    // --- Calculation (Handles Strides) ---
    let output_shape = a_guard.shape.clone();
    let numel = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel);

    let strides = &a_guard.strides;
    let offset = a_guard.offset;

    if numel > 0 {
        // Optimized iteration for contiguous case
        if a_guard.is_contiguous() {
            let start = a_guard.offset;
            let end = start + numel;
            if end <= a_data_arc.len() { // Check bounds
                result_data_vec.extend(a_data_arc[start..end].iter().map(|&x| -x));
            } else {
                 return Err(NeuraRustError::InternalError(
                     "Contiguous offset calculation resulted in out-of-bounds access in neg_op".to_string()
                 ));
            }
        } else {
            // Non-contiguous case: iterate using coordinates (slower)
            let mut current_coords = vec![0; output_shape.len()];
            for linear_idx in 0..numel {
                let mut relative_offset = 0;
                for i in 0..output_shape.len() {
                    relative_offset += current_coords[i] * strides[i];
                }
                let logical_offset = offset + relative_offset;

                 if logical_offset < a_data_arc.len() { // Check bounds
                    let val_a = a_data_arc[logical_offset];
                    result_data_vec.push(-val_a);
                 } else {
                     return Err(NeuraRustError::InternalError(
                         "Non-contiguous offset calculation resulted in out-of-bounds access in neg_op".to_string()
                     ));
                 }

                // Increment coordinates
                if linear_idx < numel - 1 {
                    let mut dim_to_inc = output_shape.len() - 1;
                    loop {
                        current_coords[dim_to_inc] += 1;
                        if current_coords[dim_to_inc] < output_shape[dim_to_inc] {
                            break;
                        }
                        current_coords[dim_to_inc] = 0;
                        if dim_to_inc == 0 {
                            break;
                        }
                        dim_to_inc -= 1;
                    }
                }
            }
        }
    }

    // Drop lock
    drop(a_guard);

    // --- Create Result ---
    let result_tensor = Tensor::new(result_data_vec, output_shape)?;

    // --- Autograd Linkage ---
    if requires_grad {
         if let Some(a_clone) = a_maybe_clone {
             // Pass the clone to the backward context
             let backward_context = NegBackward { a: a_clone };
             let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);
             result_tensor.set_requires_grad(true)?;
             result_tensor.set_grad_fn(Some(backward_op_arc))?;
         } else {
              return Err(NeuraRustError::InternalError(
                  "requires_grad was true but failed to get input clone in neg_op".to_string()
              ));
         }
    }
    // --- End Autograd Linkage ---

    Ok(result_tensor)
}

// --- std::ops::Neg implementation ---
// Implement the Neg trait for Tensor by calling neg_op
impl<T> Neg for &Tensor<T>
where
    // Bounds must match neg_op requirements
    T: Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static,
{
    type Output = Result<Tensor<T>, NeuraRustError>;

    fn neg(self) -> Self::Output {
        neg_op(self)
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    use crate::Tensor;
    use approx::assert_abs_diff_eq;
    use num_traits::{Float, One, Zero, Signed};
    use std::cmp::PartialEq;
    use std::default::Default;
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::ops::{AddAssign, Neg};

    fn create_test_tensor<
        T: Clone
            + Debug
            + PartialEq
            + Zero
            + One
            + AddAssign
            + Add<Output = T>
            + Neg<Output = T>
            + Signed
            + Copy
            + Default
            + Sum
            + PartialOrd
            + Float
            + approx::AbsDiffEq<Epsilon = T>
            + approx::RelativeEq<Epsilon = T>
            + approx::UlpsEq<Epsilon = T>
            + Send
            + Sync
            + 'static,
    >(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T> {
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }

    #[test]
    fn test_neg_ok() {
        let t1 = create_test_tensor(vec![1.0f64, -2.0, 3.0, -4.0], vec![2, 2]);
        let expected_data = vec![-1.0f64, 2.0, -3.0, 4.0];
        let expected_shape = vec![2, 2];
        let result = neg_op(&t1);
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");

        let res_trait = (-&t1).expect("Neg trait failed");
        let trait_buffer_arc = res_trait.borrow_data_buffer();
        let trait_cpu_data = trait_buffer_arc.cpu_data().expect("Trait result not on CPU");
        assert_eq!(trait_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_trait.shape(), expected_shape, "Shape mismatch (trait)");
    }

    #[test]
    fn test_neg_backward() {
        let a_init = create_test_tensor(vec![1.0f64, -2.0, 3.0, -4.0], vec![2, 2]);
        a_init.set_requires_grad(true).unwrap();
        let a = a_init;

        let forward_fn = |inputs: &[Tensor<f64>]| neg_op(&inputs[0]);

        check_grad(
            forward_fn,
            &[a.clone()],
            &Tensor::ones(vec![2, 2]).unwrap(),
            1e-6,
            1e-6,
        )
        .expect("Gradient check failed for negation");

        let z = neg_op(&a).unwrap();
        z.backward(Some(Tensor::ones(z.shape()).unwrap())).expect("Backward pass failed");

        let grad_a = a.grad().unwrap();

        let expected_grad_a = vec![-1.0f64, -1.0, -1.0, -1.0];

        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();

        assert_eq!(grad_a.shape(), vec![2, 2]);
        for (i, &val) in grad_a_data.iter().enumerate() {
            assert_abs_diff_eq!(val, expected_grad_a[i], epsilon = 1e-6);
        }
    }
}
