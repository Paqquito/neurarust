use crate::autograd::backward_op::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::Tensor;
use std::fmt::Debug;
// Add Add trait needed for potential acc_grad, Send/Sync for BackwardOp
use std::sync::{Arc, RwLock};
use crate::buffer::{Buffer, CpuBuffer};
use crate::types::DType;

// --- Backward Operation Structure ---

/// Backward operation for the negation function.
#[derive(Debug)]
struct NegBackward { // Removed generic <T>
    // Store the input tensor ID for the graph traversal
    input_node: Arc<RwLock<TensorData>>,
}

// --- Backward Operation Implementation ---

impl BackwardOp for NegBackward {
    /// Computes gradient for the negation operation z = -a.
    /// grad(a) = grad_output * (-1) = -neg_output
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // grad_a = -grad_output
        // Reuse neg_op for the backward pass
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

    // inputs() method is no longer needed with the simplified Variable approach for now.
    // If needed later for graph construction, it would return pointers/IDs of input TensorData.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        // Return the pointer to the stored input node
        vec![Arc::as_ptr(&self.input_node)]
    }
}

// --- Forward Operation ---

/// Performs element-wise negation on a tensor.
/// -input
/// Currently only supports f32 tensors on CPU.
pub fn neg_op(input: &Tensor) -> Result<Tensor, NeuraRustError> {
    // Lock input data for reading
    let input_data_guard = input.data.read().map_err(|_| NeuraRustError::LockError {
            lock_type: "read".to_string(),
            reason: "Failed to lock input TensorData for read in neg_op".to_string(),
        })?;

    // --- Device and DType Check ---
    if input_data_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "neg_op".to_string(),
            expected: StorageDevice::CPU,
            actual: input_data_guard.device,
        });
    }
    if input_data_guard.dtype != DType::F32 {
        // Use UnsupportedOperation for incorrect dtype for now
        return Err(NeuraRustError::UnsupportedOperation(
            format!("neg_op currently only supports F32, got {:?}", input_data_guard.dtype)
        ));
    }

    // --- Data Access and Computation ---
    let output_data_vec: Vec<f32> = match &*input_data_guard.buffer {
        Buffer::Cpu(cpu_buffer) => match cpu_buffer {
            CpuBuffer::F32(data_arc) => {
                // Get slice from buffer based on offset and strides
                // Use get_f32_slice helper or direct iteration if possible
                // For now, assume contiguous and use simple iterator
                // TODO: Handle non-contiguous tensors here if neg needs it (elementwise usually doesn't)
                if !input_data_guard.is_contiguous() {
                    // For simplicity, elementwise ops often create contiguous output anyway
                    // Or could error if non-contiguous input needs special handling not implemented
                    // Let's assume for now neg creates a new contiguous buffer
                    // This matches the current collect() behavior
                    data_arc.iter().map(|&x| -x).collect()
                } else {
                    // Apply negation using optimized iterators on the (assumed) contiguous data
                    data_arc.iter().map(|&x| -x).collect()
                }
            }
            // Non-F32 case is caught by the dtype check above
        },
        // Add explicit non-CPU arm
        _ => return Err(NeuraRustError::DeviceMismatch {
            operation: "neg_op (buffer access)".to_string(),
            expected: StorageDevice::CPU,
            actual: input_data_guard.device,
        }),
    };

    // --- Output Tensor Creation ---
    let output_shape = input_data_guard.shape.clone();
    // Keep input requires_grad status and node Arc before dropping the read lock
    let input_requires_grad = input_data_guard.requires_grad;
    let input_node_arc = if input_requires_grad { Some(Arc::clone(&input.data)) } else { None };

    // Drop the read lock BEFORE creating the new TensorData/Tensor
    // (to avoid potential deadlocks if new() or set_grad_fn tried to lock again)
    drop(input_data_guard);

    // Create the output tensor using Tensor::new which handles TensorData creation
    let output_tensor = Tensor::new(output_data_vec, output_shape)?; // Using Vec<f32> directly

    // --- Autograd Setup ---
    // Only set up backward graph if the input requires gradients
    if input_requires_grad {
        if let Some(node_arc) = input_node_arc {
            // Acquire write lock on the NEW output tensor's data
            let mut output_data_write_guard = output_tensor.data.write().map_err(|_| NeuraRustError::LockError {
                 lock_type: "write".to_string(),
                 reason: "Failed to lock output TensorData for write (autograd setup in neg_op)".to_string(),
             })?;
            output_data_write_guard.requires_grad = true;
            // Create NegBackward op with the input node Arc
            let backward_op = NegBackward { input_node: node_arc }; // Removed T
            // Set the grad_fn for the output tensor
            output_data_write_guard.grad_fn = Some(Arc::new(backward_op));
        } else {
            // This case should not happen if input_requires_grad was true
            // Return an internal error if it does occur
             return Err(NeuraRustError::InternalError("Input requires grad but its Node Arc is missing in neg_op".to_string()));
        }
    }

    // --- Return Result --- 
    Ok(output_tensor) // Return the newly created tensor
}

// --- std::ops::Neg implementation ---
// Implement the Neg trait for Tensor by calling neg_op
/* // Remove the generic implementation for now
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
*/

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    use crate::error::NeuraRustError;
    // Importer seulement check_tensor_near, l'autre est supprimé
    use crate::utils::testing::{check_tensor_near};
    use crate::autograd::grad_check::check_grad; 

    #[test]
    fn test_neg_ok() {
        let t1 = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let t2 = neg_op(&t1).unwrap();
        // Use check_tensor_near for comparison
        check_tensor_near(&t2, &[2, 2], &[-1.0, 2.0, -3.0, 4.0], 1e-6);
        assert_eq!(t2.dtype(), DType::F32); // Check dtype
    }

    #[test]
    fn test_neg_backward() {
        // Input data for the test
        let input_data = vec![1.0f32, -2.0, 3.0, 0.0];
        let input_shape = vec![4];

        // 1. Create the input Tensor and set requires_grad
        let input_tensor = Tensor::from_vec_f32(input_data, input_shape.clone()).unwrap();
        input_tensor.set_requires_grad(true).expect("Failed to set requires_grad"); // Call separately

        // 2. Define the function to be checked
        let neg_fn_for_check = |inputs: &[Tensor]| -> Result<Tensor, NeuraRustError> {
             if inputs.len() != 1 {
                 // Use InternalError for test helper issue
                 return Err(NeuraRustError::InternalError("neg_fn_for_check expects exactly one input tensor".to_string()));
             }
             neg_op(&inputs[0])
        };

        // 3. Determine the shape of the output of neg_op
        let output_shape = input_tensor.shape().clone();

        // 4. Create the initial gradient for the output (tensor of ones)
        // Use the public function from crate::tensor::create
        let output_grad_tensor = crate::tensor::ones(&output_shape).unwrap();

        // 5. Prepare the input slice for check_grad
        let inputs_slice = &[input_tensor]; // Slice containing the input tensor

        // 6. Define epsilon and tolerance (use f64 for check_grad internals)
        let epsilon = 1e-4f64; // Small value for finite difference
        let tolerance = 1e-2f64; // Looser tolerance for f32 comparisons

        // 7. Call check_grad with the correct arguments
        check_grad(
            neg_fn_for_check,   // The function (neg_op wrapper)
            inputs_slice,       // Slice of input tensors
            &output_grad_tensor,// Initial output gradient (ones)
            epsilon,            // Epsilon for finite difference
            tolerance,          // Tolerance for comparison
        ).unwrap(); // Panic if check fails
    }
}
