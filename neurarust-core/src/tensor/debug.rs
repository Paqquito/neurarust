// src/tensor/debug.rs
use crate::tensor::Tensor;
use crate::tensor_data::TensorData; // Need TensorData to access fields
use std::fmt;

// Manual implementation of Debug trait
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Acquire read lock to access TensorData fields safely
        match self.data.read() {
            Ok(guard) => {
                // Customize the output format
                write!(f, "Tensor(shape={:?}, strides={:?}, offset={}, device={:?}, dtype={:?}, requires_grad={}, has_grad={}, has_grad_fn={})",
                       guard.shape,
                       guard.strides,
                       guard.offset,
                       guard.device,
                       guard.dtype,
                       guard.requires_grad,
                       guard.grad.is_some(),
                       guard.grad_fn.is_some()
                       // TODO: Optionally include a preview of the data? Be careful with large tensors.
                       // Could add a helper function in TensorData/Buffer to get a data preview string.
                )
            }
            Err(_) => write!(f, "Tensor(Error: RwLock poisoned)"), // Handle lock poisoning
        }
    }
}