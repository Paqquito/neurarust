use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};
use std::fmt::Debug;
use crate::types::DType;
use crate::error::NeuraRustError;

/// A wrapper around a Tensor that indicates it is a trainable parameter.
/// Stores the tensor itself and potentially metadata in the future.
#[derive(Debug, Clone)]
pub struct Parameter(pub Tensor);

impl Parameter {
    /// Creates a new `Parameter` wrapping the given `Tensor`.
    /// The tensor will automatically have `requires_grad` set to `true`.
    pub fn new(tensor: Tensor) -> Self {
        let t = tensor;
        let _ = t.set_requires_grad(true);
        Parameter(t)
    }

    /// Consumes the Parameter and returns the underlying Tensor.
    pub fn into_inner(self) -> Tensor {
        self.0
    }

    /// Converts the data type of the underlying Tensor of this Parameter.
    /// This operation creates a new Tensor with the converted data and replaces
    /// the internal Tensor. The gradient of the parameter will be reset to None.
    pub fn to_dtype(&mut self, new_dtype: DType) -> Result<(), NeuraRustError> {
        let current_tensor = &self.0;
        let current_dtype = current_tensor.dtype();

        if current_dtype == new_dtype {
            return Ok(());
        }

        let shape = current_tensor.shape();
        // Create new tensor with converted data
        let new_tensor = match (current_dtype, new_dtype) {
            (DType::F32, DType::F64) => {
                let data_f32 = current_tensor.get_f32_data()?;
                let data_f64: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();
                Tensor::new_f64(data_f64, shape)?
            }
            (DType::F64, DType::F32) => {
                let data_f64 = current_tensor.get_f64_data()?;
                let data_f32: Vec<f32> = data_f64.into_iter().map(|x| x as f32).collect();
                Tensor::new(data_f32, shape)?
            }
            _ => {
                // Should not happen if current_dtype != new_dtype check is exhaustive
                // or if we only support F32/F64
                return Err(NeuraRustError::UnsupportedOperation(
                    format!("Unsupported dtype conversion from {:?} to {:?}", current_dtype, new_dtype)
                ));
            }
        };

        let _ = new_tensor.set_requires_grad(true);
        // Reset gradient after dtype conversion
        let mut new_tensor_data = new_tensor.write_data();
        new_tensor_data.grad = None;
        // Drop write guard before replacing self.0
        drop(new_tensor_data);

        self.0 = new_tensor;
        Ok(())
    }

    // Placeholder for to_device
    pub fn to_device(&mut self, new_device: crate::device::StorageDevice) -> Result<(), NeuraRustError> {
        let current_device = self.0.device();
        if current_device == new_device {
            if new_device == crate::device::StorageDevice::CPU { // Only check if already CPU
                 return Ok(());
            }
        }
        // For now, only CPU is supported, so any other target is an error.
        // If current is CPU and target is different, it's an error.
        if new_device != crate::device::StorageDevice::CPU {
            Err(NeuraRustError::UnsupportedOperation(
                format!("Device transfer to {:?} not yet supported. Only CPU is supported.", new_device)
            ))
        } else {
            // This case (current_device != CPU and new_device == CPU) is not yet handled.
            // But since we only support CPU, current_device should always be CPU.
            Ok(())
        }
    }

    /// Returns a reference to the underlying `Tensor`.
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }

    /// Returns a mutable reference to the underlying `Tensor`.
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.0
    }

    /// Clears the gradient of the underlying tensor by setting it to `None`.
    pub fn zero_grad(&mut self) {
        self.0.clear_grad(); // Assuming Tensor has a clear_grad method
    }
}

// Allow accessing the underlying Tensor immutably via Deref.
impl Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Allow accessing the underlying Tensor mutably via DerefMut.
impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[path = "parameter_test.rs"]
mod tests; 