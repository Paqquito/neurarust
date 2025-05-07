use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};
use std::fmt::Debug;
use crate::types::DType;
use crate::error::NeuraRustError;

/// A wrapper around a Tensor that indicates it is a trainable parameter.
/// Stores the tensor itself and an optional name.
#[derive(Debug, Clone)]
pub struct Parameter {
    pub tensor: Tensor,
    pub name: Option<String>,
}

impl Parameter {
    /// Creates a new `Parameter` wrapping the given `Tensor` with an optional name.
    /// The tensor will automatically have `requires_grad` set to `true`.
    pub fn new(tensor: Tensor, name: Option<String>) -> Self {
        let _ = tensor.set_requires_grad(true); // Ensure requires_grad is set on the tensor itself
        Parameter { tensor, name }
    }

    /// Creates a new `Parameter` wrapping the given `Tensor` without a name.
    /// The tensor will automatically have `requires_grad` set to `true`.
    pub fn new_unnamed(tensor: Tensor) -> Self {
        let _ = tensor.set_requires_grad(true);
        Parameter { tensor, name: None }
    }
    
    /// Returns the name of the parameter, if set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Consumes the Parameter and returns the underlying Tensor.
    pub fn into_inner(self) -> Tensor {
        self.tensor
    }

    /// Converts the data type of the underlying Tensor of this Parameter.
    /// This operation creates a new Tensor with the converted data and replaces
    /// the internal Tensor. The gradient of the parameter will be reset to None.
    /// The name of the parameter is preserved.
    pub fn to_dtype(&mut self, new_dtype: DType) -> Result<(), NeuraRustError> {
        let current_tensor = &self.tensor;
        let current_dtype = current_tensor.dtype();

        if current_dtype == new_dtype {
            return Ok(());
        }

        let shape = current_tensor.shape();
        // Create new tensor with converted data
        let new_tensor_data_result = match (current_dtype, new_dtype) {
            (DType::F32, DType::F64) => {
                let data_f32 = current_tensor.get_f32_data()?;
                let data_f64: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();
                Tensor::new_f64(data_f64, shape)
            }
            (DType::F64, DType::F32) => {
                let data_f64 = current_tensor.get_f64_data()?;
                let data_f32: Vec<f32> = data_f64.into_iter().map(|x| x as f32).collect();
                Tensor::new(data_f32, shape)
            }
            _ => {
                return Err(NeuraRustError::UnsupportedOperation(
                    format!("Unsupported dtype conversion from {:?} to {:?} for Parameter '{}'", 
                            current_dtype, new_dtype, self.name().unwrap_or_default())
                ));
            }
        };
        
        let new_tensor = new_tensor_data_result?;

        let _ = new_tensor.set_requires_grad(true);
        // Reset gradient after dtype conversion
        let mut new_tensor_data_guard = new_tensor.write_data();
        new_tensor_data_guard.grad = None;
        // Drop write guard before replacing self.tensor
        drop(new_tensor_data_guard);

        self.tensor = new_tensor;
        Ok(())
    }

    // Placeholder for to_device
    pub fn to_device(&mut self, new_device: crate::device::StorageDevice) -> Result<(), NeuraRustError> {
        let current_device = self.tensor.device();
        if current_device == new_device {
            if new_device == crate::device::StorageDevice::CPU { // Only check if already CPU
                 return Ok(());
            }
        }
        // For now, only CPU is supported, so any other target is an error.
        // If current is CPU and target is different, it's an error.
        if new_device != crate::device::StorageDevice::CPU {
            Err(NeuraRustError::UnsupportedOperation(
                format!("Device transfer to {:?} not yet supported for Parameter '{}'. Only CPU is supported.", 
                        new_device, self.name().unwrap_or_default())
            ))
        } else {
            // This case (current_device != CPU and new_device == CPU) is not yet handled.
            // But since we only support CPU, current_device should always be CPU.
            // Actual device transfer logic for self.tensor would go here when supported.
            // self.tensor = self.tensor.to(new_device)?;
            Ok(())
        }
    }

    /// Returns a reference to the underlying `Tensor`.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Returns a mutable reference to the underlying `Tensor`.
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }

    /// Clears the gradient of the underlying tensor by setting it to `None`.
    pub fn zero_grad(&mut self) {
        self.tensor.clear_grad();
    }
}

// Allow accessing the underlying Tensor immutably via Deref.
impl Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

// Allow accessing the underlying Tensor mutably via DerefMut.
impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}

#[cfg(test)]
#[path = "parameter_test.rs"]
mod tests; 