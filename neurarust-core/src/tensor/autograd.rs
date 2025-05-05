// src/tensor/autograd.rs

use crate::{ // Imports needed for autograd methods
    autograd::{graph::ComputationGraph, BackwardOp},
    error::NeuraRustError,
    tensor::Tensor,
    tensor_data::TensorData,
    types::DType, // Needed for backward initial grad
    // tensor::create::ones_like, // Need a way to create initial grad
};
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Checks if the tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.read_data().requires_grad
    }

    /// Creates a new tensor with the specified `requires_grad` status.
    /// Deprecated in favor of `requires_grad_`.
    #[deprecated = "Use requires_grad_() for inplace modification on leaf nodes"]
    pub fn set_requires_grad(self, requires_grad: bool) -> Result<Self, NeuraRustError> {
        if self.read_data().grad_fn.is_some() && requires_grad {
             return Err(NeuraRustError::RequiresGradOnNonLeaf);
        }
        self.write_data().requires_grad = requires_grad;
        Ok(self) 
    }

    /// Sets the `requires_grad` status of this tensor **in-place**.
    /// Only allowed on leaf tensors.
    pub fn requires_grad_(&self, requires_grad: bool) -> Result<(), NeuraRustError> {
        let mut guard = self.write_data(); 
        if guard.grad_fn.is_some() {
            return Err(NeuraRustError::RequiresGradOnNonLeaf);
        }
        guard.requires_grad = requires_grad;
        Ok(())
    }

    /// Returns an optional reference to the gradient function (`BackwardOp`) node.
    pub fn grad_fn(&self) -> Option<Arc<dyn BackwardOp + Send + Sync>> {
        self.read_data().grad_fn.clone()
    }

    /// Returns an optional reference to the gradient tensor.
    pub fn grad(&self) -> Option<Tensor> {
        self.read_data().grad.clone()
    }

    /// Sets the gradient tensor.
    pub(crate) fn set_grad(&self, grad_tensor: Option<Tensor>) -> Result<(), NeuraRustError> {
        self.write_data().grad = grad_tensor;
        Ok(())
    }

     /// Creates a new tensor that shares the same data but is detached
     /// from the computation graph.
     pub fn detach(&self) -> Tensor {
         let guard = self.read_data();
         let detached_data = TensorData {
             buffer: Arc::clone(&guard.buffer), 
             device: guard.device,
             dtype: guard.dtype,
             shape: guard.shape.clone(),
             strides: guard.strides.clone(),
             offset: guard.offset,
             requires_grad: false, 
             grad: None,           
             grad_fn: None,        
         };
         Tensor {
             data: Arc::new(RwLock::new(detached_data)),
         }
     }

    /// Computes the gradients of this tensor w.r.t. graph leaves.
    pub fn backward(&self) -> Result<(), NeuraRustError> {
        let guard = self.read_data();

        if !guard.requires_grad {
            return Err(NeuraRustError::RequiresGradNotMet);
        }
        if guard.numel() != 1 {
            return Err(NeuraRustError::BackwardNonScalar);
        }
        if guard.grad_fn.is_none() {
             log::debug!("backward() called on a leaf tensor. No operation to perform.");
            return Ok(());
        }

        // Create initial gradient = 1.0
        // This is inefficient, ideally use ones_like or similar
        let initial_grad_data: Box<dyn std::any::Any> = match guard.dtype {
             DType::F32 => Box::new(vec![1.0f32]),
             DType::F64 => Box::new(vec![1.0f64]),
         };
         let initial_grad_shape = guard.shape.clone(); 
         let initial_grad = match guard.dtype {
             DType::F32 => Tensor::new(
                 *initial_grad_data.downcast::<Vec<f32>>().unwrap(), initial_grad_shape
             )?,
             DType::F64 => Tensor::new_f64(
                 *initial_grad_data.downcast::<Vec<f64>>().unwrap(), initial_grad_shape
             )?,
         };

        let mut graph = ComputationGraph::new();
        graph.backward(self.clone(), initial_grad)?;

        Ok(())
    }

    /// Resets the gradient of this tensor to None.
    pub fn zero_grad(&self) {
        let mut guard = self.write_data();
        guard.grad = None;
    }
}

// Manual implementation of Clone trait since derive was removed
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data), // Clone the Arc, not the TensorData
        }
    }
}