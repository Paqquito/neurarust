use crate::autograd::graph::{topological_sort, NodeId};
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Checks if this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.data.read().unwrap().requires_grad
    }

    /// Sets the `requires_grad` flag for this tensor.
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<(), NeuraRustError> {
        let mut guard = self.data.write().map_err(|_| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: "Failed to lock TensorData for set_requires_grad".to_string(),
        })?;
        if requires_grad && guard.grad_fn.is_some() {
            eprintln!("Warning: Setting requires_grad=true on a non-leaf tensor. Gradients will not accumulate here during backward(). Did you mean to use .detach()?");
        }
        guard.requires_grad = requires_grad;
        Ok(())
    }

    /// Returns a clone of the gradient tensor, if it exists.
    pub fn grad(&self) -> Option<Tensor> {
        self.data.read().unwrap().grad.clone()
    }

    /// Returns a clone of the `Arc` pointing to the backward operation node (`grad_fn`).
    pub fn grad_fn(&self) -> Option<Arc<dyn BackwardOp + Send + Sync>> {
        self.data.read().unwrap().grad_fn.clone()
    }

    /// Sets the backward operation node (`grad_fn`) for this tensor.
    pub fn set_grad_fn(
        &self,
        grad_fn: Option<Arc<dyn BackwardOp + Send + Sync>>,
    ) -> Result<(), NeuraRustError> {
        let mut guard = self.data.write().map_err(|_| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: "Failed to lock TensorData for set_grad_fn".to_string(),
        })?;
        guard.grad_fn = grad_fn;
        Ok(())
    }

    /// Accumulates the given gradient into the tensor's `grad` field.
    pub fn acc_grad(&self, grad_to_add: Tensor) -> Result<(), NeuraRustError> {
        let mut guard = self.data.write().map_err(|_| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: "Failed to lock TensorData for acc_grad".to_string(),
        })?;
        let self_device = guard.device;
        let self_dtype = guard.dtype;

        let grad_to_add_device = grad_to_add.device();
        if self_device != grad_to_add_device {
            return Err(NeuraRustError::DeviceMismatch {
                expected: self_device,
                actual: grad_to_add_device,
                operation: "acc_grad".to_string(),
            });
        }
        let grad_to_add_dtype = grad_to_add.dtype();
        if self_dtype != grad_to_add_dtype {
             return Err(NeuraRustError::UnsupportedOperation(format!(
                 "acc_grad dtype mismatch: self={:?}, grad={:?}",
                 self_dtype, grad_to_add_dtype
             )));
        }

        match guard.grad.take() {
            Some(existing_grad) => {
                let existing_shape = existing_grad.shape();
                let grad_to_add_shape = grad_to_add.shape();
                if existing_shape != grad_to_add_shape {
                    return Err(NeuraRustError::GradientAccumulationShapeMismatch {
                        expected: existing_shape,
                        actual: grad_to_add_shape,
                    });
                }
                let sum_grad = crate::ops::arithmetic::add_op(&existing_grad, &grad_to_add)?;
                guard.grad = Some(sum_grad);
            }
            None => {
                let self_shape = guard.shape.clone();
                let grad_to_add_shape = grad_to_add.shape();
                if self_shape != grad_to_add_shape {
                     return Err(NeuraRustError::GradientAccumulationShapeMismatch {
                        expected: self_shape,
                        actual: grad_to_add_shape,
                    });
                }
                guard.grad = Some(grad_to_add);
            }
        }
        Ok(())
    }

    /// Performs the backward pass starting from this tensor.
    ///
    /// Computes the gradient of this tensor with respect to graph leaves.
    /// The graph is differentiated using the chain rule.
    ///
    /// # Arguments
    /// * `gradient`: Optional gradient tensor to use as the initial gradient for this tensor.
    ///               If `None`, it defaults to a tensor containing `1.0` if this tensor is
    ///               a scalar (0-dimensional or 1 element), or if the tensor has 0 elements.
    ///               Otherwise, it returns a `BackwardNonScalar` error.
    /// * `retain_graph`: If `false` (default), the graph used to compute grads will be freed.
    ///                   Set to `true` if you need to backward through the graph again.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// * `gradient` is provided but has the wrong shape or device.
    /// * `gradient` is `None`, but the tensor is not a scalar or empty.
    /// * The tensor does not require gradients.
    /// * An error occurs during graph traversal or gradient computation.
    pub fn backward(&self, gradient: Option<Tensor>) -> Result<(), NeuraRustError> {
        if !self.requires_grad() {
            return Ok(());
        }

        let grad_init = match gradient {
            Some(g) => {
                let self_shape = self.shape();
                let g_shape = g.shape();
                if g_shape != self_shape {
                    return Err(NeuraRustError::BackwardError(format!(
                        "Gradient shape mismatch: expected {:?}, got {:?}",
                        self_shape, g_shape
                    )));
                }
                let self_device = self.device();
                let g_device = g.device();
                if g_device != self_device {
                    return Err(NeuraRustError::BackwardError(format!(
                        "Gradient device mismatch: expected {:?}, got {:?}",
                        self_device, g_device
                    )));
                }
                g
            }
            None => {
                let s_shape = self.shape();
                let numel = self.numel();
                let is_scalar_like = s_shape.is_empty() || numel <= 1;
                let is_empty = numel == 0;

                if is_scalar_like {
                    todo!("Create scalar tensor with value 1.0 of correct dtype/device");
                } else if is_empty {
                    todo!("Implement Tensor::zeros_like or equivalent");
                } else {
                    return Err(NeuraRustError::BackwardNonScalar);
                }
            }
        };

        let node_id_type_alias: NodeId = Arc::as_ptr(&self.data);
        let mut grad_map: HashMap<NodeId, Tensor> = HashMap::new();
        grad_map.insert(node_id_type_alias, grad_init);

        let sorted_nodes = topological_sort(node_id_type_alias)?;

        for node_id in sorted_nodes {
            if let Some(accumulated_grad) = grad_map.remove(&node_id) {
                let tensor_data_ref = unsafe { &*node_id };
                let guard = tensor_data_ref.read().map_err(|_| {
                    NeuraRustError::BackwardError("Failed to lock during backward".to_string())
                })?;

                if let Some(op) = guard.grad_fn.as_ref() {
                     let grad_fn_clone = Arc::clone(op);
                     let requires_grad_check = guard.requires_grad;
                     drop(guard);

                     if requires_grad_check {
                        let input_grads = grad_fn_clone.backward(&accumulated_grad)?;
                        let input_ids = grad_fn_clone.inputs();

                        if input_grads.len() != input_ids.len() {
                            return Err(NeuraRustError::BackwardError(format!(
                                "BackwardOp mismatch: {} grads vs {} inputs (op: {:?})",
                                input_grads.len(), input_ids.len(), grad_fn_clone
                            )));
                        }

                        for (input_node_id, grad_to_add) in input_ids.into_iter().zip(input_grads) {
                             Self::accumulate_grad_static(input_node_id, grad_to_add)?;
                        }
                     }
                }
            }
        }
        Ok(())
    }

    fn accumulate_grad_static(
        tensor_data_ptr: *const RwLock<TensorData>,
        grad_to_add: Tensor,
    ) -> Result<(), NeuraRustError> {
         if tensor_data_ptr.is_null() {
             return Err(NeuraRustError::InternalError("Null pointer encountered in accumulate_grad_static".to_string()));
         }
         let tensor_data_lock = unsafe { &*tensor_data_ptr };

         let mut guard = tensor_data_lock.write().map_err(|_| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: "Failed to lock TensorData for accumulate_grad_static".to_string(),
         })?;

         if !guard.requires_grad {
             return Ok(());
         }

         let self_device = guard.device;
         let self_dtype = guard.dtype;

         let grad_to_add_device = grad_to_add.device();
         if self_device != grad_to_add_device {
             return Err(NeuraRustError::DeviceMismatch {
                 expected: self_device,
                 actual: grad_to_add_device,
                 operation: "accumulate_grad_static".to_string(),
             });
         }
         let grad_to_add_dtype = grad_to_add.dtype();
         if self_dtype != grad_to_add_dtype {
             return Err(NeuraRustError::UnsupportedOperation(format!(
                 "accumulate_grad_static dtype mismatch: self={:?}, grad={:?}",
                 self_dtype, grad_to_add_dtype
             )));
         }

         match guard.grad.take() {
             Some(existing_grad) => {
                 let existing_shape = existing_grad.shape();
                 let grad_to_add_shape = grad_to_add.shape();
                 if existing_shape != grad_to_add_shape {
                     guard.grad = Some(existing_grad);
                     return Err(NeuraRustError::GradientAccumulationShapeMismatch {
                         expected: existing_shape,
                         actual: grad_to_add_shape,
                     });
                 }
                 let sum_grad = crate::ops::arithmetic::add_op(&existing_grad, &grad_to_add)?;
                 guard.grad = Some(sum_grad);
             }
             None => {
                 let self_shape = guard.shape.clone();
                 let grad_to_add_shape = grad_to_add.shape();
                 if self_shape != grad_to_add_shape {
                     return Err(NeuraRustError::GradientAccumulationShapeMismatch {
                         expected: self_shape,
                         actual: grad_to_add_shape,
                     });
                 }
                 guard.grad = Some(grad_to_add);
             }
         }
         Ok(())
    }

    /// Creates a new tensor that shares the same underlying data buffer
    /// but is detached from the current computation graph.
    ///
    /// The returned tensor will have `requires_grad = false` and `grad_fn = None`.
    /// Changes to the data in the original tensor will be reflected in the detached tensor,
    /// and vice-versa (as they share the same buffer).
    pub fn detach(&self) -> Tensor {
        let old_data_guard = self.read_data();
        // Clone the necessary data, excluding autograd info
        let new_td = TensorData {
            buffer: old_data_guard.buffer.clone(), // Clone the Arc<Buffer>
            shape: old_data_guard.shape.clone(),
            strides: old_data_guard.strides.clone(),
            dtype: old_data_guard.dtype,
            device: old_data_guard.device,
            offset: old_data_guard.offset,
            requires_grad: false, // Detached -> no grad requirement
            grad: None,           // Detached -> no grad
            grad_fn: None,        // Detached -> no grad function
             // Retain other fields like maybe a name? Add if necessary.
        };
        drop(old_data_guard);

        Tensor {
            data: Arc::new(RwLock::new(new_td)),
        }
    }
}
