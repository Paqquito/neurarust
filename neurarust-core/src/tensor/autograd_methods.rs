use crate::autograd::graph::{topological_sort, NodeId};
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// This `impl` block provides methods related to automatic differentiation (autograd).
impl Tensor {
    /// Checks if this tensor requires gradient computation.
    ///
    /// Tensors that require gradients track their history and accumulate gradients
    /// during the `backward` pass. Leaf tensors created by the user typically have
    /// this set explicitly, while tensors resulting from operations inherit this status
    /// based on their inputs.
    /// Acquires a read lock internally.
    pub fn requires_grad(&self) -> bool {
        self.data.read().unwrap().requires_grad
    }

    /// Sets the `requires_grad` flag for this tensor.
    ///
    /// **Caution:** Modifying this flag directly can have implications for gradient computation.
    /// - Setting `requires_grad = true` on a tensor that resulted from an operation
    ///   (i.e., a non-leaf tensor with a `grad_fn`) is usually **not** what you want.
    ///   Gradients are only accumulated in leaf nodes during the backward pass.
    ///   If you need to perform operations on such a tensor without tracking gradients,
    ///   consider using [`detach()`](#method.detach) first.
    ///   This method will print a warning if you set `requires_grad = true` on a non-leaf tensor.
    /// - Setting `requires_grad = false` prevents the tensor from tracking history or accumulating gradients.
    ///
    /// Acquires a write lock internally.
    ///
    /// # Arguments
    /// * `requires_grad`: The boolean value to set the flag to.
    ///
    /// # Errors
    /// Returns `NeuraRustError::LockError` if the internal lock cannot be acquired.
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

    /// Returns a clone of the gradient tensor (`.grad`), if it exists.
    ///
    /// The gradient is accumulated in this field during the `backward()` pass
    /// **only** if the tensor is a leaf node (`grad_fn()` is `None`) and
    /// `requires_grad()` is `true`.
    /// The gradient tensor will have the same shape and device as the original tensor.
    /// Acquires a read lock internally.
    pub fn grad(&self) -> Option<Tensor> {
        self.data.read().unwrap().grad.clone()
    }

    /// Returns a clone of the `Arc` pointing to the backward operation (`grad_fn`) that produced this tensor.
    ///
    /// If this tensor is a leaf node (created by the user), `grad_fn` will be `None`.
    /// If it resulted from an operation involving tensors that require gradients,
    /// `grad_fn` will hold a reference to the corresponding backward operation object.
    /// This forms the basis of the computation graph used by `backward()`.
    /// Acquires a read lock internally.
    pub fn grad_fn(&self) -> Option<Arc<dyn BackwardOp + Send + Sync>> {
        self.data.read().unwrap().grad_fn.clone()
    }

    /// Sets the backward operation node (`grad_fn`) for this tensor.
    ///
    /// **Internal Use:** This method is primarily used internally by tensor operations
    /// to build the computation graph when operating on tensors that require gradients.
    /// Manually setting this is usually not necessary and potentially error-prone.
    /// Acquires a write lock internally.
    ///
    /// # Arguments
    /// * `grad_fn`: An `Option` containing an `Arc` to the backward operation.
    ///
    /// # Errors
    /// Returns `NeuraRustError::LockError` if the internal lock cannot be acquired.
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

    /// Accumulates the given gradient into the tensor's `.grad` field.
    ///
    /// If the tensor's current `.grad` is `None`, it is initialized with `grad_to_add`.
    /// If it already exists, `grad_to_add` is added to the existing gradient.
    /// This method is used internally during the `backward` pass.
    /// Acquires a write lock internally.
    ///
    /// # Arguments
    /// * `grad_to_add`: The gradient tensor to accumulate.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The internal lock cannot be acquired (`LockError`).
    /// - `grad_to_add` has a different device or data type (`DeviceMismatch`, `UnsupportedOperation`).
    /// - `grad_to_add` has a different shape than the tensor or its existing gradient (`GradientAccumulationShapeMismatch`).
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

    /// Performs the backward pass (automatic differentiation) starting from this tensor.
    ///
    /// Computes the gradient of this tensor with respect to all leaf nodes in the
    /// computation graph that have `requires_grad = true`.
    /// The gradients are accumulated in the `.grad` field of the respective leaf tensors.
    ///
    /// The computation graph is traversed using reverse-mode automatic differentiation (backpropagation).
    ///
    /// # Arguments
    /// * `gradient`: An optional `Tensor` representing the initial gradient (dL/dSelf) to start the chain rule.
    ///   - If `None`: Defaults to a tensor of ones with the same shape, dtype, and device as `self`.
    ///     This is typically used when calling `backward` on a scalar loss tensor.
    ///     **TODO:** Currently panics if `self` is not scalar-like and `gradient` is `None`.
    ///   - If `Some(g)`: The tensor `g` is used as the initial gradient. It must have the same shape
    ///     and device as `self`.
    /// * `retain_graph`: **(Not yet implemented)** If `false` (default), the computation graph might be freed
    ///    after the backward pass to save memory. If `true`, the graph is kept, allowing for multiple
    ///    backward passes (e.g., for calculating higher-order derivatives), but consuming more memory.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - This tensor does not require gradients (`self.requires_grad()` is `false`).
    /// - The provided `gradient` (if `Some`) has a mismatched shape or device.
    /// - `gradient` is `None`, but `self` is not scalar-like (non-scalar backward without initial grad).
    /// - An error occurs during graph traversal (e.g., cycle detection, lock errors).
    /// - An error occurs during the `backward` call of an operation node in the graph.
    /// - A shape mismatch occurs during gradient accumulation.
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
                    // TODO: Replace with Tensor::ones_like(self) or similar factory function
                    todo!("Create scalar tensor with value 1.0 of correct dtype/device");
                } else if is_empty {
                     // TODO: Replace with Tensor::zeros_like(self) or similar factory function
                     todo!("Create empty tensor with correct shape/dtype/device");
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

    /// Static helper function to accumulate gradient into a TensorData identified by its raw pointer.
    /// This is used internally by `backward` to avoid holding `Tensor` instances (and their Arcs)
    /// during graph traversal, which could prevent parts of the graph from being dropped.
    fn accumulate_grad_static(
        tensor_data_ptr: *const RwLock<TensorData>,
        grad_to_add: Tensor,
    ) -> Result<(), NeuraRustError> {
        // Safety: The pointer must be valid and point to a live TensorData's RwLock.
        // This is ensured by the topological sort and the way nodes are handled in `backward`.
        let tensor_data_ref = unsafe { &*tensor_data_ptr };
        let mut guard = tensor_data_ref.write().map_err(|_| NeuraRustError::LockError {
            lock_type: "write".to_string(),
            reason: "Failed to lock TensorData (static) for acc_grad".to_string(),
        })?;

        // Only accumulate if it requires grad and is a leaf
        if !guard.requires_grad || guard.grad_fn.is_some() {
            return Ok(());
        }

        let self_device = guard.device;
        let self_dtype = guard.dtype;
        let self_shape = guard.shape.clone(); // Clone shape before potentially taking grad

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
                 "acc_grad static dtype mismatch: self={:?}, grad={:?}",
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

    /// Creates a new tensor that shares the same underlying data but is detached
    /// from the computation graph.
    ///
    /// The returned tensor will have `requires_grad = false` and `grad_fn = None`.
    /// Any changes to the data of the original tensor will be reflected in the detached tensor
    /// (and vice-versa), but gradient computation history is severed.
    ///
    /// This is useful when you want to use a tensor's value in a context where gradients
    /// are not needed or should not propagate (e.g., updating model weights, certain types
    /// of evaluation metrics).
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// use neurarust_core::ops::arithmetic::add_op;
    ///
    /// let a = Tensor::new(vec![2.0f32], vec![]).unwrap();
    /// a.set_requires_grad(true).unwrap();
    /// let b = Tensor::new(vec![3.0f32], vec![]).unwrap();
    /// b.set_requires_grad(true).unwrap();
    ///
    /// let c = add_op(&a, &b).unwrap(); // c requires grad and has a grad_fn
    /// assert!(c.requires_grad());
    /// assert!(c.grad_fn().is_some());
    ///
    /// let d = c.detach(); // d shares data with c
    /// assert!(!d.requires_grad());
    /// assert!(d.grad_fn().is_none());
    /// assert_eq!(c.item_f32().unwrap(), d.item_f32().unwrap());
    /// ```
    pub fn detach(&self) -> Tensor {
        let current_guard = self.read_data();
        // Create a new TensorData with copied metadata but no autograd history
        let detached_td = TensorData {
            buffer: Arc::clone(&current_guard.buffer), // Share the same buffer
            shape: current_guard.shape.clone(),
            strides: current_guard.strides.clone(),
            offset: current_guard.offset,
            device: current_guard.device,
            dtype: current_guard.dtype,
            requires_grad: false, // Key difference: requires_grad is false
            grad: None,           // Key difference: grad is None
            grad_fn: None,        // Key difference: grad_fn is None
        };
        drop(current_guard); // Release the lock on the original tensor

        // Create a new Tensor struct wrapping the new TensorData
        Tensor {
            data: Arc::new(RwLock::new(detached_td))
        }
    }

    /// Clears the gradient (`.grad` field) of this tensor by setting it to `None`.
    ///
    /// Acquires a write lock internally.
    pub fn clear_grad(&self) {
        if let Ok(mut guard) = self.data.write() {
            guard.grad = None;
        }
        // Silently ignore lock errors, as clearing grad is often best-effort
    }
}
