use crate::autograd::backward_op::BackwardOp;
use crate::autograd::graph::{topological_sort, NodeId};
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::types::DType;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// #[cfg(test)]
// use crate::backend::cpu::CpuStorage; // Commented out as its path or existence is unclear

/// This `impl` block provides methods related to automatic differentiation (autograd).
/// It includes the core backward pass logic for gradient computation.
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
        let grad_init = match gradient {
            Some(ref g) => {
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
                g.clone()
            }
            None => {
                let s_shape = self.shape();
                let numel = self.numel();
                let is_scalar_like = s_shape.is_empty() || numel <= 1;
                let is_empty = numel == 0;

                if is_scalar_like {
                    let self_dtype = self.dtype();
                    match self_dtype {
                        DType::F32 => crate::tensor::create::full(&[], 1.0f32)?,
                        DType::F64 => crate::tensor::create::full_f64(&[], 1.0f64)?,
                        // TODO: Add other DType variants if necessary or return an error
                    }
                } else if is_empty {
                     // TODO: Replace with Tensor::zeros_like(self) or similar factory function
                     todo!("Create empty tensor with correct shape/dtype/device");
                } else {
                    return Err(NeuraRustError::BackwardNonScalar);
                }
            }
        };

        let start_node_id = Arc::as_ptr(&self.data);
        // println!("[BACKWARD_FN] Tensor.backward() called for start_node_id: {:?}, is_leaf: {}, with initial gradient: {:?}", start_node_id, self.grad_fn().is_none(), gradient.as_ref().map(|g| (g.shape(), g.dtype())));

        if !self.requires_grad() {
            // println!("[BACKWARD_FN] Tensor does not require grad. Skipping backward pass.");
            return Ok(());
        }

        let mut grad_map: HashMap<NodeId, Tensor> = HashMap::new();
        grad_map.insert(start_node_id, grad_init.clone());

        let mut sorted_nodes = topological_sort(start_node_id)?;
        // println!("[BACKWARD_FN] Topological sort completed. Node count: {}", sorted_nodes.len());

        // --- Key Correction for V5: Reverse the sorted nodes for backward pass ---
        sorted_nodes.reverse();
        // println!("[BACKWARD_FN] Reversed sorted_nodes for backward pass.");

        // --- Refactored Backward Loop V4 --- 
        for node_id in sorted_nodes {
            // println!("[BACKWARD_FN_V4] Processing node_id: {:?}", node_id);

            // Retrieve the gradient accumulated for this node so far from the map.
            // Use get() as we might need it for both accumulation and propagation.
            let grad_for_node_option = grad_map.get(&node_id).cloned(); // Clone Option<Tensor>

            // Access TensorData to check properties
            let tensor_data_ref = unsafe { &*node_id };
            let guard = tensor_data_ref.read().map_err(|_| {
                NeuraRustError::BackwardError("Failed to lock node during backward V4".to_string())
            })?;
            let requires_grad = guard.requires_grad;
            // let is_leaf = guard.grad_fn.is_none(); // is_leaf check removed from accumulation
            let grad_fn_option = guard.grad_fn.clone(); // Clone Option<Arc<...>>
            // println!("[BACKWARD_FN_V4] Node {:?}: requires_grad={}", node_id, requires_grad);
            drop(guard); // Release read lock

            // --- Accumulate gradient into .grad field if requires_grad AND gradient exists --- 
            if requires_grad {
                if let Some(current_grad) = grad_for_node_option.as_ref() { // Borrow the option
                    // println!("[BACKWARD_FN_V4] Node {:?} requires grad. Accumulating received grad (shape {:?}) into .grad field.", node_id, current_grad.shape());
                    Self::accumulate_grad_static(node_id, current_grad.clone())?; // Pass clone
                } else {
                     // println!("[BACKWARD_FN_V4] Node {:?} requires grad but received no grad via map for accumulation.", node_id);
                }
            }

            // --- Propagate gradient via grad_fn if it exists AND gradient exists --- 
            if let Some(grad_fn) = grad_fn_option {
                if let Some(gradient_to_propagate) = grad_for_node_option { // Take ownership of the Option's value
                     // println!("[BACKWARD_FN_V4] Node {:?} has grad_fn. Propagating using accumulated grad (shape {:?})", node_id, gradient_to_propagate.shape());
                     let input_ids = grad_fn.inputs();

                     // println!("[BACKWARD_FN_V4] Node {:?} ({}) - Calling grad_fn.backward()", node_id, op_debug_name);
                     let input_grads = grad_fn.backward(&gradient_to_propagate)?;
                     // println!("[BACKWARD_FN_V4] Node {:?} ({}) - grad_fn.backward() returned {} grads.", node_id, op_debug_name, input_grads.len());

                     if input_grads.len() != input_ids.len() {
                         return Err(NeuraRustError::BackwardError(format!(
                             "BackwardOp mismatch: {} grads vs {} inputs (op: {:?})",
                             input_grads.len(), input_ids.len(), grad_fn
                         )));
                     }

                     // Accumulate the calculated gradients into the grad_map for parent nodes
                     for (parent_node_id, grad_for_parent) in input_ids.into_iter().zip(input_grads) {
                         // println!("[BACKWARD_FN_V4]   -> Accumulating grad (shape {:?}) into grad_map for parent: {:?}\", grad_for_parent.shape(), parent_node_id);
                         if parent_node_id.is_null() {
                              eprintln!("Warning: grad_fn returned null parent_node_id. Skipping.");
                              continue;
                         }
                         // Corrected Logic for grad_map accumulation
                         if let Some(existing_grad) = grad_map.get_mut(&parent_node_id) {
                             // println!("[BACKWARD_FN_V4]     Parent {:?} already in map, adding gradients.", parent_node_id);
                             match crate::ops::arithmetic::add_op(existing_grad, &grad_for_parent) {
                                 Ok(sum_grad) => { *existing_grad = sum_grad; },
                                 Err(e) => return Err(NeuraRustError::BackwardError(format!("Failed to add gradients in grad_map for node {:?}: {}", parent_node_id, e))),
                             }
                         } else {
                             // println!("[BACKWARD_FN_V4]     Parent {:?} not in map, inserting clone.", parent_node_id);
                             grad_map.insert(parent_node_id, grad_for_parent.clone());
                         }
                     }
                } else {
                     // Node has grad_fn but no accumulated gradient reached it.
                     // println!("[BACKWARD_FN_V4] Node {:?} has grad_fn but no accumulated gradient to propagate.", node_id);
                }
            }
            // Remove the node's gradient from map after processing? Optional for memory, maybe not needed.
            // grad_map.remove(&node_id);
        }

        // println!("[BACKWARD_FN] Backward pass V4 completed.");
        Ok(())
    }

    /// Static helper function to accumulate gradient into a TensorData identified by its raw pointer.
    /// This function is responsible for adding the `grad_to_add` to the existing `.grad`
    /// field of the TensorData pointed to by `tensor_data_rwlock_ptr`.
    /// It handles cases where `.grad` is None (initializes it) or Some (adds to it).
    /// It also checks for shape compatibility.
    fn accumulate_grad_static(
        tensor_data_rwlock_ptr: *const RwLock<TensorData>,
        grad_to_add: Tensor,
    ) -> Result<(), NeuraRustError> {
        // println!("[ACC_GRAD_STATIC] Called for tensor_data_rwlock_ptr: {:?}, grad_to_add shape: {:?}, dtype: {:?}", tensor_data_rwlock_ptr, grad_to_add.shape(), grad_to_add.dtype());
        if tensor_data_rwlock_ptr.is_null() {
            return Err(NeuraRustError::BackwardError(
                "accumulate_grad_static called with null tensor_data_rwlock_ptr".to_string(),
            ));
        }

        // Obtain a write guard by locking the RwLock pointed to.
        // This is unsafe because we are dereferencing a raw pointer.
        // The caller (backward pass) must ensure this pointer is valid.
        let rwlock_ref = unsafe { &*tensor_data_rwlock_ptr };
        let mut guard = rwlock_ref.write().map_err(|_| {
            NeuraRustError::BackwardError(
                "Failed to lock TensorData (via RwLock ptr) in accumulate_grad_static".to_string(),
            )
        })?;
        // Now `guard` is an RwLockWriteGuard<'_, TensorData>, so we can use its DerefMut to TensorData.

        // println!("[ACC_GRAD_STATIC] Ptr {:?}: Current requires_grad={}, is_leaf={}, existing_grad.is_some()={}", tensor_data_rwlock_ptr, guard.requires_grad, guard.grad_fn.is_none(), guard.grad.is_some());

        if !guard.requires_grad { // Use the guard
            // println!("[ACC_GRAD_STATIC] Ptr {:?}: Skipping accumulation (requires_grad: {}).", tensor_data_rwlock_ptr, guard.requires_grad);
            return Ok(()); // Do not accumulate gradient if not required
        }

        let self_shape = guard.shape.clone(); 
        let grad_to_add_shape = grad_to_add.shape();

        if let Some(existing_grad) = guard.grad.as_mut() { // Use the guard
            let existing_shape = existing_grad.shape();
            if existing_shape != grad_to_add_shape {
                // println!("[ACC_GRAD_STATIC] Ptr {:?}: ERROR - Shape mismatch adding grad. Expected: {:?}, Got: {:?}\", tensor_data_rwlock_ptr, existing_shape, grad_to_add_shape); 
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("{:?}", existing_shape),
                    actual: format!("{:?}", grad_to_add_shape),
                    operation: format!("accumulate_grad_static (add to existing for node {:?})", tensor_data_rwlock_ptr),
                });
            }
            let sum_grad = crate::ops::arithmetic::add_op(existing_grad, &grad_to_add)?;
            *existing_grad = sum_grad;
            // println!("[ACC_GRAD_STATIC] Ptr {:?}: Successfully added to existing grad.", tensor_data_rwlock_ptr); 
        } else {
            if self_shape != grad_to_add_shape {
                // println!("[ACC_GRAD_STATIC] Ptr {:?}: ERROR - Shape mismatch for new grad. Expected: {:?}, Got: {:?}\", tensor_data_rwlock_ptr, self_shape, grad_to_add_shape); 
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("{:?}", self_shape),
                    actual: format!("{:?}", grad_to_add_shape),
                    operation: format!("accumulate_grad_static (set new for node {:?})", tensor_data_rwlock_ptr),
                });
            }
            guard.grad = Some(grad_to_add); // Use the guard
            // println!("[ACC_GRAD_STATIC] Ptr {:?}: Successfully set new grad.", tensor_data_rwlock_ptr); 
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
    ///
    /// let a = Tensor::new(vec![2.0f32, 5.0], vec![2]).unwrap(); // Make it non-scalar
    /// a.set_requires_grad(true).unwrap();
    ///
    /// // Use a public tensor method like sum() to create a grad_fn
    /// let c_res = a.sum(None::<Vec<usize>>.as_deref(), false); // Sum all elements
    /// assert!(c_res.is_ok());
    /// let c = c_res.unwrap();
    /// 
    /// assert!(c.requires_grad());
    /// assert!(c.grad_fn().is_some());
    ///
    /// let d = c.detach(); // d shares data with c
    /// assert!(!d.requires_grad());
    /// assert!(d.grad_fn().is_none());
    /// // Check value (sum is 7.0)
    /// assert!((c.item_f32().unwrap() - 7.0).abs() < 1e-6);
    /// assert!((d.item_f32().unwrap() - 7.0).abs() < 1e-6);
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

    /// Returns a raw pointer to the underlying `RwLock<TensorData>`.
    /// This pointer serves as a unique identifier for this tensor's data in the computation graph.
    /// 
    /// **Caution:** This is a low-level method. The pointer should only be used as an ID
    /// and not be dereferenced unsafely. The lifetime and validity are tied to the `Arc`
    /// managing the `TensorData`.
    pub fn node_id(&self) -> NodeId {
        Arc::as_ptr(&self.data)
    }
}

// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module (Tensor, NeuraRustError, etc.)
    // Import necessary ops for testing
    use crate::ops::arithmetic::add::add_op;

    #[test]
    fn test_detach_basic() {
        let t1_res = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(t1_res.is_ok());
        let t1 = t1_res.unwrap();
        t1.set_requires_grad(true).unwrap();
        // Use add_op to create a grad_fn
        // Note: Using add_op directly here is okay because this is an internal module test
        let t1_added = add_op(&t1, &t1);
        assert!(t1_added.is_ok());
        // Check the *result* of add_op for grad_fn
        let t1_added = t1_added.unwrap(); 
        assert!(t1_added.requires_grad(), "Result of add should require grad");
        assert!(t1_added.grad_fn().is_some(), "Result of add should have grad_fn");

        // Detach the result of the operation
        let t2 = t1_added.detach();

        // Check detached properties
        assert!(!t2.requires_grad(), "Detached tensor should not require grad");
        assert!(t2.grad_fn().is_none(), "Detached tensor should not have grad_fn");
        assert!(t2.grad().is_none(), "Detached tensor should not have grad");

        // Check metadata and data sharing
        assert_eq!(t1_added.shape(), t2.shape(), "Shapes should be equal");
        assert_eq!(t1_added.dtype(), t2.dtype(), "DTypes should be equal");
        assert_eq!(t1_added.device(), t2.device(), "Devices should be equal");
        assert_eq!(t1_added.strides(), t2.strides(), "Strides should be equal");
        
        // Verify data content equality
        let t1_added_data = t1_added.get_f32_data().unwrap();
        let t2_data = t2.get_f32_data().unwrap();
        assert_eq!(t1_added_data, t2_data, "Data content should be equal");
    }

    // TODO: Add more tests for autograd methods if needed
}
