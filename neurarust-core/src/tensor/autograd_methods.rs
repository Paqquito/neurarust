use crate::autograd::graph::{topological_sort, NodeId};
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::Copy;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use std::sync::RwLock;
use crate::buffer::Buffer;

// Note: T bounds for the impl block cover all methods inside
impl<
        T: 'static
            + Debug
            + Copy
            + Zero
            + One
            + Add<Output = T>
            + AddAssign
            + Sum
            + PartialEq
            + Default
            + Send
            + Sync
            + PartialOrd,
    > Tensor<T>
{
    /// Checks if this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.read_data().requires_grad
    }

    /// Sets the `requires_grad` flag for this tensor.
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<(), NeuraRustError> {
        let mut guard = self.write_data();
        if requires_grad && guard.grad_fn.is_some() {
            eprintln!("Warning: Setting requires_grad=true on a non-leaf tensor. Gradients will not accumulate here during backward(). Did you mean to use .detach()?");
        }
        guard.requires_grad = requires_grad;
        Ok(())
    }

    /// Returns a clone of the gradient tensor, if it exists.
    pub fn grad(&self) -> Option<Tensor<T>> {
        self.read_data().grad.clone()
    }

    /// Accumulates the given gradient into the tensor's `grad` field.
    pub fn acc_grad(&self, grad_to_add: Tensor<T>) -> Result<(), NeuraRustError>
    where
        T: Add<Output = T> + AddAssign + Zero + One + Sum + PartialEq + Default + Send + Sync,
    {
        let mut guard = self.write_data();

        if guard.device != grad_to_add.device() {
            return Err(NeuraRustError::DeviceMismatch {
                expected: guard.device,
                actual: grad_to_add.device(),
                operation: "acc_grad".to_string(),
            });
        }

        match guard.grad.take() {
            Some(existing_grad) => {
                let sum_grad = crate::ops::arithmetic::add::add_op(&existing_grad, &grad_to_add)?;
                guard.grad = Some(sum_grad);
            }
            None => {
                guard.grad = Some(grad_to_add.clone());
            }
        }
        Ok(())
    }

    /// Returns a clone of the `Arc` pointing to the backward operation node (`grad_fn`).
    pub fn grad_fn(&self) -> Option<Arc<dyn BackwardOp<T> + Send + Sync>> {
        self.read_data().grad_fn.clone()
    }

    /// Sets the backward operation node (`grad_fn`) for this tensor.
    pub fn set_grad_fn(
        &self,
        grad_fn: Option<Arc<dyn BackwardOp<T> + Send + Sync>>,
    ) -> Result<(), NeuraRustError> {
        let mut guard = self.write_data();
        guard.grad_fn = grad_fn;
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
    pub fn backward(&self, gradient: Option<Tensor<T>>) -> Result<(), NeuraRustError> {
        // TODO: Add retain_graph functionality
        if !self.requires_grad() {
            // Optionally return Ok(()) or an specific error/warning
            // Return Ok for now, as calling backward on non-requiring tensor is a no-op.
            return Ok(());
            // return Err(NeuraRustError::BackwardError("Cannot call backward on tensor that does not require grad".to_string()));
        }

        // Determine initial gradient (dL/dself)
        let grad_init = match gradient {
            Some(g) => {
                // Validate provided gradient
                if g.shape() != self.shape() {
                    return Err(NeuraRustError::BackwardError(format!(
                        "Gradient shape mismatch: expected {:?}, got {:?}",
                        self.shape(),
                        g.shape()
                    )));
                }
                if g.device() != self.device() {
                    return Err(NeuraRustError::BackwardError(format!(
                        "Gradient device mismatch: expected {:?}, got {:?}",
                        self.device(),
                        g.device()
                    )));
                }
                g
            }
            None => {
                let s_shape = self.shape();
                let numel = s_shape.iter().product::<usize>();
                let is_scalar_like = s_shape.is_empty() || numel <= 1;
                let is_empty = numel == 0;

                if is_scalar_like {
                    // Create a scalar tensor with value One<T>
                    Tensor::<T>::ones(vec![])? // Assumes Tensor::ones exists and works for scalar
                } else if is_empty {
                    // If empty but not scalar, create a zero tensor of the same shape.
                    // This gradient won't actually be used in calculations but might be needed
                    // to satisfy types in the backward pass structure.
                    Tensor::<T>::zeros_like(self)? // Assumes Tensor::zeros_like exists
                } else {
                    return Err(NeuraRustError::BackwardNonScalar);
                }
            }
        };

        // Use a map to accumulate gradients for each node ID
        // Key: NodeId (*const RwLock<TensorData<T>>), Value: Accumulated Gradient Tensor
        let mut grad_map: HashMap<NodeId<T>, Tensor<T>> = HashMap::new();
        grad_map.insert(self.get_node_id(), grad_init);

        // Perform topological sort starting from the current tensor's NodeId
        let sorted_nodes = topological_sort(self.get_node_id())?; // Pass NodeId

        // Iterate through sorted nodes in reverse topological order
        for node_id in sorted_nodes {
            // Retrieve the accumulated gradient for the current node
            // If a node isn't in grad_map, it means no gradient flowed back to it (or it's the start node)
            if let Some(accumulated_grad) = grad_map.remove(&node_id) {
                // Check if the node requires grad. We only need to call backward if its inputs might require grad.
                // Get the TensorData reference from the raw pointer (REQUIRES CAREFUL HANDLING)
                let tensor_data_ref = unsafe { &*node_id }; // De-reference the raw pointer
                let guard = tensor_data_ref.read().map_err(|_| {
                    NeuraRustError::BackwardError(
                        "Failed to acquire read lock during backward traversal".to_string(),
                    )
                })?;

                // If this node has a grad_fn, call its backward method
                if let Some(op) = guard.grad_fn.as_ref() {
                     // Check if node requires grad - conceptually needed, but grad_fn implies it?
                     // if !guard.requires_grad { continue; } // Skip if node didn't require grad?

                    // Release the read guard before calling op.backward to avoid deadlock
                    // if op.backward needs to access this node's data again (unlikely but possible).
                     let grad_fn_clone = Arc::clone(op);
                     drop(guard);

                    // Call the backward method of the operation
                    let input_grads = grad_fn_clone.backward(&accumulated_grad)?;
                    let input_ids = grad_fn_clone.inputs();

                    if input_grads.len() != input_ids.len() {
                        return Err(NeuraRustError::BackwardError(format!(
                            "BackwardOp returned {} gradients, but expected {} (for op: {:?})",
                            input_grads.len(),
                            input_ids.len(),
                            grad_fn_clone // Use the cloned Arc for Debug print
                        )));
                    }

                    // Accumulate gradients for the inputs of this operation
                    for (input_node_id, grad_to_add) in input_ids.into_iter().zip(input_grads) {
                        // Get the input TensorData reference (unsafe)
                        let input_tensor_data_ref = unsafe { &*input_node_id };

                        // Check if the input tensor requires gradient accumulation
                        let input_guard = input_tensor_data_ref.read().map_err(|_| NeuraRustError::BackwardError(
                            "Failed to acquire read lock on input node during backward".to_string(),
                        ))?;

                        if input_guard.requires_grad {
                            // Accumulate gradient (handles device checks and None case internally)
                            // Need a way to get a Tensor instance from the NodeId/RwLock ptr
                            // This is tricky. The graph traversal only gives us IDs.
                            // We need the actual Tensor objects to call acc_grad.

                            // Option 1: Pass Tensors along with IDs in graph traversal (complex).
                            // Option 2: Have a global map from NodeId back to Weak<Tensor> or similar?
                            // Option 3: Modify acc_grad to operate directly on TensorData under lock?

                            // Let's assume for now we can get the Tensor back somehow or modify acc_grad.
                            // **Placeholder:** Need a mechanism here.
                            // If we modify acc_grad, it needs the TensorData RwLock and the grad_to_add.
                            // Let's try modifying acc_grad.
                            drop(input_guard);
                            Self::accumulate_grad_static(input_tensor_data_ref, grad_to_add)?;

                        }
                    }
                }
                // If node is a leaf (no grad_fn) or retain_graph is false, clear grad_fn?
                // TODO: Handle retain_graph = false (clear grad_fn and maybe intermediate grads)
            }
        }

        Ok(())
    }

     // Static helper potentially called by backward
     // Takes RwLock<TensorData> to avoid needing Tensor instance from NodeId
     fn accumulate_grad_static(
         tensor_data_lock: &RwLock<TensorData<T>>,
         grad_to_add: Tensor<T>,
     ) -> Result<(), NeuraRustError> {
         // Need AddAssign bound for element-wise addition below
         // Note: T already has AddAssign bound from the main impl block

         let mut guard = tensor_data_lock.write().map_err(|_| {
             NeuraRustError::BackwardError("Failed to acquire write lock for grad accumulation".to_string())
         })?;

         // --- Device Check ---
         // Gradient to add must be on the same device as the tensor receiving it.
         if grad_to_add.device() != guard.device {
             return Err(NeuraRustError::DeviceMismatch {
                 expected: guard.device,
                 actual: grad_to_add.device(),
                 operation: "acc_grad".to_string(),
             });
         }

         // --- Shape Check ---
         // Check if shapes match before attempting accumulation.
         let expected_shape = guard.shape.clone(); // Clone shape for potential use in new Tensor
         if grad_to_add.shape() != expected_shape {
             return Err(NeuraRustError::ShapeMismatch{
                 expected: expected_shape.clone(),
                 actual: grad_to_add.shape(),
                 operation: "acc_grad".to_string(),
             });
         }

         // --- Accumulation Logic (Clone & Replace approach) ---
         if let Some(existing_grad) = guard.grad.as_ref() { // Borrow immutably first

            // Need Add trait bound for element-wise addition
            // Note: T already has Add bound from the main impl block

             // Read guards for tensor data (gradient tensors)
             let existing_grad_guard = existing_grad.read_data();
             let grad_to_add_guard = grad_to_add.read_data();

             // --- Device Specific Accumulation (CPU for now) ---
             match (existing_grad_guard.data.as_ref(), grad_to_add_guard.data.as_ref()) {
                 (Buffer::Cpu(existing_arc), Buffer::Cpu(to_add_arc)) => {
                     let existing_data = existing_arc.as_slice();
                     let to_add_data = to_add_arc.as_slice();
                     let numel = expected_shape.iter().product(); // Use shape from receiving tensor

                     if existing_data.len() != numel || to_add_data.len() != numel {
                         return Err(NeuraRustError::InternalError(
                             "Gradient buffer length mismatch despite shape match in acc_grad".to_string()
                         ));
                     }

                     // Create new buffer for the sum
                     let mut sum_data = Vec::with_capacity(numel);

                     // Perform element-wise addition into the new buffer
                     // Requires T: Add<Output = T> + Copy
                     for i in 0..numel {
                        // Handle potential strides/offsets IF gradients could be views.
                        // Assuming gradient tensors are always contiguous for now.
                        // If not, we'd need existing_grad.get(coords) + grad_to_add.get(coords) logic.
                        // For simplicity, assume flat buffer access corresponds to logical elements.
                         sum_data.push(existing_data[i] + to_add_data[i]);
                     }

                     // Create a new gradient tensor from the summed data
                     // Use the shape of the tensor receiving the gradient
                     let new_grad_tensor = Tensor::new(sum_data, expected_shape)?;

                     // Replace the old gradient tensor in the guard
                     // Need mutable guard again, release read guards first
                     drop(existing_grad_guard);
                     drop(grad_to_add_guard);
                     guard.grad = Some(new_grad_tensor); // guard is mut TensorData guard
                 }
                 (Buffer::Gpu { .. }, Buffer::Gpu { .. }) => {
                     // TODO: Implement GPU gradient accumulation (clone & replace)
                     return Err(NeuraRustError::UnsupportedOperation(
                        "GPU gradient accumulation not yet implemented".to_string(),
                    ));
                 }
                 _ => {
                     return Err(NeuraRustError::DeviceMismatch {
                         expected: guard.device,
                         actual: grad_to_add.device(),
                         operation: "acc_grad buffer type mismatch".to_string(),
                     });
                 }
             }
         } else {
             // If no existing gradient, just set it. Clone grad_to_add as guard takes ownership.
             guard.grad = Some(grad_to_add.clone());
         }
         Ok(())
     }

}
