use crate::autograd::graph::{topological_sort, NodeId};
use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::ops;
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::Copy;
use std::ops::{Add, AddAssign};
use std::sync::Arc;

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

    /// Performs the backward pass to compute gradients for tensors in the computation graph.
    ///
    /// Starts the gradient computation from this tensor. If this tensor is non-scalar,
    /// an initial gradient (`gradient`) must be provided. If it's scalar, the initial
    /// gradient defaults to 1.0 (or equivalent One for type T).
    ///
    /// # Arguments
    /// * `gradient`: An optional `Tensor` representing the initial gradient to propagate.
    ///               Required if `self` is not a scalar tensor. Must have the same shape
    ///               and be on the same device as `self`.
    /// * `retain_graph`: If `false` (default, not implemented yet), the computation graph
    ///                   is freed during the backward pass to save memory. If `true`,
    ///                   the graph is kept, allowing for subsequent backward passes (e.g., for higher-order gradients).
    ///
    /// # Returns
    /// * `Ok(())` if the backward pass is successful.
    /// * `Err(NeuraRustError)` if an error occurs (e.g., shape mismatch, device mismatch, cycle detected, requires_grad issues).
    pub fn backward(
        &self,
        gradient: Option<Tensor<T>>, /*, retain_graph: bool = false */
    ) -> Result<(), NeuraRustError> {
        // --- 1. Initial Checks ---
        if !self.requires_grad() {
            // Nothing to do if the starting tensor doesn't require grad
            return Ok(());
        }

        // --- 2. Determine Initial Gradient ---
        let initial_gradient = match gradient {
            Some(g) => {
                // Check shape and device if gradient is provided
                if g.shape() != self.shape() {
                    return Err(NeuraRustError::ShapeMismatch {
                        expected: self.shape(),
                        actual: g.shape(),
                        operation: "backward initial gradient".to_string(),
                    });
                }
                if g.device() != self.device() {
                    return Err(NeuraRustError::DeviceMismatch {
                        expected: self.device(),
                        actual: g.device(),
                        operation: "backward initial gradient".to_string(),
                    });
                }
                g
            }
            None => {
                // If no gradient provided, self must be scalar
                if self.numel() != 1 {
                    return Err(NeuraRustError::BackwardNonScalar);
                }
                // Create a scalar tensor with value One<T> on the same device
                // Assuming `Tensor::ones` works for scalar shape and only supports CPU for now.
                // Check device first
                if self.device() != StorageDevice::CPU {
                    return Err(NeuraRustError::UnsupportedOperation(
                        "Backward with implicit scalar gradient only supported on CPU currently"
                            .to_string(),
                    ));
                }
                Tensor::<T>::ones(vec![])? // Use `vec![]` for scalar shape
            }
        };

        // --- 3. Topological Sort ---
        let self_node_id: NodeId<T> = self.get_node_id(); // Need to implement this helper
        let sorted_nodes = topological_sort(self_node_id)?;

        // --- 4. Initialize Gradient Accumulation Map ---
        // NodeId -> Accumulated Gradient Tensor for that node
        let mut grad_map: HashMap<NodeId<T>, Tensor<T>> = HashMap::new();
        grad_map.insert(self_node_id, initial_gradient);

        // --- 5. Backward Pass Loop ---
        // Iterate through nodes in reverse topological order
        for node_id in sorted_nodes.iter().rev() {
            // Iterate from output towards inputs
            // Retrieve the accumulated gradient for the current node
            let current_grad = match grad_map.get(node_id) {
                Some(grad) => grad.clone(), // Clone the gradient Tensor for use
                None => continue, // Skip nodes that don't have accumulated gradients (e.g., not required grad)
            };

            // Access the node's TensorData to get the BackwardOp (grad_fn)
            // Unsafe block to dereference the raw pointer. This is safe because:
            // 1. The pointers come from `Arc::as_ptr` on Arcs that are kept alive by the `Tensor` instances involved
            //    in the graph, which must exist for the backward pass to be called.
            // 2. We only hold the read lock for a short duration.
            let grad_fn_option = unsafe {
                (*(*node_id))
                    .read()
                    .expect("RwLock poisoned during grad_fn read")
                    .grad_fn
                    .clone()
            };

            if let Some(backward_op) = grad_fn_option {
                // Calculate gradients for the inputs of this operation
                let input_grads = backward_op.backward(&current_grad)?;
                let parent_node_ids = backward_op.inputs();

                // Ensure the number of gradients matches the number of inputs
                if input_grads.len() != parent_node_ids.len() {
                    return Err(NeuraRustError::InternalError(format!(
                        "BackwardOp returned {} gradients, but expected {} inputs.",
                        input_grads.len(),
                        parent_node_ids.len()
                    )));
                }

                // Accumulate gradients for each parent node
                for (parent_id, grad_to_add) in parent_node_ids.iter().zip(input_grads) {
                    // Check if the parent requires gradient before accumulating
                    let parent_data = unsafe {
                        (*(*parent_id))
                            .read()
                            .expect("RwLock poisoned during parent requires_grad read")
                    };
                    if parent_data.requires_grad {
                        // Get or create the gradient entry in the map
                        let parent_shape = parent_data.shape.clone();
                        let parent_device = parent_data.device;
                        drop(parent_data); // Drop read lock before potentially locking again in entry().or_insert_with

                        let acc_grad_entry = grad_map.entry(*parent_id).or_insert_with(|| {
                            // If gradient doesn't exist, create a zero tensor
                            // Assuming `Tensor::zeros` only supports CPU for now.
                             // Check device first
                             if parent_device != StorageDevice::CPU {
                                 panic!("Attempted to create zero gradient on non-CPU device, which is unsupported."); // Panic for now
                                 // TODO: Return proper error
                                 // return Err(NeuraRustError::UnsupportedOperation(...))
                             }
                            Tensor::<T>::zeros(parent_shape).expect("Failed to create zero tensor for gradient accumulation")
                        });

                        // Perform device-aware addition using add_op
                        // Note: add_op creates a *new* tensor. We need to update the map entry.
                        // Check devices before adding
                        if acc_grad_entry.device() != grad_to_add.device() {
                            return Err(NeuraRustError::DeviceMismatch {
                                expected: acc_grad_entry.device(),
                                actual: grad_to_add.device(),
                                operation: "gradient accumulation".to_string(),
                            });
                        }
                        // TODO: Replace with an in-place add or update the map entry carefully.
                        // For now, create sum and update map.
                        let sum_grad = ops::arithmetic::add::add_op(acc_grad_entry, &grad_to_add)?;
                        *acc_grad_entry = sum_grad;
                    }
                }
            }
            // --- TODO: Handle retain_graph = false (clear grad_fn) ---
            // If retain_graph is false (default), we might want to clear the grad_fn
            // after processing the node to free the graph structure.
            // This requires a write lock.
            // if !retain_graph {
            //     let mut node_data = unsafe { (*(*node_id)).write().expect("RwLock poisoned during grad_fn clear") };
            //     node_data.grad_fn = None;
            // }
        }

        // --- 6. Final Gradient Assignment ---
        // Iterate through the final accumulated gradients and assign them to TensorData.grad
        for (node_id, final_grad) in grad_map {
            // Skip assigning gradient to self if it was the starting point?
            // No, the map contains the correct final gradient for all nodes.

            // Acquire write lock to update the grad field
            let mut node_data = unsafe {
                (*node_id)
                    .write()
                    .expect("RwLock poisoned during final grad assignment")
            };

            // Check device just in case
            if node_data.device != final_grad.device() {
                return Err(NeuraRustError::InternalError(format!(
                    "Final gradient device ({:?}) mismatch with tensor device ({:?}) for node.",
                    final_grad.device(),
                    node_data.device
                )));
            }

            // Use the existing `acc_grad` logic within TensorData might be cleaner?
            // Or directly assign here. Direct assignment is simpler now.
            node_data.grad = Some(final_grad);
        }

        Ok(())
    }
}
