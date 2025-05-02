// Contenu du module autograd supprimé pour revenir à la Phase 0. 

use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData; // Import TensorData
use std::fmt::Debug;
use std::sync::RwLock; // Import RwLock

/// Trait for operations that support backward pass in the computation graph.
/// Implementors define how to compute gradients for their inputs given the output gradient.
/// Needs Send + Sync bounds because the Arc<dyn BackwardOp> might be shared across threads.
pub trait BackwardOp<T: 'static + Debug + Copy>: Debug + Send + Sync {
    /// Performs the backward pass.
    ///
    /// # Arguments
    /// * `grad_output` - The gradient flowing back from the operation's output tensor.
    ///                   Must be on the same device as the expected input gradients.
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of gradient `Tensor`s corresponding to each
    /// input tensor of the forward operation, or a `NeuraRustError` if the backward
    /// pass fails (e.g., device mismatch, invalid shape). The order must match the
    /// order of inputs in the forward pass.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError>;

    /// Returns identifiers for the input `TensorData` nodes used in the forward pass.
    ///
    /// This is used to traverse the computation graph backwards.
    /// Returning raw pointers (`*const RwLock<TensorData<T>>>`) provides a stable identifier
    /// for each `TensorData` involved, suitable for use in graph structures (e.g., HashMap keys).
    /// It's crucial that these pointers remain valid for the lifetime of the graph traversal.
    /// Using `Arc::as_ptr` on the `Arc<RwLock<TensorData<T>>>` held by the corresponding
    /// input `Tensor`s is a common way to obtain these pointers.
    fn inputs(&self) -> Vec<*const RwLock<TensorData<T>>>;
}

// Placeholder for graph traversal logic (to be implemented later in Phase 1.2)
pub mod graph {
    // Functions like topological_sort will go here
} 