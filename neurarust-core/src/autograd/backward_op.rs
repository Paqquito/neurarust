// Define the BackwardOp trait here
// (Copied from the old mod.rs and dependencies added)

use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::RwLock;

/// Defines the interface for the backward pass of a differentiable tensor operation.
///
/// Any operation that creates a non-leaf `Tensor` (a tensor resulting from an operation
/// on inputs that require gradients) must have an associated `BackwardOp` implementation.
/// This implementation is stored in the output tensor\'s `grad_fn` field and is used
/// during the `backward()` call to propagate gradients according to the chain rule.
///
/// The trait requires `Debug + Send + Sync` bounds because the `Arc<dyn BackwardOp>` holding
/// the state might be shared and potentially accessed across different threads during the
/// backward pass or for debugging purposes.
pub trait BackwardOp: Debug + Send + Sync {
    /// Computes the gradients of the operation\'s inputs with respect to the loss,
    /// given the gradient of the operation\'s output with respect to the loss.
    ///
    /// This method implements the core logic of the chain rule for the specific operation.
    /// It receives \( \\frac{dL}{d\\text{Output}} \) (`grad_output`)\
    /// and must compute \( \\frac{dL}{d\\text{Input}_i} \) for each input \( i \).
    ///
    /// Mathematically, if the operation is \( \\text{Output} = f(\text{Input}_1, ..., \text{Input}_n) \\),\
    /// this method computes:\
    /// \\[ \\frac{dL}{d\\text{Input}_i} = \\frac{dL}{d\\text{Output}} \\cdot \\frac{d\\text{Output}}{d\\text{Input}_i} \\]\
    /// (where multiplication might represent dot products, element-wise multiplication, etc.,
    /// depending on the operation and tensor shapes).
    ///
    /// # Arguments
    /// * `grad_output`: A reference to the `Tensor` representing the gradient flowing into the
    ///                  output node of this operation (dL/dOutput).
    ///                  It\'s expected to have the same shape and device as the operation\'s output tensor.
    ///
    /// # Returns
    /// * `Ok(Vec<Tensor>)`: A `Vec` containing the computed gradient `Tensor` for each input.
    ///    The order of tensors in the vector **must** strictly match the order of the corresponding
    ///    input tensors used during the forward pass and returned by the `inputs()` method.
    ///    Each gradient tensor (dL/dInput_i) should have the same shape and device as the
    ///    corresponding input tensor (Input_i).
    /// * `Err(NeuraRustError)`: If an error occurs during gradient computation (e.g., shape mismatch,
    ///    device mismatch, numerical issues).
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError>;

    /// Returns identifiers for the input `TensorData` nodes that participated in the forward operation.
    ///
    /// This method is crucial for reconstructing and traversing the computation graph during the
    /// backward pass. It provides the links from the current operation node back to its predecessors.
    ///
    /// The identifiers returned are raw pointers (`*const RwLock<TensorData>`) to the shared, mutable
    /// internal data of the input tensors. Using pointers allows for a stable identity even if
    /// `Tensor` structs (which are just wrappers around `Arc<RwLock<TensorData>>`) are cloned or dropped,
    /// making them suitable as keys in graph structures like `HashMap`.
    ///
    /// **Safety:** The validity of these pointers relies on the `Arc`s to the corresponding `TensorData`
    /// being kept alive for the duration of the backward pass (typically ensured by the graph structure
    /// and the backward algorithm).
    ///
    /// # Returns
    /// A `Vec` of raw pointers, where each pointer corresponds to an input `TensorData` used in the
    /// forward pass. The order **must** match the order of gradients returned by `backward()`.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>>;
}
