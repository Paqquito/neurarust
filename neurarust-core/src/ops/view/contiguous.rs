//! Contains the backward operation for the `contiguous()` method.

use crate::autograd::BackwardOp;
use crate::autograd::graph::NodeId;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;

/// Backward pass structure for the `contiguous()` operation.
///
/// When `z = a.contiguous()` where `a` was non-contiguous, the gradient flow is simple:
/// `dL/da = dL/dz`. The gradient `dL/dz` (grad_output) just needs to be passed back
/// to the original tensor `a`.
#[derive(Debug)]
pub(crate) struct ContiguousBackward {
    /// Node ID of the original tensor (`a`).
    pub(crate) a_node: NodeId, // We only need the ID to link the graph
}

impl BackwardOp for ContiguousBackward {
    /// Passes the gradient through to the original tensor.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // The gradient flows directly back to the original tensor 'a'.
        // No transformation needed on grad_output itself.
        Ok(vec![grad_output.clone()])
    }

    /// Returns the identifier of the original input tensor node.
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.a_node]
    }
}

// Safety: NodeId is treated as an opaque identifier here and is not dereferenced
// in a way that would cause data races across threads in the current single-threaded
// autograd execution model. These impls assert thread safety based on current usage.
unsafe impl Send for ContiguousBackward {}
unsafe impl Sync for ContiguousBackward {} 