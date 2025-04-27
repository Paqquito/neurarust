// src/tensor_data.rs
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::rc::{Rc, Weak};
use crate::autograd::BackwardOp;
use crate::tensor::Tensor; // Need Tensor for grad field

/// Holds the actual data and metadata for a tensor.
/// Uses Rc<RefCell<...>> for shared ownership and interior mutability.
pub(crate) struct TensorData<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Tensor<T>>,
    pub(crate) grad_fn: Option<Rc<dyn BackwardOp<T>>>,
    pub(crate) _ctx: Option<Weak<dyn BackwardOp<T>>>, // Keep for backward?
}

// Manual implementation of Debug
impl<T: Debug> Debug for TensorData<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("TensorData")
         .field("data", &self.data)
         .field("shape", &self.shape)
         .field("requires_grad", &self.requires_grad)
         .field("grad_defined", &self.grad.is_some())
         .field("grad_fn_defined", &self.grad_fn.is_some())
         .field("_ctx_defined", &self._ctx.is_some())
         .finish()
    }
}

// Manual implementation of PartialEq
impl<T: PartialEq> PartialEq for TensorData<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape && self.requires_grad == other.requires_grad
    }
}

// Eq requires that a == a always holds, which is true for our PartialEq implementation IF T: Eq.
impl<T: Eq> Eq for TensorData<T> {}

impl<T> TensorData<T> {
    // Helper to get number of elements, used internally
    pub(crate) fn numel(&self) -> usize {
        self.data.len()
    }
} 