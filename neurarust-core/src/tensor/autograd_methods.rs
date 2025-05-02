use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::autograd::BackwardOp;
use std::sync::Arc;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::fmt::Debug;
use std::marker::Copy;
use std::ops::{Add, AddAssign};

// Note: T bounds for the impl block cover all methods inside
impl<T: 'static + Debug + Copy> Tensor<T> {
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
        T: Add<Output = T> + AddAssign + Zero + One + Sum + PartialEq + Default + Send + Sync
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
    pub fn set_grad_fn(&self, grad_fn: Option<Arc<dyn BackwardOp<T> + Send + Sync>>) -> Result<(), NeuraRustError> {
        let mut guard = self.write_data();
        guard.grad_fn = grad_fn;
        Ok(())
    }
} 