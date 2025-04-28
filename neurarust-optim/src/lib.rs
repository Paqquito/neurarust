use neurarust_core::tensor::Tensor;
use std::fmt::Debug;

// Define modules for optimizers
pub mod sgd;
pub mod adam;

/// Trait for optimization algorithms.
/// Optimizers update the parameters of a model based on their gradients.
pub trait Optimizer<T: Debug + Copy + 'static> {
    /// Performs a single optimization step (parameter update).
    /// 
    /// # Arguments
    /// * `params` - A mutable slice of Tensors representing the model parameters to be updated.
    fn step(&mut self, params: &mut [&mut Tensor<T>]);

    /// Clears the gradients of all parameters managed by the optimizer.
    /// Should be called before the backward pass to avoid accumulating gradients
    /// from multiple iterations.
    /// 
    /// # Arguments
    /// * `params` - A mutable slice of Tensors representing the model parameters whose gradients should be cleared.
    fn zero_grad(&self, params: &mut [&mut Tensor<T>]);
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
