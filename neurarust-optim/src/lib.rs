use neurarust_core::tensor::Tensor;
use std::fmt::Debug;
// Import traits needed by optimizers for bounds
use num_traits::{Float, FromPrimitive, Zero, One, Pow}; 
use std::ops::{AddAssign, Sub, Neg, Mul, Add, Div, SubAssign};
use std::iter::Sum as IterSum;


// Define modules for optimizers
pub mod sgd;
pub mod adam;

// Re-export optimizers
pub use sgd::SGD;
pub use adam::Adam;

/// Trait for optimization algorithms.
/// Optimizers update the parameters of a model based on their gradients.
// Keep minimal bounds on the trait itself
pub trait Optimizer<T> 
where 
    T: Debug + Copy + 'static, // Basic requirements for storing/debugging
{
    /// Performs a single optimization step (parameter update).
    /// 
    /// # Arguments
    /// * `params` - A mutable slice of Tensors representing the model parameters to be updated.
    // Place the extensive calculation bounds here, specific to the step operation
    fn step(&mut self, params: &mut [&mut Tensor<T>])
    where 
        T: Float + FromPrimitive + Zero + One + Pow<T, Output = T> + Pow<i32, Output = T> 
        + AddAssign + Sub<Output = T> + Neg<Output = T> + Mul<Output = T> + Add<Output = T> 
        + Div<Output = T> + SubAssign + IterSum + Default;

    /// Clears the gradients of all parameters managed by the optimizer.
    /// Should be called before the backward pass to avoid accumulating gradients
    /// from multiple iterations.
    /// 
    /// # Arguments
    /// * `params` - A mutable slice of Tensors representing the model parameters whose gradients should be cleared.
    // Place only the Zero bound here, specific to zero_grad
    fn zero_grad(&self, params: &mut [&mut Tensor<T>])
    where T: Zero;
}
