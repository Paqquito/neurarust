use std::fmt::Debug;
// Import traits needed by optimizers for bounds
use num_traits::{Float, FromPrimitive, Zero, One, Pow}; 
use std::ops::{AddAssign, Sub, Neg, Mul, Add, Div, SubAssign};
use std::iter::Sum as IterSum;
use neurarust_core::nn::Parameter; // Import Parameter
use neurarust_core::error::NeuraRustError; // Import NeuraRustError


// Define modules for optimizers
pub mod sgd;
pub mod adam;

// Re-export optimizers
pub use sgd::SGD;
pub use adam::Adam;

/// Trait for optimization algorithms.
/// Optimizers update the parameters of a model based on their gradients.
pub trait Optimizer<T> 
where 
    T: Float + FromPrimitive + Zero + One + Pow<T, Output = T> + Pow<i32, Output = T> 
    + AddAssign + Sub<Output = T> + Neg<Output = T> + Mul<Output = T> + Add<Output = T> 
    + Div<Output = T> + SubAssign + IterSum + Default + Debug + Copy + 'static + Send + Sync,
{
    /// Performs a single optimization step (parameter update).
    fn step(&mut self) -> Result<(), NeuraRustError>;

    /// Clears the gradients of all parameters managed by the optimizer.
    /// Should be called before the backward pass to avoid accumulating gradients
    /// from multiple iterations.
    fn zero_grad(&mut self);

    /// Returns a clone of the parameters managed by this optimizer.
    fn get_params(&self) -> Vec<Parameter<T>>;
}
