use crate::error::NeuraRustError;
use super::param_group::ParamGroup;
use super::optimizer_state::OptimizerState;

/// Trait defining the common interface for all optimizers.
///
/// Optimizers are responsible for updating model parameters based on their gradients.
pub trait Optimizer {
    /// Performs a single optimization step.
    ///
    /// This method applies the optimization algorithm (e.g., SGD, Adam)
    /// to update the parameters managed by the optimizer, using their
    /// accumulated gradients.
    /// 
    /// # Returns
    ///
    /// `Ok(())` if the step was successful, or a `NeuraRustError` otherwise.
    fn step(&mut self) -> Result<(), NeuraRustError>;

    /// Clears the gradients of all parameters managed by the optimizer.
    ///
    /// This is typically called before the backward pass in a new training iteration
    /// to prevent gradients from accumulating across iterations (unless desired behavior).
    fn zero_grad(&mut self);

    /// Adds a new parameter group to the optimizer.
    ///
    /// This allows specifying different hyperparameters (e.g., learning rate)
    /// for different sets of parameters within the same optimizer.
    ///
    /// # Arguments
    ///
    /// * `param_group`: The `ParamGroup` to add.
    fn add_param_group(&mut self, param_group: ParamGroup);

    /// Returns an immutable slice of the parameter groups managed by the optimizer.
    /// Each group contains parameters and their specific hyperparameters (e.g., learning rate).
    fn param_groups(&self) -> &[ParamGroup];

    /// Returns a mutable slice of the parameter groups managed by the optimizer.
    /// This allows modifying hyperparameters like the learning rate for each group.
    fn param_groups_mut(&mut self) -> &mut [ParamGroup];

    /// Loads the optimizer's state from an `OptimizerState` object.
    ///
    /// This is useful for resuming training from a saved state or for
    /// transferring learning. The structure of `state_dict` should match
    /// what is expected by the specific optimizer implementation.
    ///
    /// # Arguments
    ///
    /// * `state_dict`: The state dictionary to load.
    ///
    /// # Returns
    ///
    /// `Ok(())` if loading was successful, or a `NeuraRustError` if the state
    /// is incompatible or an error occurs.
    fn load_state_dict(&mut self, state_dict: &OptimizerState) -> Result<(), NeuraRustError>;

    /// Returns the optimizer's current state as an `OptimizerState` object.
    ///
    /// This state can be saved and later used with `load_state_dict` to resume
    /// the optimizer's progress (e.g., momentum buffers, step counts for Adam).
    ///
    /// # Returns
    ///
    /// `Ok(OptimizerState)` containing the current state, or a `NeuraRustError` if
    /// the state cannot be collected.
    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError>;
} 