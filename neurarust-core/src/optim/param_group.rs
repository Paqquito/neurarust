use std::sync::{Arc, Mutex};
use crate::nn::parameter::Parameter;

/// Defines a group of parameters with specific optimizer hyperparameters.
/// 
/// This allows applying different settings (like learning rate or weight decay)
/// to different parts of a model.
#[derive(Debug)]
pub struct ParamGroup {
    /// The parameters included in this group.
    /// These are typically `Arc<Mutex<Parameter>>` to allow shared ownership
    /// and interior mutability, as optimizers will modify these parameters.
    pub params: Vec<Arc<Mutex<Parameter>>>,
    /// Learning rate for this group.
    pub lr: f32,
    /// Weight decay (L2 penalty) for this group.
    /// A value of 0.0 means no weight decay.
    pub weight_decay: f32,
    // Other optimizer-specific hyperparameters can be added here in the future.
    // For example, for SGD:
    // pub momentum: f32,
    // pub nesterov: bool,
    // For Adam:
    // pub betas: (f32, f32),
    // pub eps: f32,
    // pub amsgrad: bool,
}

impl ParamGroup {
    /// Creates a new parameter group with specified parameters, learning rate, and weight decay.
    ///
    /// # Arguments
    ///
    /// * `params`: A vector of `Arc<Mutex<Parameter>>` that this group will manage.
    /// * `lr`: The learning rate to apply to the parameters in this group.
    /// * `weight_decay`: The weight decay (L2 penalty) to apply.
    pub fn new(params: Vec<Arc<Mutex<Parameter>>>, lr: f32, weight_decay: f32) -> Self {
        ParamGroup {
            params,
            lr,
            weight_decay,
        }
    }
} 