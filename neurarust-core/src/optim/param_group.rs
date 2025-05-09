use std::sync::{Arc, RwLock};
use crate::nn::parameter::Parameter;

/// Defines a group of parameters with specific optimizer hyperparameters.
/// 
/// This allows applying different settings (like learning rate or weight decay)
/// to different parts of a model.
#[derive(Clone, Debug)]
pub struct ParamGroup {
    /// The parameters included in this group.
    /// These are typically `Arc<Mutex<Parameter>>` to allow shared ownership
    /// and interior mutability, as optimizers will modify these parameters.
    pub params: Vec<Arc<RwLock<Parameter>>>,
    
    /// Specific options/hyperparameters for this group.
    pub options: ParamGroupOptions,
}

/// Options specific to a parameter group.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamGroupOptions {
    pub lr: Option<f32>,
    pub betas: Option<(f32, f32)>,
    pub eps: Option<f32>,
    pub weight_decay: Option<f32>,
    pub amsgrad: Option<bool>,
    pub momentum: Option<f32>,
    pub dampening: Option<f32>,
    pub nesterov: Option<bool>,
    pub lr_decay: Option<f32>,
}

impl ParamGroup {
    /// Creates a new parameter group with default options.
    pub fn new(params: Vec<Arc<RwLock<Parameter>>>) -> Self {
        ParamGroup {
            params,
            options: ParamGroupOptions::default(),
        }
    }

    /// Returns weak references to the parameters in this group.
    pub fn get_params(&self) -> Vec<std::sync::Weak<RwLock<Parameter>>> {
        self.params.iter().map(Arc::downgrade).collect()
    }
    
    pub fn set_lr(&mut self, lr: f32) {
        self.options.lr = Some(lr);
    }
    
    pub fn get_lr(&self) -> Option<f32> {
        self.options.lr
    }
} 