use crate::tensor::Tensor;
use std::collections::HashMap;

// Import pour RmsPropParamState. Assurez-vous que RmsPropParamState est pub dans rmsprop.rs.
// Pour l'instant, cette ligne est commentée pour éviter une erreur si rmsprop.rs n'est pas encore compilable
// ou si RmsPropParamState n'est pas pub.
// Si RmsPropParamState est bien pub et dans crate::optim::rmsprop, décommentez :
// use crate::optim::rmsprop::{RmsPropParamState};

// Alternative temporaire pour la compilation, si RmsPropParamState n'est pas accessible:
// Remplacer `RmsPropParamState` dans la variante RmsProp par une structure locale ou () 
// et ajuster rmsprop.rs en conséquence pour state_dict / load_state_dict.
// Pour cet exemple, je vais assumer que RmsPropParamState sera rendu pub.
use crate::optim::rmsprop::RmsPropParamState; 

/// Represents the state of an optimizer.
///
/// This enum will be extended to hold specific state information
/// for different types of optimizers (e.g., momentum buffers for SGD,
/// first and second moment estimates for Adam).
/// The specific states might include Tensors, so `Tensor` might need to be in scope
/// when variants are fully defined. For now, it's kept simple.
#[derive(Debug, Clone)]
pub enum OptimizerState {
    /// State specific to the SGD optimizer.
    Sgd {
        /// Momentum buffers associated with parameters.
        /// The key could be a unique identifier for the parameter (e.g., its memory address or a generated ID).
        /// Using `usize` as a placeholder key type for now.
        momentum_buffers: HashMap<usize, Tensor>,
    },
    /// State specific to the Adam optimizer.
    Adam { 
        // Adaptez selon la structure réelle de l'état Adam
        // state: HashMap<String, crate::optim::adam::AdamStateInternal>, // Si AdamState est interne à adam.rs
        // iterations: u64,
        // lr: f32, 
        // ... autres hyperparams d'Adam à sauvegarder
    },
    /// State specific to the RmsProp optimizer.
    RmsProp {
        param_states: HashMap<String, RmsPropParamState>, // Clé: nom du paramètre
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
        iterations: u64,
    },
    /// A generic placeholder state for optimizers without specific state yet
    /// or for initialization.
    Placeholder,
}

impl Default for OptimizerState {
    fn default() -> Self {
        OptimizerState::Placeholder
    }
} 