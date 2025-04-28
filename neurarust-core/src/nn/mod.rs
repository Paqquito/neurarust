// src/nn/mod.rs
// Module principal pour les couches de réseau de neurones, les conteneurs, etc.

pub mod layers;
pub mod module; // Trait Module
pub mod parameter; // struct Parameter
pub mod losses; // Declare losses module
// pub mod containers; // Future: Sequential, ModuleList 

// Re-export common items
pub use module::Module;
pub use parameter::Parameter;
pub use layers::linear::Linear;
pub use losses::MSELoss; // Re-export MSELoss 