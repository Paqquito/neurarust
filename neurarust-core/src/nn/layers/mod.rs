// src/nn/layers/mod.rs
// Module pour les différentes couches (Linear, Conv2d, etc.)

pub mod linear;
pub mod relu;
// pub mod conv;
// ... autres couches ... 

// Re-export key layer structs
pub use linear::Linear;
pub use relu::ReLU; 