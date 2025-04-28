// Déclare les modules principaux de la crate
pub mod autograd;
// pub mod creation;
// pub mod indexing;
pub mod ops;
pub mod tensor;
pub mod tensor_data;

// Declare new top-level modules
pub mod nn;
// pub mod optim; // REMOVED: Optimizers moved to neurarust-optim crate
pub mod utils;

// Declare new sub-modules within ops
// (ops/mod.rs needs to declare them too)
// pub mod activation; // Declaration should be in ops/mod.rs
// pub mod reduction; // Declaration should be in ops/mod.rs
// pub mod loss; // Declaration should be in ops/mod.rs

// Ré-exporte le type Tensor pour qu'il soit accessible directement via `neurarust_core::Tensor`
pub use tensor::Tensor;

// Le reste du code (structs, impls, anciens tests) a été déplacé
// dans les modules correspondants.
