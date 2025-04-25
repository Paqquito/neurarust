// Déclare les modules principaux de la crate
pub mod autograd;
pub mod creation;
pub mod indexing;
pub mod ops;
pub mod tensor;

// Ré-exporte le type Tensor pour qu'il soit accessible directement via `neurarust_core::Tensor`
pub use tensor::Tensor;

// Le reste du code (structs, impls, anciens tests) a été déplacé
// dans les modules correspondants.
