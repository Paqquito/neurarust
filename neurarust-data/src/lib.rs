// pub mod dataset; // Ancienne déclaration
// pub mod vec_dataset; // Ancienne déclaration
// pub mod tensor_dataset; // Ancienne déclaration
pub mod datasets; // Nouvelle déclaration
pub mod samplers;

// pub use dataset::Dataset; // Ancien export
// pub use vec_dataset::VecDataset; // Ancien export
// pub use tensor_dataset::TensorDataset; // Ancien export
pub use datasets::{Dataset, VecDataset, TensorDataset}; // Nouveaux exports
pub use samplers::{Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler}; 