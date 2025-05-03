// src/ops/reduction/mod.rs
// Module pour les opérations de réduction (Sum, Mean, Max, Min, etc.)

pub mod sum;
pub mod mean;
pub mod max;

// Re-export the adapted reduction operations using pub(crate)
pub(crate) use sum::sum_op;

// Remove old/incorrect exports
// pub use sum::sum_axes;
// pub use mean::mean_axes;
