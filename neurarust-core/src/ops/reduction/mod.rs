// src/ops/reduction/mod.rs
// Module pour les opérations de réduction (Sum, Mean, Max, Min, etc.)

pub mod sum;
pub mod mean;

// Add re-export for sum_axes
pub use sum::sum_axes;
pub use mean::mean_op;
