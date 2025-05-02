// Déclare les sous-modules d'opérations
pub mod arithmetic;
pub mod linalg;
pub mod loss;
pub mod math_elem;
pub mod reduction;
pub mod view_ops;
// pub mod activation; // REMOVED
// pub mod indexing; // REMOVED
// pub mod stack; // REMOVED
// pub mod reshape; // REMOVED

// Declare new operation categories
// pub mod activation; // REMOVED
// pub mod loss; // REMOVED (part of nn, not ops base)
// pub mod indexing; // REMOVED
// pub mod stack; // REMOVED
// pub mod math_elem; // REMOVED

// Individual ops that might not fit cleanly into categories yet
// pub mod reshape; // REMOVED

// Potentially re-export specific operations if needed
// pub use arithmetic::add::add;

// Re-export core operations for easier access (Phase 0)
pub use arithmetic::{add_op, div_op, mul_op, neg_op, sub_op}; // Keep arithmetic ops
// pub use arithmetic::pow_op; // Pow might be Phase 1+
// Comment out problematic exports
// pub use linalg::{matmul, transpose}; // Keep existing linalg exports
pub use reduction::sum_axes; // Keep existing reduction export

// Remove problematic exports causing linter errors
// pub use math_elem::{exp_op, ln_op, pow_op};
// pub use reduction::{mean_op, sum_axes_op, sum_op};
// pub use linalg::matmul_op;
// pub use activation::relu_op;

// Potentially export view ops later if needed publicly
// pub use view_ops::{slice_op, ...};