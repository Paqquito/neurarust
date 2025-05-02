// Déclare les sous-modules d'opérations
pub mod arithmetic;
// pub mod linalg; // REMOVED

// Declare new operation categories
// pub mod activation; // REMOVED
pub mod reduction;
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
// pub use linalg::{matmul, transpose}; // REMOVED
pub use reduction::sum_axes; // Keep sum
// pub use indexing::slice; // REMOVED
// pub use activation::relu; // REMOVED