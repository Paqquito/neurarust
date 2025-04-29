// Déclare les sous-modules d'opérations
pub mod arithmetic;
pub mod linalg;

// Declare new operation categories
// pub mod activation;
pub mod reduction;
pub mod loss;
pub mod indexing;
pub mod stack;
pub mod math_elem;

// Potentially re-export specific operations if needed
// pub use arithmetic::add::add; 

// Re-export core operations for easier access
pub use arithmetic::{add, div, mul, neg, pow, sub};
pub use linalg::{matmul, transpose};
pub use reduction::sum;
// pub use indexing::slice; // Removed - slice is a Tensor method, not a standalone op here
// pub use activation::relu; // Example if activation module existed 