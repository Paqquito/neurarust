// Déclare les sous-modules d'opérations
pub mod arithmetic;
pub mod linalg;

// Declare new operation categories
pub mod activation;
pub mod reduction;
pub mod loss;
pub mod indexing;
pub mod stack;

// Potentially re-export specific operations if needed
// pub use arithmetic::add::add; 