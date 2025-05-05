//!
//! # Element-wise Arithmetic Operations
//!
//! This module provides functions for performing basic element-wise arithmetic
//! operations on tensors, such as addition, subtraction, multiplication, division,
//! negation, and exponentiation.
//!
//! These operations typically support broadcasting and automatic differentiation.

// Export foundational arithmetic operations directly
pub mod add;
pub mod div;
pub mod mul;
pub mod neg;
pub mod sub;
pub mod pow;

// Re-export the primary operation functions with their new names
pub use add::add_op;
pub use div::div_op;
pub use mul::mul_op;
pub use neg::neg_op;
pub use sub::sub_op;
pub use pow::pow_op;
// pub use pow::pow_op; // Commented out until pow.rs is refactored
