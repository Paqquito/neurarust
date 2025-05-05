//!
//! # Element-wise Mathematical Functions
//!
//! This module provides element-wise mathematical functions beyond basic arithmetic,
//! such as exponentiation, logarithm, square root, trigonometric functions, etc.
//!
//! These operations typically support automatic differentiation.
//!
//! ## Currently Implemented:
//! - [`ln_op`](ln/fn.ln_op.html): Natural logarithm (base e).
//!
//! ## Future Work:
//! - `exp_op`: Exponential function (e^x).
//! - `sqrt_op`: Square root.
//! - Trigonometric functions (sin, cos, tan, etc.).
//! - Power functions involving scalars (pow with scalar exponent/base).

// Declare the sqrt module within math_elem
// pub mod sqrt;

// Re-export the public function
// pub use sqrt::sqrt_op;

// Potentially re-export SqrtOp if needed directly, but unlikely
// pub use sqrt::SqrtOp;

// Comment out or remove module declarations for which files don't exist
// TODO: Re-introduce these modules when their implementation starts

// Declare element-wise math operations
pub mod ln;
// pub mod exp; // Add later if needed

// Re-export the operation functions
pub use ln::ln_op;
// pub use exp::exp_op; // Add later if needed
