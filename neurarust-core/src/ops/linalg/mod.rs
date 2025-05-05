// src/ops/linalg/mod.rs

//! # Linear Algebra Operations
//!
//! This module provides functions for performing linear algebra operations
//! on tensors, such as matrix multiplication.
//!
//! ## Currently Implemented:
//! - Matrix Multiplication (`matmul`)
//!
//! ## Future Work:
//! - Transpose (as an op, distinct from the view method)
//! - Dot product
//! - Vector/Matrix norms
//! - Decompositions (SVD, QR, etc.)
//! - Inverse, determinant, etc.

// Comment out or remove module declarations for which files don't exist
// pub mod transpose;
pub mod matmul;

// TODO: Re-introduce these modules when their implementation starts

// Re-export only matmul (commented out as module is commented out)
// pub use matmul::matmul_op;

// MatmulBackward struct and impl moved to matmul.rs
// Tests related to matmul moved to matmul.rs
// Remove leftover imports and test module from previous structure

// --- Matmul Backward Operation ---
// (struct and impl moved)

// --- Tests ---
// (tests moved)
