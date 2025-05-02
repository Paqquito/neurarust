pub mod transpose;
pub mod matmul;

// Re-export only matmul
pub use matmul::matmul;
// REMOVED: pub use transpose::transpose; // Transpose is a method, not a free function

// MatmulBackward struct and impl moved to matmul.rs
// Tests related to matmul moved to matmul.rs
// Remove leftover imports and test module from previous structure

// --- Matmul Backward Operation --- 
// (struct and impl moved)

// --- Tests --- 
// (tests moved)
