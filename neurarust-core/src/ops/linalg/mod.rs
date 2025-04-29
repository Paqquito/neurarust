pub mod transpose;
pub mod matmul;

// Re-export the public matmul function
pub use matmul::matmul;

// MatmulBackward struct and impl moved to matmul.rs
// Tests related to matmul moved to matmul.rs
// Remove leftover imports and test module from previous structure

// --- Matmul Backward Operation --- 
// (struct and impl moved)

// --- Tests --- 
// (tests moved)
