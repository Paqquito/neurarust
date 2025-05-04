// neurarust-core/src/ops/comparison/mod.rs

// Declare comparison operations
pub mod equal;
// Add others like greater, less, etc. later

// Re-export the operation functions
pub use equal::equal_op; 