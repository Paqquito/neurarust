// Declare the sqrt module within math_elem
pub mod sqrt;

// Re-export the public function
pub use sqrt::sqrt_op;

// Potentially re-export SqrtOp if needed directly, but unlikely
// pub use sqrt::SqrtOp; 