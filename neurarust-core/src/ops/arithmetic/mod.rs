// Declare arithmetic operation modules
pub mod add;
pub mod sub;
pub mod mul;
pub mod div;
pub mod neg;
pub mod pow;

// Re-export the public, fallible functions
pub use add::add;
pub use sub::sub;
pub use mul::mul;
pub use div::div;
pub use neg::neg;
pub use pow::pow_scalar; // Consider renaming re-export e.g., `pub use pow::pow_scalar as pow;` 