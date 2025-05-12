use num_traits::{Float, NumAssignOps, NumOps}; // Importer NumAssignOps et NumOps pour les opérations
use std::fmt::Debug;

/// A trait representing numeric types usable in NeuraRust tensor operations.
///
/// This trait bounds the types (like `f32`, `f64`) that can be used within
/// the generic kernels of tensor operations. It ensures that the type
/// supports necessary mathematical operations, comparisons, and other properties.
/// \[NEURARUST\] Ce trait est strictement réservé aux types flottants (f32, f64).
/// Pour les entiers, utiliser le trait `NeuraIntegral` (voir ci-dessous).
pub trait NeuraNumeric:
    Float // Includes Num + Copy + Bounded + Signed + etc.
    + NumAssignOps // Includes AddAssign, SubAssign, MulAssign, DivAssign, RemAssign
    + NumOps // Includes Add, Sub, Mul, Div, Rem (needed explicitly beyond Float's ops for some generic contexts)
    + PartialOrd
    + Debug
    + Copy // Float requires Copy, explicitly listed for clarity
    + Send
    + Sync
    + 'static
{
    // Potentially add associated constants or methods if needed later,
    // beyond what `Float` provides (e.g., specialized casting).
    // `Float` already provides `zero()`, `one()`, `min_value()`, `max_value()`, etc.
}

// Implement the trait for f32 and f64.
// The compiler checks if f32/f64 satisfy all the bounds of NeuraNumeric.
impl NeuraNumeric for f32 {}
impl NeuraNumeric for f64 {}

// Optional: Add simple compile-time tests to ensure the trait bounds work.
#[cfg(test)]
mod tests {
    use super::*;

    // Function requiring NeuraNumeric bound
    fn process_numeric<T: NeuraNumeric>(_value: T) {
        // Do nothing, just check if it compiles
    }

    #[test]
    fn test_f32_impl_neurarumeric() {
        process_numeric(1.0f32);
    }

    #[test]
    fn test_f64_impl_neurarumeric() {
        process_numeric(1.0f64);
    }
}

// === TRAIT POUR LES ENTIERS ===
use num_traits::{PrimInt, NumAssign, Bounded};

/// Un trait représentant les types entiers utilisables dans les opérations sur les tenseurs NeuraRust.
/// Ce trait est conçu pour les types comme i32, i64, etc.
pub trait NeuraIntegral:
    PrimInt // Inclut Copy, Ord, Eq, Num, Bounded, etc.
    + NumAssign // Inclut AddAssign, SubAssign, etc.
    + NumAssignOps // Pour compatibilité générique
    + NumOps // Add, Sub, Mul, Div, Rem
    + Bounded
    + Debug
    + Copy
    + Send
    + Sync
    + 'static
{
    // Possibilité d'ajouter des méthodes associées spécifiques plus tard
}

impl NeuraIntegral for i32 {}
impl NeuraIntegral for i64 {}

#[cfg(test)]
mod integral_tests {
    use super::*;
    fn process_integral<T: NeuraIntegral>(_value: T) {}
    #[test]
    fn test_i32_impl_neuraintegral() { process_integral(1i32); }
    #[test]
    fn test_i64_impl_neuraintegral() { process_integral(1i64); }
}

// === TRAIT POUR LES BOOLÉENS ===
/// Un trait représentant les types booléens utilisables dans les opérations sur les tenseurs NeuraRust.
/// Ce trait est conçu pour le type bool uniquement.
pub trait NeuraBoolean:
    std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Not<Output = Self>
    + PartialEq
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
{
    // Possibilité d'ajouter des méthodes associées spécifiques plus tard
}

impl NeuraBoolean for bool {}

#[cfg(test)]
mod boolean_tests {
    use super::*;
    fn process_boolean<T: NeuraBoolean>(_value: T) {}
    #[test]
    fn test_bool_impl_neuraboolean() { process_boolean(true); }
} 