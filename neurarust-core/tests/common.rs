use neurarust_core::tensor::Tensor;
// Remove unused traits
// use num_traits::{One, Zero};
// use std::cmp::PartialEq;
// use std::default::Default;
// use std::fmt::Debug;
// use std::iter::Sum;
// use std::ops::AddAssign;

// Helper function to create a basic F32 tensor for testing
// Made public within the crate (`pub(crate)`) so other test modules can use it.
// Added allow(dead_code) because usage across different test crates isn't detected easily.
#[allow(dead_code)]
pub(crate) fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor // No <T>
{
    // Call the non-generic Tensor::new
    Tensor::new(data, shape).expect("Test tensor creation failed")
} 