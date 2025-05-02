use neurarust_core::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::PartialEq;
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

// Helper function to create a basic tensor for testing
// It requires many trait bounds, ensure they are available or adjust tests.
// Made public within the crate (`pub(crate)`) so other test modules can use it.
// Added allow(dead_code) because usage across different test crates isn't detected easily.
#[allow(dead_code)]
pub(crate) fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: Clone
        + Debug
        + PartialEq
        + Zero
        + One
        + Copy
        + AddAssign
        + Sum
        + Default
        + Send
        + Sync
        + 'static,
{
    Tensor::new(data, shape).expect("Test tensor creation failed")
} 