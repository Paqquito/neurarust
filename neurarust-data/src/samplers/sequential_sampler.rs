// neurarust-data/src/samplers/sequential_sampler.rs

use super::traits::Sampler; // Utilisez super::traits pour accéder au Sampler
use std::fmt::Debug;

/// Samples elements sequentially, always in the same order.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialSampler;

impl SequentialSampler {
    /// Creates a new `SequentialSampler`.
    pub fn new() -> Self {
        SequentialSampler
    }
}

impl Sampler for SequentialSampler {
    fn iter(&self, dataset_len: usize) -> Box<dyn Iterator<Item = usize> + Send + Sync> {
        Box::new(0..dataset_len)
    }

    fn len(&self, dataset_len: usize) -> usize {
        dataset_len
    }
}

#[cfg(test)]
#[path = "sequential_sampler_test.rs"]
mod tests; 