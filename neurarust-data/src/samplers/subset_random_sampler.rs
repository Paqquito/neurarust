// neurarust-data/src/samplers/subset_random_sampler.rs

use super::traits::Sampler;
use rand::seq::SliceRandom;
use std::fmt::Debug;

/// A sampler that randomly samples indices from a provided subset of indices.
#[derive(Debug, Clone)]
pub struct SubsetRandomSampler {
    indices: Vec<usize>,
}

impl SubsetRandomSampler {
    /// Creates a new `SubsetRandomSampler`.
    ///
    /// # Arguments
    ///
    /// * `indices`: A vector of indices from which to sample randomly.
    pub fn new(indices: Vec<usize>) -> Self {
        SubsetRandomSampler {
            indices
        }
    }
}

impl Sampler for SubsetRandomSampler {
    fn iter(&self, _dataset_len: usize) -> Box<dyn Iterator<Item = usize> + Send + Sync> {
        if self.indices.is_empty() {
            return Box::new(std::iter::empty());
        }
        let mut rng = rand::thread_rng();
        let mut mut_indices = self.indices.clone();
        mut_indices.shuffle(&mut rng);
        Box::new(mut_indices.into_iter())
    }

    fn len(&self, _dataset_len: usize) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
#[path = "subset_random_sampler_test.rs"]
mod tests; 