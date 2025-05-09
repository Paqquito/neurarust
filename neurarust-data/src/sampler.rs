// neurarust-data/src/sampler.rs

use std::fmt::Debug;

/// A Sampler trait that defines how to iterate over indices of a dataset.
///
/// Samplers are used by `DataLoader` to generate a sequence of indices
/// to fetch data from a `Dataset`.
pub trait Sampler: Debug + Send + Sync {
    /// Returns an iterator over the indices of a dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset_len` - The total number of items in the dataset.
    fn iter(&self, dataset_len: usize) -> Box<dyn Iterator<Item = usize> + Send + Sync>;

    /// Returns the total number of samples that will be yielded by the iterator.
    ///
    /// This might be different from `dataset_len`, especially for samplers
    /// that allow specifying a fixed number of samples or sample with replacement.
    ///
    /// # Arguments
    ///
    /// * `dataset_len` - The total number of items in the dataset.
    fn len(&self, dataset_len: usize) -> usize;
}

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
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler_len() {
        let sampler = SequentialSampler::new();
        assert_eq!(sampler.len(0), 0);
        assert_eq!(sampler.len(5), 5);
        assert_eq!(sampler.len(100), 100);
    }

    #[test]
    fn test_sequential_sampler_iter_empty() {
        let sampler = SequentialSampler::new();
        let mut iter = sampler.iter(0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_sequential_sampler_iter_non_empty() {
        let sampler = SequentialSampler::new();
        let dataset_len = 5;
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), dataset_len);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sequential_sampler_iter_collect_and_count() {
        let sampler = SequentialSampler::new();
        let dataset_len = 3;
        let iter = sampler.iter(dataset_len);
        // Check that count consumes the iterator and gives the correct number
        assert_eq!(iter.count(), dataset_len); 
    }
} 