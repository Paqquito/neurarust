// neurarust-data/src/samplers/traits.rs

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