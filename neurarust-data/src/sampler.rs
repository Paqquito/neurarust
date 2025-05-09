// neurarust-data/src/sampler.rs

use std::fmt::Debug;
use rand::seq::SliceRandom;
use rand::Rng;

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

/// A sampler that randomly samples indices from a dataset.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    replacement: bool,
    num_samples: Option<usize>,
}

impl RandomSampler {
    /// Creates a new `RandomSampler`.
    ///
    /// # Arguments
    ///
    /// * `replacement`: If `true`, an index can be selected multiple times.
    /// * `num_samples`: The total number of samples to draw. If `None`, it defaults to the dataset size.
    pub fn new(replacement: bool, num_samples: Option<usize>) -> Self {
        RandomSampler {
            replacement,
            num_samples,
        }
    }
}

impl Sampler for RandomSampler {
    fn iter(&self, dataset_len: usize) -> Box<dyn Iterator<Item = usize> + Send + Sync> {
        if dataset_len == 0 {
            return Box::new(std::iter::empty());
        }

        let mut rng = rand::thread_rng();
        let actual_num_samples = self.num_samples.unwrap_or(dataset_len);

        if self.replacement {
            let indices: Vec<usize> = (0..actual_num_samples)
                .map(|_| rng.gen_range(0..dataset_len))
                .collect();
            Box::new(indices.into_iter())
        } else {
            if actual_num_samples > dataset_len {
                // This case is problematic without replacement if num_samples > dataset_len.
                // PyTorch throws a ValueError. We will mimic this by returning an empty iterator
                // for now, or we could panic/return Result. For simplicity, empty iter.
                // Consider adding an error type to iter if robust error handling is needed.
                eprintln!(
                    "Warning: RandomSampler: num_samples ({}) > dataset_len ({}) without replacement. Returning empty iterator.",
                    actual_num_samples,
                    dataset_len
                );
                return Box::new(std::iter::empty());
            }
            let mut indices: Vec<usize> = (0..dataset_len).collect();
            indices.shuffle(&mut rng);
            let selected_indices = indices.into_iter().take(actual_num_samples).collect::<Vec<_>>();
            Box::new(selected_indices.into_iter())
        }
    }

    fn len(&self, dataset_len: usize) -> usize {
        self.num_samples.unwrap_or(dataset_len)
    }
}

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
        // dataset_len is ignored here as we only sample from the provided subset.
        if self.indices.is_empty() {
            return Box::new(std::iter::empty());
        }
        let mut rng = rand::thread_rng();
        let mut mut_indices = self.indices.clone(); // Clone to shuffle, and declare as mutable
        mut_indices.shuffle(&mut rng);
        Box::new(mut_indices.into_iter())
    }

    fn len(&self, _dataset_len: usize) -> usize {
        // dataset_len is ignored.
        self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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

    #[test]
    fn test_random_sampler_len_default() {
        let sampler = RandomSampler::new(false, None);
        assert_eq!(sampler.len(10), 10);
    }

    #[test]
    fn test_random_sampler_len_with_num_samples() {
        let sampler = RandomSampler::new(false, Some(5));
        assert_eq!(sampler.len(10), 5);
    }

    #[test]
    fn test_random_sampler_iter_no_replacement_less_than_dataset() {
        let dataset_len = 10;
        let num_samples = 5;
        let sampler = RandomSampler::new(false, Some(num_samples));
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), num_samples);
        let unique_indices: HashSet<usize> = indices.into_iter().collect();
        assert_eq!(unique_indices.len(), num_samples);
        for index in unique_indices {
            assert!(index < dataset_len);
        }
    }

    #[test]
    fn test_random_sampler_iter_no_replacement_equal_to_dataset() {
        let dataset_len = 10;
        let sampler = RandomSampler::new(false, None);
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), dataset_len);
        let unique_indices: HashSet<usize> = indices.into_iter().collect();
        assert_eq!(unique_indices.len(), dataset_len);
    }

    #[test]
    fn test_random_sampler_iter_no_replacement_more_than_dataset_returns_empty() {
        // Mimicking PyTorch's ValueError by returning an empty iterator for now
        let dataset_len = 5;
        let num_samples = 10;
        let sampler = RandomSampler::new(false, Some(num_samples));
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert!(indices.is_empty(), "Should return empty if num_samples > dataset_len without replacement");
    }

    #[test]
    fn test_random_sampler_iter_with_replacement_less_than_dataset() {
        let dataset_len = 10;
        let num_samples = 5;
        let sampler = RandomSampler::new(true, Some(num_samples));
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), num_samples);
        for &index in &indices {
            assert!(index < dataset_len);
        }
    }

    #[test]
    fn test_random_sampler_iter_with_replacement_more_than_dataset() {
        let dataset_len = 5;
        let num_samples = 10;
        let sampler = RandomSampler::new(true, Some(num_samples));
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), num_samples);
        for &index in &indices {
            assert!(index < dataset_len);
        }
        // Check for possible duplicates (though not guaranteed, it's likely with replacement)
        let unique_indices: HashSet<usize> = indices.into_iter().collect();
        assert!(unique_indices.len() <= num_samples);
    }

    #[test]
    fn test_random_sampler_iter_empty_dataset() {
        let sampler = RandomSampler::new(false, None);
        let indices: Vec<usize> = sampler.iter(0).collect();
        assert!(indices.is_empty());

        let sampler_replacement = RandomSampler::new(true, Some(5));
        let indices_replacement: Vec<usize> = sampler_replacement.iter(0).collect();
        assert!(indices_replacement.is_empty());
    }

    #[test]
    fn test_random_sampler_iter_no_replacement_num_samples_none() {
        let dataset_len = 7;
        let sampler = RandomSampler::new(false, None); // num_samples defaults to dataset_len
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), dataset_len);
        let unique_indices: HashSet<usize> = indices.iter().cloned().collect();
        assert_eq!(unique_indices.len(), dataset_len);
        for index in &indices {
            assert!(*index < dataset_len);
        }
    }

    #[test]
    fn test_random_sampler_iter_with_replacement_num_samples_none() {
        let dataset_len = 6;
        let sampler = RandomSampler::new(true, None); // num_samples defaults to dataset_len
        let indices: Vec<usize> = sampler.iter(dataset_len).collect();
        assert_eq!(indices.len(), dataset_len);
        for index in &indices {
            assert!(*index < dataset_len);
        }
    }

    #[test]
    fn test_subset_random_sampler_new() {
        let indices = vec![1, 3, 5, 7];
        let sampler = SubsetRandomSampler::new(indices.clone());
        assert_eq!(sampler.indices, indices);
    }

    #[test]
    fn test_subset_random_sampler_len() {
        let indices = vec![1, 3, 5, 7, 9];
        let sampler = SubsetRandomSampler::new(indices.clone());
        // dataset_len argument to len() is ignored by SubsetRandomSampler
        assert_eq!(sampler.len(100), indices.len());
        assert_eq!(sampler.len(0), indices.len());

        let empty_sampler = SubsetRandomSampler::new(vec![]);
        assert_eq!(empty_sampler.len(10), 0);
    }

    #[test]
    fn test_subset_random_sampler_iter_non_empty() {
        let source_indices = vec![2, 4, 6, 8];
        let sampler = SubsetRandomSampler::new(source_indices.clone());
        let sampled_indices: Vec<usize> = sampler.iter(10).collect(); // dataset_len is ignored

        assert_eq!(sampled_indices.len(), source_indices.len());

        // Check that all sampled indices are from the source_indices
        let source_set: HashSet<usize> = source_indices.into_iter().collect();
        let sampled_set: HashSet<usize> = sampled_indices.into_iter().collect();

        assert_eq!(sampled_set, source_set);
    }

    #[test]
    fn test_subset_random_sampler_iter_empty() {
        let sampler = SubsetRandomSampler::new(vec![]);
        let sampled_indices: Vec<usize> = sampler.iter(10).collect(); // dataset_len is ignored
        assert!(sampled_indices.is_empty());
    }

    #[test]
    fn test_subset_random_sampler_iter_single_element() {
        let source_indices = vec![42];
        let sampler = SubsetRandomSampler::new(source_indices.clone());
        let sampled_indices: Vec<usize> = sampler.iter(100).collect();
        assert_eq!(sampled_indices.len(), 1);
        assert_eq!(sampled_indices[0], 42);
    }

    #[test]
    fn test_subset_random_sampler_iter_shuffles() {
        // This test has a small chance of failing if shuffle results in the same order.
        // For a robust test, one might need to run it multiple times or check statistical properties.
        let source_indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let sampler = SubsetRandomSampler::new(source_indices.clone());
        let sampled_indices1: Vec<usize> = sampler.iter(20).collect();
        let sampled_indices2: Vec<usize> = sampler.iter(20).collect();

        assert_eq!(sampled_indices1.len(), source_indices.len());
        assert_eq!(sampled_indices2.len(), source_indices.len());

        // It's highly probable they are not identical if shuffled.
        // If source_indices.len() is small, this might be flaky.
        if source_indices.len() > 5 { // Only assert if the list is reasonably long
             assert_ne!(sampled_indices1, sampled_indices2, "Two iterations with shuffle should ideally produce different orders for non-trivial inputs.");
        }
        
        let set1: HashSet<usize> = sampled_indices1.into_iter().collect();
        let set2: HashSet<usize> = sampled_indices2.into_iter().collect();
        let source_set: HashSet<usize> = source_indices.into_iter().collect();
        assert_eq!(set1, source_set);
        assert_eq!(set2, source_set);
    }
} 