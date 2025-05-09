// neurarust-data/src/samplers/random_sampler.rs

use super::traits::Sampler;
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt::Debug;

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

#[cfg(test)]
#[path = "random_sampler_test.rs"]
mod tests; 