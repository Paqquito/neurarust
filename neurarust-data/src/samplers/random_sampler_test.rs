// neurarust-data/src/samplers/random_sampler_test.rs

use super::*;
use std::collections::HashSet;

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
    // It's possible for duplicates with replacement, so not checking uniqueness like before.
}

#[test]
fn test_random_sampler_iter_empty_dataset() {
    let sampler_no_replace = RandomSampler::new(false, None);
    assert_eq!(sampler_no_replace.iter(0).count(), 0);

    let sampler_replace = RandomSampler::new(true, None);
    assert_eq!(sampler_replace.iter(0).count(), 0);

    let sampler_no_replace_samples = RandomSampler::new(false, Some(5));
    assert_eq!(sampler_no_replace_samples.iter(0).count(), 0);

    let sampler_replace_samples = RandomSampler::new(true, Some(5));
    assert_eq!(sampler_replace_samples.iter(0).count(), 0);
}


#[test]
fn test_random_sampler_iter_no_replacement_num_samples_none() {
    // num_samples is None, should default to dataset_len
    let dataset_len = 7;
    let sampler = RandomSampler::new(false, None);
    let indices: Vec<usize> = sampler.iter(dataset_len).collect();
    assert_eq!(indices.len(), dataset_len);
    let unique_indices: HashSet<usize> = indices.into_iter().collect();
    assert_eq!(unique_indices.len(), dataset_len);
}

#[test]
fn test_random_sampler_iter_with_replacement_num_samples_none() {
    // num_samples is None, should default to dataset_len
    let dataset_len = 6;
    let sampler = RandomSampler::new(true, None);
    let indices: Vec<usize> = sampler.iter(dataset_len).collect();
    assert_eq!(indices.len(), dataset_len);
    // Can't guarantee uniqueness with replacement, but all indices must be valid
    for &index in &indices {
        assert!(index < dataset_len);
    }
} 