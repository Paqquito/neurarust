// neurarust-data/src/samplers/subset_random_sampler_test.rs

use super::*;
use std::collections::HashSet;

#[test]
fn test_subset_random_sampler_new() {
    let indices = vec![1, 2, 3];
    let sampler = SubsetRandomSampler::new(indices.clone());
    assert_eq!(sampler.indices, indices);
}

#[test]
fn test_subset_random_sampler_len() {
    let indices_empty = vec![];
    let sampler_empty = SubsetRandomSampler::new(indices_empty);
    assert_eq!(sampler_empty.len(0), 0);
    assert_eq!(sampler_empty.len(10), 0); // dataset_len is ignored

    let indices_non_empty = vec![10, 20, 5];
    let sampler_non_empty = SubsetRandomSampler::new(indices_non_empty.clone());
    assert_eq!(sampler_non_empty.len(0), indices_non_empty.len());
    assert_eq!(sampler_non_empty.len(100), indices_non_empty.len());
}

#[test]
fn test_subset_random_sampler_iter_non_empty() {
    let source_indices = vec![1, 5, 2, 8, 3];
    let sampler = SubsetRandomSampler::new(source_indices.clone());
    let iterated_indices: Vec<usize> = sampler.iter(100).collect(); // dataset_len is ignored

    assert_eq!(iterated_indices.len(), source_indices.len());

    let iterated_set: HashSet<usize> = iterated_indices.iter().cloned().collect();
    let source_set: HashSet<usize> = source_indices.iter().cloned().collect();

    assert_eq!(iterated_set, source_set, "All original indices must be present in the output exactly once");
}

#[test]
fn test_subset_random_sampler_iter_empty() {
    let sampler = SubsetRandomSampler::new(vec![]);
    let mut iter = sampler.iter(10); // dataset_len is ignored
    assert_eq!(iter.next(), None);
}

#[test]
fn test_subset_random_sampler_iter_single_element() {
    let indices = vec![42];
    let sampler = SubsetRandomSampler::new(indices.clone());
    let iterated_indices: Vec<usize> = sampler.iter(50).collect();
    assert_eq!(iterated_indices, indices);
}

#[test]
fn test_subset_random_sampler_iter_shuffles() {
    let source_indices: Vec<usize> = (0..100).collect();
    let sampler = SubsetRandomSampler::new(source_indices.clone());

    // Run multiple times to increase chance of detecting non-shuffling
    let mut all_same_order = true;
    let first_iteration: Vec<usize> = sampler.iter(0).collect();

    for _ in 0..10 {
        let current_iteration: Vec<usize> = sampler.iter(0).collect();
        assert_eq!(current_iteration.len(), source_indices.len());
        if current_iteration != first_iteration {
            all_same_order = false;
            break;
        }
    }
    // With 100 elements, it's highly improbable they remain in the same order across 10 shuffles
    // if shuffling is working. This is not a foolproof test for perfect randomness but checks for basic shuffling.
    if source_indices.len() > 1 { // Shuffling a single element or empty list doesn't change order
        assert!(!all_same_order, "SubsetRandomSampler did not appear to shuffle indices. Note: this test is probabilistic.");
    }
} 