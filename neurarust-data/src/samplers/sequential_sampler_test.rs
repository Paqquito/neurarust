// neurarust-data/src/samplers/sequential_sampler_test.rs

use super::*;
// Pas besoin d'importer Sampler explicitement ici si les méthodes testées sont sur SequentialSampler directement
// ou si Sampler est en scope via `super::*` et l'utilisation de `super::super::traits::Sampler` dans le module parent.

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