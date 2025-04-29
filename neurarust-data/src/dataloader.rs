use crate::Dataset;
use rand::seq::SliceRandom; // For shuffling
use rand::thread_rng;

/// Provides an iterator over a Dataset, yielding batches of data.
#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>, // Indices to iterate over, possibly shuffled
    current_pos: usize,  // Current position in the indices vector
}

impl<D: Dataset> DataLoader<D> {
    /// Creates a new DataLoader.
    /// 
    /// # Arguments
    /// * `dataset` - The dataset to load data from.
    /// * `batch_size` - The number of samples per batch.
    /// * `shuffle` - Whether to shuffle the data at the beginning of each epoch.
    ///
    /// # Panics
    /// Panics if `batch_size` is 0.
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_pos: 0,
        }
    }

    /// Resets the iterator, optionally shuffling the indices if `shuffle` is true.
    fn reset(&mut self) {
        self.current_pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
}

/// Iterator implementation for DataLoader.
/// 
/// Yields batches of data. Each batch is currently returned as a Vec of items 
/// obtained from `dataset.get()`. 
/// TODO: Implement collation to combine items into batch tensors.
impl<D> Iterator for DataLoader<D>
where
    D: Dataset,
    D::Item: Clone, // Item needs to be cloneable to put into Vec
{
    // The type of the items yielded by the iterator.
    // For now, it's a Vec of individual dataset items.
    type Item = Vec<D::Item>; 

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we reached the end of the dataset for this epoch
        if self.current_pos >= self.indices.len() {
            // Reset for the next potential epoch (or return None if called again)
            self.reset();
            return None; 
        }

        // Determine the end index for the current batch
        let end = (self.current_pos + self.batch_size).min(self.indices.len());
        
        // Collect items for the current batch
        let batch_indices = &self.indices[self.current_pos..end];
        let mut batch_data = Vec::with_capacity(batch_indices.len());
        for &index in batch_indices {
            // Clone the item from the dataset
            batch_data.push(self.dataset.get(index).clone()); 
        }

        // Move the current position forward
        self.current_pos = end;

        // Return the collected batch
        Some(batch_data)
    }
}

// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VecDataset, Dataset};
    use neurarust_core::tensor::Tensor;
    use std::collections::HashSet;

    #[test]
    fn test_dataloader_basic_iteration_no_shuffle() {
        let inputs = vec![
            Tensor::new(vec![1i32], vec![1]), Tensor::new(vec![2i32], vec![1]), 
            Tensor::new(vec![3i32], vec![1]), Tensor::new(vec![4i32], vec![1]),
            Tensor::new(vec![5i32], vec![1]),
        ];
        let targets = vec![
            Tensor::new(vec![format!("1a")], vec![1]), Tensor::new(vec![format!("2a")], vec![1]), 
            Tensor::new(vec![format!("3a")], vec![1]), Tensor::new(vec![format!("4a")], vec![1]),
            Tensor::new(vec![format!("5a")], vec![1]),
        ];
        type MyDataset = VecDataset<i32, String>;
        let dataset = MyDataset::new(inputs.clone(), targets.clone());
        let mut dataloader = DataLoader::new(dataset, 2, false);

        // Batch 1
        let batch1 = dataloader.next().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0], (inputs[0].clone(), targets[0].clone()));
        assert_eq!(batch1[1], (inputs[1].clone(), targets[1].clone()));

        // Batch 2
        let batch2 = dataloader.next().unwrap();
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2[0], (inputs[2].clone(), targets[2].clone()));
        assert_eq!(batch2[1], (inputs[3].clone(), targets[3].clone()));
        
        // Batch 3 (last batch, might be smaller)
        let batch3 = dataloader.next().unwrap();
        assert_eq!(batch3.len(), 1);
        assert_eq!(batch3[0], (inputs[4].clone(), targets[4].clone()));

        // End of epoch
        assert!(dataloader.next().is_none());

        // Check reset for next epoch (no shuffle)
        let batch1_epoch2 = dataloader.next().unwrap();
        assert_eq!(batch1_epoch2.len(), 2);
        assert_eq!(batch1_epoch2[0], (inputs[0].clone(), targets[0].clone()));
        assert_eq!(batch1_epoch2[1], (inputs[1].clone(), targets[1].clone()));
    }

    #[test]
    fn test_dataloader_shuffle() {
        let inputs = (0..100).map(|i| Tensor::new(vec![i as i32], vec![1])).collect::<Vec<_>>();
        let targets = (0..100).map(|i| Tensor::new(vec![format!("{}", i)], vec![1])).collect::<Vec<_>>();
        type MyDataset = VecDataset<i32, String>;
        let dataset = MyDataset::new(inputs, targets);
        let mut dataloader = DataLoader::new(dataset, 10, true);

        let mut first_batch_indices = HashSet::new();
        let batch1 = dataloader.next().unwrap();
        for (input, _) in batch1 {
            first_batch_indices.insert(input.data()[0]);
        }
        assert_eq!(first_batch_indices.len(), 10);

        // Complete the epoch
        while let Some(_) = dataloader.next() {}
        assert!(dataloader.next().is_none());

        // Start next epoch, indices should be reshuffled
        let mut second_batch_indices = HashSet::new();
        let batch1_epoch2 = dataloader.next().unwrap();
        for (input, _) in batch1_epoch2 {
            second_batch_indices.insert(input.data()[0]);
        }
        assert_eq!(second_batch_indices.len(), 10);
        
        // Very unlikely the batches are identical if shuffled
        assert_ne!(first_batch_indices, second_batch_indices, "Batches should be different across shuffled epochs");
    }

    #[test]
    #[should_panic(expected = "batch_size must be greater than 0")]
    fn test_dataloader_zero_batch_size() {
        let dataset: VecDataset<i32, i32> = VecDataset::new(vec![], vec![]);
        DataLoader::new(dataset, 0, false);
    }

     #[test]
    fn test_dataloader_exact_batch_size() {
        let inputs = (0..6).map(|i| Tensor::new(vec![i], vec![1])).collect::<Vec<_>>();
        let targets = (0..6).map(|i| Tensor::new(vec![i*10], vec![1])).collect::<Vec<_>>();
        let dataset = VecDataset::new(inputs.clone(), targets.clone());
        let mut dataloader = DataLoader::new(dataset, 3, false);

        let batch1 = dataloader.next().unwrap();
        assert_eq!(batch1.len(), 3);
        assert_eq!(batch1[0], (inputs[0].clone(), targets[0].clone()));
        assert_eq!(batch1[2], (inputs[2].clone(), targets[2].clone()));

        let batch2 = dataloader.next().unwrap();
        assert_eq!(batch2.len(), 3);
        assert_eq!(batch2[0], (inputs[3].clone(), targets[3].clone()));
        assert_eq!(batch2[2], (inputs[5].clone(), targets[5].clone()));

        assert!(dataloader.next().is_none());
    }
} 