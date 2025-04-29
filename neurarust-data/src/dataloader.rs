use crate::Dataset;
use rand::seq::SliceRandom; // For shuffling
use rand::thread_rng;
use neurarust_core::tensor::Tensor;
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::ops::AddAssign;
use std::marker::PhantomData; // Import PhantomData

/// Provides an iterator over a Dataset, yielding batches of data.
#[derive(Debug)]
pub struct DataLoader<I, T, D: Dataset<Item = (Tensor<I>, Tensor<T>)>> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>, // Indices to iterate over, possibly shuffled
    current_pos: usize,  // Current position in the indices vector
    _marker: PhantomData<(I, T)>, // Add PhantomData for unused I, T
}

impl<I, T, D: Dataset<Item = (Tensor<I>, Tensor<T>)>> DataLoader<I, T, D> {
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
        let len = dataset.len(); // Store length before moving dataset
        let mut indices: Vec<usize> = (0..len).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_pos: 0,
            _marker: PhantomData, // Initialize PhantomData
        }
    }

    /// Resets the iterator, optionally shuffling the indices if `shuffle` is true.
    #[allow(dead_code)] // Keep reset for potential future use (e.g., manual epoch reset)
    fn reset(&mut self) {
        self.current_pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
}

/// Iterator implementation for DataLoader.
/// 
/// Yields collated batches of data as tuples of Tensors `(Tensor<I>, Tensor<T>)`.
impl<I, T, D> Iterator for DataLoader<I, T, D>
where
    D: Dataset<Item = (Tensor<I>, Tensor<T>)>, // Dataset provides the tuples
    // Add bounds required by collate_batch
    I: Clone + Debug + Default + Zero + One + AddAssign + 'static, 
    T: Clone + Debug + Default + Zero + One + AddAssign + 'static,
{
    // Change the associated type to the collated batch tuple
    type Item = (Tensor<I>, Tensor<T>); 

    fn next(&mut self) -> Option<Self::Item> {
        // --- Début d'époque / Réinitialisation --- 
        // Si on a atteint la fin, réinitialiser pour l'époque suivante
        if self.current_pos >= self.indices.len() {
            self.current_pos = 0; // Reset position
            if self.shuffle {      // Reshuffle if needed
                self.indices.shuffle(&mut thread_rng());
            }
            // Si le dataset est vide après reset, retourner None
            if self.indices.is_empty() { 
                 return None;
            }
        }

        // --- Logique de batch actuelle --- 
        // Déterminer l'index de fin pour le batch courant
        let end = (self.current_pos + self.batch_size).min(self.indices.len());
        
        // Collecter les items pour le batch courant
        let batch_indices = &self.indices[self.current_pos..end];
        let batch_items: Vec<(Tensor<I>, Tensor<T>)> = batch_indices
            .iter()
            .map(|&index| self.dataset.get(index).clone()) // Clone item from dataset
            .collect();

        // Avancer la position courante
        self.current_pos = end;

        // Assembler le batch
        collate_batch(batch_items)
    }
}

/// Collates a vector of dataset items (tuples of Tensors) into a single tuple of batch Tensors.
/// 
/// Assumes each item in the batch is a tuple `(InputTensor, TargetTensor)`. 
/// Stacks the input tensors along the first dimension (dim 0) to create a batch input tensor, 
/// and does the same for target tensors.
///
/// # Arguments
/// * `batch` - A vector where each element is `Dataset::Item` (assumed to be `(Tensor<I>, Tensor<T>)`).
///
/// # Returns
/// A tuple `(Tensor<I>, Tensor<T>)` where the tensors represent the collated batch.
/// Returns `None` if the input batch is empty.
/// 
/// # Panics
/// Panics if `Tensor::stack` fails (e.g., inconsistent shapes).
pub fn collate_batch<I, T>(batch: Vec<(Tensor<I>, Tensor<T>)>) -> Option<(Tensor<I>, Tensor<T>)> 
where 
    I: Clone + Debug + Default + Zero + One + AddAssign + 'static, // AddAssign needed by Tensor::stack -> stack_op
    T: Clone + Debug + Default + Zero + One + AddAssign + 'static, // AddAssign needed by Tensor::stack -> stack_op
{
    if batch.is_empty() {
        return None;
    }

    // Separate input and target tensors
    let mut input_tensors = Vec::with_capacity(batch.len());
    let mut target_tensors = Vec::with_capacity(batch.len());
    for (input, target) in batch {
        input_tensors.push(input); // Collect owned tensors
        target_tensors.push(target);
    }

    // Stack tensors along a new batch dimension (dim 0)
    let batch_inputs = Tensor::stack(&input_tensors, 0)
        .expect("Failed to stack input tensors into a batch. Ensure all input tensors have the same shape.");
    let batch_targets = Tensor::stack(&target_tensors, 0)
        .expect("Failed to stack target tensors into a batch. Ensure all target tensors have the same shape.");
    
    Some((batch_inputs, batch_targets))
}


// --- Tests --- 
#[cfg(test)]
mod tests {
    use super::*;
    use crate::VecDataset; // Removed unused 'Dataset' import
    use neurarust_core::tensor::Tensor;
    use std::collections::HashSet;
    
    
    

    #[test]
    fn test_dataloader_basic_iteration_no_shuffle() {
        let inputs_data = vec![1i32, 2, 3, 4, 5];
        let targets_data = vec![10i32, 20, 30, 40, 50];
        let _dataset_len = inputs_data.len(); // Prefixed with _ as it's unused in this test
        let batch_size = 2;

        let inputs = inputs_data.iter().map(|&x| Tensor::new(vec![x], vec![1])).collect::<Vec<_>>();
        let targets = targets_data.iter().map(|&x| Tensor::new(vec![x], vec![1])).collect::<Vec<_>>();
        
        type MyDataset = VecDataset<i32, i32>;
        let dataset = MyDataset::new(inputs.clone(), targets.clone());
        let mut dataloader: DataLoader<i32, i32, _> = DataLoader::new(dataset, batch_size, false);

        // Epoch 1
        // Batch 1
        let (batch1_inputs, batch1_targets) = dataloader.next().unwrap();
        assert_eq!(batch1_inputs.shape(), vec![2, 1]);
        assert_eq!(batch1_inputs.data().to_vec(), vec![1, 2]);
        assert_eq!(batch1_targets.shape(), vec![2, 1]);
        assert_eq!(batch1_targets.data().to_vec(), vec![10, 20]);

        // Batch 2
        let (batch2_inputs, batch2_targets) = dataloader.next().unwrap();
        assert_eq!(batch2_inputs.shape(), vec![2, 1]);
        assert_eq!(batch2_inputs.data().to_vec(), vec![3, 4]);
        assert_eq!(batch2_targets.shape(), vec![2, 1]);
        assert_eq!(batch2_targets.data().to_vec(), vec![30, 40]);
        
        // Batch 3
        let (batch3_inputs, batch3_targets) = dataloader.next().unwrap();
        assert_eq!(batch3_inputs.shape(), vec![1, 1]);
        assert_eq!(batch3_inputs.data().to_vec(), vec![5]);
        assert_eq!(batch3_targets.shape(), vec![1, 1]);
        assert_eq!(batch3_targets.data().to_vec(), vec![50]);

        // Epoch 2 starts automatically
        // Batch 1 (epoch 2)
        let (batch1_epoch2_inputs, batch1_epoch2_targets) = dataloader.next().unwrap();
        assert_eq!(batch1_epoch2_inputs.shape(), vec![2, 1]);
        assert_eq!(batch1_epoch2_inputs.data().to_vec(), vec![1, 2]); // Same order due to shuffle=false
        assert_eq!(batch1_epoch2_targets.shape(), vec![2, 1]);
        assert_eq!(batch1_epoch2_targets.data().to_vec(), vec![10, 20]);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let dataset_len = 100;
        let batch_size = 10;
        let num_batches = (dataset_len + batch_size - 1) / batch_size; // = 10

        let inputs = (0..dataset_len).map(|i| Tensor::new(vec![i as i32], vec![1])).collect::<Vec<_>>();
        let targets = (0..dataset_len).map(|i| Tensor::new(vec![i as i32 * 10], vec![1])).collect::<Vec<_>>();
        type MyDataset = VecDataset<i32, i32>; 
        let dataset = MyDataset::new(inputs, targets);
        let mut dataloader: DataLoader<i32, i32, _> = DataLoader::new(dataset, batch_size, true);

        // Epoch 1
        let mut first_epoch_indices = HashSet::new();
        // Get first batch
        let (batch1_inputs, _) = dataloader.next().unwrap(); 
        for val in batch1_inputs.data().iter() {
            first_epoch_indices.insert(*val);
        }
        assert_eq!(first_epoch_indices.len(), batch_size);

        // Consume the rest of the first epoch (num_batches - 1 more calls)
        for _ in 1..num_batches {
            // Consume batch without checking content, just to advance iterator
            let _ = dataloader.next().unwrap(); 
        }

        // Epoch 2 starts automatically
        let mut second_epoch_indices = HashSet::new();
        // Get first batch of epoch 2
        let (batch1_epoch2_inputs, _) = dataloader.next().unwrap(); 
        for val in batch1_epoch2_inputs.data().iter() {
            second_epoch_indices.insert(*val);
        }
        assert_eq!(second_epoch_indices.len(), batch_size);
        
        // Check that batches from epoch 1 and epoch 2 are different due to shuffling
        assert_ne!(first_epoch_indices, second_epoch_indices, 
                   "First batch indices should differ across shuffled epochs");
    }

    #[test]
    #[should_panic(expected = "batch_size must be greater than 0")]
    fn test_dataloader_zero_batch_size() {
        let dataset: VecDataset<i32, i32> = VecDataset::new(vec![], vec![]);
        let _: DataLoader<i32, i32, _> = DataLoader::new(dataset, 0, false);
    }

     #[test]
    fn test_dataloader_exact_batch_size() {
        let inputs = (0..6).map(|i| Tensor::new(vec![i], vec![1])).collect::<Vec<_>>();
        let targets = (0..6).map(|i| Tensor::new(vec![i*10], vec![1])).collect::<Vec<_>>();
        let dataset = VecDataset::new(inputs.clone(), targets.clone());
        let mut dataloader: DataLoader<i32, i32, _> = DataLoader::new(dataset, 3, false);

        let (batch1_inputs, batch1_targets) = dataloader.next().unwrap();
        assert_eq!(batch1_inputs.shape(), vec![3, 1]);
        assert_eq!(batch1_inputs.data().to_vec(), vec![0i32, 1i32, 2i32]);
        assert_eq!(batch1_targets.shape(), vec![3, 1]);
        assert_eq!(batch1_targets.data().to_vec(), vec![0i32*10, 1i32*10, 2i32*10]);

        let (batch2_inputs, batch2_targets) = dataloader.next().unwrap();
        assert_eq!(batch2_inputs.shape(), vec![3, 1]);
        assert_eq!(batch2_inputs.data().to_vec(), vec![3i32, 4i32, 5i32]);
        assert_eq!(batch2_targets.shape(), vec![3, 1]);
        assert_eq!(batch2_targets.data().to_vec(), vec![3i32*10, 4i32*10, 5i32*10]);
    }

    #[test]
    fn test_collate_batch_simple() {
        let batch = vec![
            (Tensor::new(vec![1.0f64, 2.0], vec![2]), Tensor::new(vec![0i32], vec![1])),
            (Tensor::new(vec![3.0, 4.0], vec![2]), Tensor::new(vec![1i32], vec![1])),
        ];

        type InputType = f64;
        type TargetType = i32;

        if let Some((batch_inputs, batch_targets)) = collate_batch::<InputType, TargetType>(batch) {
            assert_eq!(batch_inputs.shape(), vec![2, 2]);
            assert_eq!(batch_inputs.data().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(batch_targets.shape(), vec![2, 1]);
            assert_eq!(batch_targets.data().to_vec(), vec![0, 1]);
        } else {
            panic!("Collation failed for non-empty batch");
        }

        let empty_batch: Vec<(Tensor<f64>, Tensor<i32>)> = vec![];
        assert!(collate_batch::<InputType, TargetType>(empty_batch).is_none());
    }
}