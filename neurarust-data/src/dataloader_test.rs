#[cfg(test)]
mod tests {
    use crate::dataloader::DataLoader;
    use crate::datasets::VecDataset;
    use crate::samplers::sequential_sampler::SequentialSampler;

    #[test]
    fn test_dataloader_sequential() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let dataset = VecDataset::new(data);
        let sampler = SequentialSampler::new();
        let mut loader = DataLoader::new(dataset, 2, sampler, false, None);
        let mut batches = Vec::new();
        while let Some(batch) = loader.next() {
            let batch = batch.expect("Batch should not error");
            batches.push(batch);
        }
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![1, 2]);
        assert_eq!(batches[1], vec![3, 4]);
        assert_eq!(batches[2], vec![5, 6]);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let data = vec![1, 2, 3, 4, 5];
        let dataset = VecDataset::new(data);
        let sampler = SequentialSampler::new();
        let mut loader = DataLoader::new(dataset, 2, sampler, true, None);
        
        let mut batches = Vec::new();
        while let Some(batch) = loader.next() {
            let batch = batch.expect("Batch should not error");
            batches.push(batch);
        }
        
        assert_eq!(batches.len(), 2); // Le dernier batch de taille 1 est ignoré
        assert_eq!(batches[0], vec![1, 2]);
        assert_eq!(batches[1], vec![3, 4]);
    }
} 