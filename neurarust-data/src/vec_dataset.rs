use neurarust_core::NeuraRustError;
use crate::dataset::Dataset;

/// A simple dataset that wraps a `Vec` of items.
///
/// Each item in the `Vec` corresponds to a sample in the dataset.
///
/// # Type Parameters
///
/// * `T`: The type of the items stored in the dataset. Must be `Clone + Send + \'static`.
#[derive(Debug, Clone)]
pub struct VecDataset<T: Clone + Send + 'static> {
    data: Vec<T>,
}

impl<T: Clone + Send + 'static> VecDataset<T> {
    /// Creates a new `VecDataset` from a vector of items.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of items that will constitute the dataset.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T: Clone + Send + 'static> Dataset for VecDataset<T> {
    type Item = T;

    /// Returns the item at the given index.
    ///
    /// Clones the item before returning.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the item to retrieve.
    ///
    /// # Errors
    ///
    /// Returns `NeuraRustError::IndexOutOfBounds` if the index is out of bounds.
    fn get(&self, index: usize) -> Result<Self::Item, NeuraRustError> {
        self.data.get(index).cloned().ok_or_else(|| {
            NeuraRustError::IndexOutOfBounds {
                index: vec![index],
                shape: vec![self.data.len()],
            }
        })
    }

    /// Returns the total number of items in the dataset.
    fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dataset_new() {
        let data = vec![1, 2, 3, 4, 5];
        let dataset = VecDataset::new(data.clone());
        assert_eq!(dataset.len(), 5);
    }

    #[test]
    fn test_vec_dataset_get_valid_index() {
        let data = vec![10, 20, 30];
        let dataset = VecDataset::new(data);
        assert_eq!(dataset.get(0).unwrap(), 10);
        assert_eq!(dataset.get(1).unwrap(), 20);
        assert_eq!(dataset.get(2).unwrap(), 30);
    }

    #[test]
    fn test_vec_dataset_get_invalid_index() {
        let data: Vec<i32> = vec![10, 20, 30];
        let dataset = VecDataset::new(data);
        match dataset.get(3) {
            Err(NeuraRustError::IndexOutOfBounds { index: _, shape: _ }) => assert!(true),
            _ => assert!(false, "Expected IndexOutOfBounds error"),
        }
    }
    
    #[test]
    fn test_vec_dataset_get_empty() {
        let data: Vec<i32> = Vec::new();
        let dataset = VecDataset::new(data);
        assert!(dataset.get(0).is_err());
    }

    #[test]
    fn test_vec_dataset_len_empty() {
        let data: Vec<i32> = Vec::new();
        let dataset = VecDataset::new(data);
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_vec_dataset_len_non_empty() {
        let data = vec!["a", "b", "c"];
        let dataset = VecDataset::new(data);
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_vec_dataset_item_type() {
        // Test with a more complex item type (tuple)
        let data = vec![(1, "one"), (2, "two")];
        let dataset = VecDataset::new(data);
        assert_eq!(dataset.get(0).unwrap(), (1, "one"));
        type ExpectedItem = (i32, &'static str);
        let _item: ExpectedItem = dataset.get(0).unwrap(); // type check
    }
} 