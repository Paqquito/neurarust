use neurarust_core::NeuraRustError;
use super::traits::Dataset; // Modifié pour pointer vers le nouveau chemin du trait

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
#[path = "vec_dataset_test.rs"]
mod tests; 