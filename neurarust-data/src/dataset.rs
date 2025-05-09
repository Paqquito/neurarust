use neurarust_core::NeuraRustError;

/// Represents a dataset that can be iterated over and accessed by index.
///
/// A dataset is a collection of items, where each item can be a single tensor,
/// a tuple of tensors (e.g., (features, label)), or any other custom type
/// that implements `Send + 'static`.
pub trait Dataset {
    /// The type of a single item returned by the dataset.
    ///
    /// This type must be `Send` and `'static` to allow for potential
    /// multi-threaded data loading in the future.
    type Item: Send + 'static;

    /// Returns the item at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the item to retrieve.
    ///
    /// # Errors
    ///
    /// Returns `NeuraRustError` if the index is out of bounds or if there's
    /// an issue retrieving the item.
    fn get(&self, index: usize) -> Result<Self::Item, NeuraRustError>;

    /// Returns the total number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
} 