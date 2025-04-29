
/// Trait representing a dataset.
/// 
/// A dataset provides access to individual data samples (e.g., input features 
/// and corresponding target labels) via an index.
/// 
/// `Item` is the type returned by accessing a single sample. It's often a tuple
/// like `(Tensor<InputType>, Tensor<TargetType>)`.
pub trait Dataset {
    /// The type of a single item returned by the dataset.
    type Item;

    /// Returns the data sample at the given index.
    /// 
    /// # Panics
    /// Panics if the index is out of bounds.
    fn get(&self, index: usize) -> Self::Item;

    /// Returns the total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns true if the dataset contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Placeholder for DataLoader and VecDataset implementation
pub mod dataloader;
pub mod vec_dataset;

// Re-export main components
pub use dataloader::DataLoader;
pub use vec_dataset::VecDataset;

// Remove default lib content
// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }
// 
// #[cfg(test)]
// mod tests {
//     use super::*;
// 
//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
