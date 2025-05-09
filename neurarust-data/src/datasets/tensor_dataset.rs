use neurarust_core::{ops::view::SliceArg, NeuraRustError, Tensor};
use super::traits::Dataset; // Modifié pour pointer vers le nouveau chemin du trait

/// A dataset composed of one or more tensors.
///
/// All tensors in a `TensorDataset` must have the same size in their first dimension.
/// This first dimension is treated as the batch dimension. When an item is fetched
/// using `get(index)`, it returns a new `Vec<Tensor>` where each tensor is a slice
/// of the original tensors at the specified index along the first dimension.
#[derive(Debug, Clone)]
pub struct TensorDataset {
    tensors: Vec<Tensor>,
    length: usize,
}

impl TensorDataset {
    /// Creates a new `TensorDataset` from a vector of tensors.
    ///
    /// All tensors must have the same size in their first dimension.
    /// If the vector of tensors is empty, the dataset will have a length of 0.
    /// If any tensor is a scalar or does not have at least one dimension, an error is returned.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A vector of `Tensor` objects.
    ///
    /// # Errors
    ///
    /// Returns `NeuraRustError::ShapeMismatch` if tensors have inconsistent first dimension sizes.
    /// Returns `NeuraRustError::RankMismatch` if any tensor is a scalar (rank 0).
    pub fn new(tensors: Vec<Tensor>) -> Result<Self, NeuraRustError> {
        if tensors.is_empty() {
            return Ok(Self {
                tensors,
                length: 0,
            });
        }

        let first_tensor_shape = tensors[0].shape();
        if first_tensor_shape.is_empty() {
            return Err(NeuraRustError::RankMismatch {
                expected: 1, // Expecting at least 1 dimension
                actual: 0,
            });
        }
        let expected_len = first_tensor_shape[0];

        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let current_shape = tensor.shape();
            if current_shape.is_empty() {
                return Err(NeuraRustError::RankMismatch {
                    expected: 1,
                    actual: 0,
                });
            }
            if current_shape[0] != expected_len {
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("First dimension of size {}", expected_len),
                    actual: format!(
                        "First dimension of size {} for tensor at index {}",
                        current_shape[0], i
                    ),
                    operation: "TensorDataset::new".to_string(),
                });
            }
        }

        Ok(Self {
            tensors,
            length: expected_len,
        })
    }
}

impl Dataset for TensorDataset {
    type Item = Vec<Tensor>;

    fn get(&self, index: usize) -> Result<Self::Item, NeuraRustError> {
        if index >= self.length {
            return Err(NeuraRustError::IndexOutOfBounds {
                index: vec![index],
                shape: vec![self.length], 
            });
        }

        let mut item_slices = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let mut slice_indices = Vec::with_capacity(tensor.rank());
            
            let first_dim_slice = SliceArg::Slice(index as isize, (index + 1) as isize, 1);
            slice_indices.push(first_dim_slice);

            for dim_size in tensor.shape().iter().skip(1) {
                slice_indices.push(SliceArg::Slice(0, *dim_size as isize, 1));
            }
            
            let slice = tensor.slice(&slice_indices)?;
            item_slices.push(slice);
        }
        Ok(item_slices)
    }

    fn len(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
#[path = "tensor_dataset_test.rs"]
mod tests; 