use neurarust_core::{ops::view::SliceArg, NeuraRustError, Tensor};
use crate::dataset::Dataset;

/// A dataset composed of one or more tensors.
///
/// All tensors in a `TensorDataset` must have the same size in their first dimension.
/// This first dimension is treated as the batch dimension. When an item is fetched
/// using `get(index)`, it returns a new `Vec<Tensor>` where each tensor is a slice
/// of the original tensors at the specified index along the first dimension.
///
/// # Examples
///
/// ```
/// // This example would require Tensor to be available and constructible.
/// // use neurarust_core::Tensor;
/// // use neurarust_data::{TensorDataset, Dataset};
/// //
/// // let features = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
/// // let labels = Tensor::new_f32(vec![0.0, 1.0, 0.0], &[3, 1]).unwrap();
/// // let tensor_dataset = TensorDataset::new(vec![features, labels]).unwrap();
/// //
/// // assert_eq!(tensor_dataset.len(), 3);
/// // let (feature_slice, label_slice) = match tensor_dataset.get(0).unwrap().as_slice() {
/// //     [f, l] => (f, l),
/// //     _ => panic!("Expected two tensors"),
/// // };
/// // // feature_slice would have shape [2] (or [1, 2] if keep_dim=true for slice)
/// // // label_slice would have shape [1] (or [1, 1] if keep_dim=true for slice)
/// ```
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

    /// Returns a vector of tensor slices corresponding to the given index.
    ///
    /// Each tensor in the returned vector is a slice of an original tensor
    /// along its first dimension at the specified `index`.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the item (slice) to retrieve.
    ///
    /// # Errors
    ///
    /// Returns `NeuraRustError::IndexOutOfBounds` if `index` is out of bounds.
    /// Returns `NeuraRustError` if slicing fails for any tensor.
    fn get(&self, index: usize) -> Result<Self::Item, NeuraRustError> {
        if index >= self.length {
            return Err(NeuraRustError::IndexOutOfBounds {
                index: vec![index],
                shape: vec![self.length], // Representing dataset length as a 1D shape
            });
        }

        let mut item_slices = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            // SliceInfo: [start, end, step]
            // We want to select the `index`-th element along the first dimension.
            // So, for the first dimension, the slice is [index, index + 1, 1].
            // For other dimensions, we take the full range [0, dim_size, 1].
            let mut slice_indices = Vec::with_capacity(tensor.rank());
            
            let first_dim_slice = SliceArg::Slice(index as isize, (index + 1) as isize, 1);
            slice_indices.push(first_dim_slice);

            for dim_size in tensor.shape().iter().skip(1) {
                slice_indices.push(SliceArg::Slice(0, *dim_size as isize, 1));
            }
            
            // The slice operation typically reduces the rank if the sliced dimension becomes 1.
            // We might need to unsqueeze it if we want to preserve the rank of the slice,
            // or ensure users are aware of the rank reduction.
            // For now, we accept the rank reduction which is typical (e.g. PyTorch behavior).
            let slice = tensor.slice(&slice_indices)?;
            item_slices.push(slice);
        }
        Ok(item_slices)
    }

    /// Returns the total number of items (slices) in the dataset.
    ///
    /// This is determined by the size of the first dimension of the tensors.
    fn len(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurarust_core::tensor::create::from_vec_f32;
    use neurarust_core::Tensor;

    // Helper to create a basic tensor for testing.
    fn create_test_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        from_vec_f32(data, shape.to_vec()).unwrap()
    }

    fn create_test_tensor_requires_grad(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let t = from_vec_f32(data, shape.to_vec()).unwrap();
        t.set_requires_grad(true).unwrap(); // Set requires_grad to true
        t
    }


    #[test]
    fn test_tensor_dataset_new_empty() {
        let dataset = TensorDataset::new(vec![]).unwrap();
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_tensor_dataset_new_single_tensor() {
        let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let dataset = TensorDataset::new(vec![t1]).unwrap();
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_tensor_dataset_new_multiple_tensors_valid() {
        let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let t2 = create_test_tensor(vec![0.0, 1.0, 0.0], &[3, 1]);
        let dataset = TensorDataset::new(vec![t1, t2]).unwrap();
        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_tensor_dataset_new_rank_mismatch_scalar() {
        let t1 = create_test_tensor(vec![1.0], &[]); // scalar
        let err = TensorDataset::new(vec![t1]).err().unwrap();
        match err {
            NeuraRustError::RankMismatch { expected, actual } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("Expected RankMismatch error"),
        }
    }
    
    #[test]
    fn test_tensor_dataset_new_rank_mismatch_in_list() {
        let t1 = create_test_tensor(vec![1.0, 2.0], &[2,1]);
        let t2 = create_test_tensor(vec![3.0], &[]); // scalar
        let err = TensorDataset::new(vec![t1, t2]).err().unwrap();
        match err {
            NeuraRustError::RankMismatch { expected, actual } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("Expected RankMismatch error"),
        }
    }


    #[test]
    fn test_tensor_dataset_new_shape_mismatch() {
        let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let t2 = create_test_tensor(vec![0.0, 1.0, 0.0], &[3, 1]); // Different first dimension
        let err = TensorDataset::new(vec![t1, t2]).err().unwrap();
        match err {
            NeuraRustError::ShapeMismatch { .. } => assert!(true),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_tensor_dataset_get_valid_index_single_tensor() {
        let t1 = create_test_tensor_requires_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let dataset = TensorDataset::new(vec![t1]).unwrap();

        let item0 = dataset.get(0).unwrap();
        assert_eq!(item0.len(), 1);
        assert_eq!(item0[0].shape(), &[1, 2]); // Slice reduces first dim to 1, then squeeze
        assert_eq!(item0[0].get_f32_data().unwrap(), vec![1.0, 2.0]);
        assert!(item0[0].requires_grad(), "Grad requirement should propagate");


        let item2 = dataset.get(2).unwrap();
        assert_eq!(item2.len(), 1);
        assert_eq!(item2[0].shape(), &[1, 2]);
        assert_eq!(item2[0].get_f32_data().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_tensor_dataset_get_valid_index_multiple_tensors() {
        let t1 = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let t2 = create_test_tensor(vec![10.0, 20.0, 30.0], &[3, 1]);
        let dataset = TensorDataset::new(vec![t1, t2]).unwrap();

        let item1 = dataset.get(1).unwrap();
        assert_eq!(item1.len(), 2);
        assert_eq!(item1[0].shape(), &[1, 2]);
        assert_eq!(item1[0].get_f32_data().unwrap(), vec![3.0, 4.0]);
        assert_eq!(item1[1].shape(), &[1, 1]);
        assert_eq!(item1[1].get_f32_data().unwrap(), vec![20.0]);
    }
    
    #[test]
    fn test_tensor_dataset_get_valid_index_multiple_tensors_grad_propagation() {
        let t1 = create_test_tensor_requires_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let t2 = create_test_tensor(vec![10.0, 20.0], &[2, 1]); // No grad
        let t3 = create_test_tensor_requires_grad(vec![100.0, 200.0], &[2, 1]);
        let dataset = TensorDataset::new(vec![t1, t2, t3]).unwrap();

        let item0 = dataset.get(0).unwrap();
        assert_eq!(item0.len(), 3);
        assert!(item0[0].requires_grad());
        assert!(!item0[1].requires_grad());
        assert!(item0[2].requires_grad());

        assert_eq!(item0[0].get_f32_data().unwrap(), vec![1.0, 2.0]);
        assert_eq!(item0[1].get_f32_data().unwrap(), vec![10.0]);
        assert_eq!(item0[2].get_f32_data().unwrap(), vec![100.0]);
    }


    #[test]
    fn test_tensor_dataset_get_invalid_index() {
        let t1 = create_test_tensor(vec![1.0, 2.0], &[1, 2]);
        let dataset = TensorDataset::new(vec![t1]).unwrap();
        let err = dataset.get(1).err().unwrap();
        match err {
            NeuraRustError::IndexOutOfBounds { .. } => assert!(true),
            _ => panic!("Expected IndexOutOfBounds error"),
        }
    }

    #[test]
    fn test_tensor_dataset_get_from_empty_dataset() {
        let dataset = TensorDataset::new(vec![]).unwrap();
        let err = dataset.get(0).err().unwrap();
         match err {
            NeuraRustError::IndexOutOfBounds { index, shape } => {
                assert_eq!(index, vec![0]);
                assert_eq!(shape, vec![0]);
            }
            _ => panic!("Expected IndexOutOfBounds error for empty dataset get"),
        }
    }
    
    #[test]
    fn test_tensor_dataset_slice_shape_consistency() {
        // Test with a tensor that would result in a scalar-like slice if not careful
        // e.g. original shape [3, 1], slice at index 0 should be [1] (or [1,1] with keepdim)
        let t1 = create_test_tensor(vec![10.0, 20.0, 30.0], &[3, 1]);
        let dataset = TensorDataset::new(vec![t1]).unwrap();

        let item0 = dataset.get(0).unwrap();
        assert_eq!(item0.len(), 1);
        // The slice [index, index+1, 1] on dim 0 results in shape [1, original_dim1_size, original_dim2_size, ...]
        // So for t1 with shape [3,1], a slice along dim 0 is [1,1]
        assert_eq!(item0[0].shape(), &[1, 1]); 
        assert_eq!(item0[0].get_f32_data().unwrap(), vec![10.0]);

        let item1 = dataset.get(1).unwrap();
        assert_eq!(item1[0].shape(), &[1, 1]);
        assert_eq!(item1[0].get_f32_data().unwrap(), vec![20.0]);
    }

    #[test]
    fn test_tensor_dataset_slice_for_1d_tensor() {
        let t1 = create_test_tensor(vec![10.0, 20.0, 30.0], &[3]); // 1D tensor
        let dataset = TensorDataset::new(vec![t1]).unwrap();

        let item0 = dataset.get(0).unwrap();
        assert_eq!(item0.len(), 1);
        // Slice of a 1D tensor [3] at index 0 (i.e., items from 0 to 1) should result in shape [1]
        assert_eq!(item0[0].shape(), &[1]); 
        assert_eq!(item0[0].get_f32_data().unwrap(), vec![10.0]);

        let item2 = dataset.get(2).unwrap();
        assert_eq!(item2[0].shape(), &[1]);
        assert_eq!(item2[0].get_f32_data().unwrap(), vec![30.0]);
    }
} 