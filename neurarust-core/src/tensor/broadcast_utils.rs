use crate::{error::NeuraRustError, ops::reduction::sum::sum_axes, tensor::Tensor};
use num_traits::Zero;
use std::{fmt::Debug, iter::Sum, ops::AddAssign};

impl<T> Tensor<T>
where
    // Bounds required by this method and sum_axes
    T: Debug
        + Copy
        + Send
        + Sync
        + Zero
        + AddAssign
        + 'static
        + Default
        + PartialEq
        + Sum
        + num_traits::One
        + PartialOrd
        + std::iter::Product,
{
    /// Reduces the tensor (gradient) to match a target shape by summing along broadcasted dimensions.
    ///
    /// This is crucial for the backward pass of operations involving broadcasting.
    /// If the tensor's shape already matches the target shape, a clone is returned.
    /// Otherwise, it identifies dimensions that were broadcasted (either added or expanded from 1)
    /// and sums along those dimensions using `sum_axes`.
    ///
    /// # Arguments
    /// * `target_shape`: The shape the tensor should be reduced to.
    ///
    /// # Returns
    /// * `Ok(Tensor<T>)` containing the reduced tensor.
    /// * `Err(NeuraRustError)` if reduction is not possible or an internal error occurs.
    pub fn reduce_to_shape(
        &self,
        target_shape: &[usize],
    ) -> Result<Tensor<T>, NeuraRustError> {
        let current_shape = self.shape();

        // If shapes match, no reduction needed.
        if current_shape == target_shape {
            return Ok(self.clone());
        }

        // Handle reduction to scalar
        if target_shape.is_empty() {
            if self.numel() == 1 {
                 // Already effectively scalar, reshape if needed (e.g., from [1])
                 return crate::ops::view::reshape_op(self, vec![]);
            }
            // Sum all elements if target is scalar and current is not
            let axes_to_reduce: Vec<usize> = (0..current_shape.len()).collect();
            return sum_axes(self, &axes_to_reduce, false);
        }

        let current_rank = current_shape.len();
        let target_rank = target_shape.len();

        // Check if ranks are compatible for broadcast reduction
        if current_rank < target_rank {
            return Err(NeuraRustError::InternalError(format!(
                "Cannot reduce gradient shape {:?} to target {:?}: Current rank is less than target rank.",
                current_shape, target_shape
            )));
        }

        let rank_diff = current_rank - target_rank;
        let mut axes_to_reduce: Vec<usize> = (0..rank_diff).collect(); // Axes added during broadcast

        for i in 0..target_rank {
            let current_dim = current_shape[rank_diff + i];
            let target_dim = target_shape[i];

            if current_dim != target_dim {
                if target_dim == 1 {
                    // This dimension was expanded from 1, reduce along it.
                    axes_to_reduce.push(rank_diff + i);
                } else {
                    // Shapes mismatch in a non-broadcastable way (target is not 1)
                    return Err(NeuraRustError::InternalError(format!(
                        "Cannot reduce gradient shape {:?} to target {:?}: Incompatible dimension {} (size {} vs target size {}).",
                        current_shape, target_shape, i, current_dim, target_dim
                    )));
                }
            }
        }

        if axes_to_reduce.is_empty() {
             // This case should ideally be caught by the initial `current_shape == target_shape` check,
             // but we add a safeguard.
            if current_shape == target_shape { // Double check shapes if no axes found
                Ok(self.clone())
            } else {
                 Err(NeuraRustError::InternalError(format!(
                     "Cannot reduce gradient shape {:?} to {:?}: Shapes differ but no reduction axes identified.",
                     current_shape, target_shape
                 )))
            }
        } else {
            // Perform summation along the identified axes.
            // keep_dims=true simplifies reshaping later.
            let reduced_grad = sum_axes(self, &axes_to_reduce, true)?;

            // Reshape the result to match the target shape EXACTLY.
            // sum_axes with keep_dims=true will keep the rank, but dimensions summed will be 1.
            // Reshape is needed to remove these dimensions if the target_shape doesn't have them
            // (e.g., reducing [2, 5] to [5] should result in shape [5], not [1, 5]).
            // However, reshape might fail if the tensor is not contiguous after sum_axes.
            // TODO: Ensure sum_axes output is contiguous or handle non-contiguous reshape.
            // For now, assume sum_axes output is contiguous or reshape can handle it.
            if reduced_grad.shape() == target_shape {
                 Ok(reduced_grad)
            } else {
                // If shapes don't match after sum_axes(keep_dims=true), it means we need to squeeze
                // the dimensions that were 1 in the target shape.
                // Let's try reshaping directly.
                 crate::ops::view::reshape_op(&reduced_grad, target_shape.to_vec())
            }
        }
    }

    /// Expands a gradient tensor to match the shape of the original tensor it corresponds to.
    ///
    /// # Arguments
    /// * `target_shape`: The shape the gradient tensor should be expanded to.
    ///
    /// # Returns
    /// * `Ok(Tensor<T>)` containing the expanded gradient tensor.
    /// * `Err(NeuraRustError)` if expansion is not possible or an internal error occurs.
    pub fn expand_gradient_to_shape(
        &self, // The gradient tensor
        target_shape: &[usize],
    ) -> Result<Tensor<T>, NeuraRustError>
    where
        T: Clone // Clone is necessary for Tensor::new
    {
        let current_shape = self.shape();

        // If shapes match, no expansion needed.
        if current_shape == target_shape {
            return Ok(self.clone());
        }

        // Handle expansion to scalar
        if target_shape.is_empty() {
            if self.numel() == 1 {
                 // Already effectively scalar, reshape if needed (e.g., from [1])
                 return crate::ops::view::reshape_op(self, vec![]);
            }
            // Sum all elements if target is scalar and current is not
            let axes_to_expand: Vec<usize> = (0..current_shape.len()).collect();
            return sum_axes(self, &axes_to_expand, false);
        }

        let current_rank = current_shape.len();
        let target_rank = target_shape.len();

        // Check if ranks are compatible for broadcast expansion
        if current_rank > target_rank {
            return Err(NeuraRustError::InternalError(format!(
                "Cannot expand gradient shape {:?} to target {:?}: Current rank is greater than target rank.",
                current_shape, target_shape
            )));
        }

        let rank_diff = target_rank - current_rank;
        let mut axes_to_expand: Vec<usize> = (0..rank_diff).collect(); // Axes added during broadcast

        for i in 0..current_rank {
            let current_dim = current_shape[i];
            let target_dim = target_shape[i];

            if current_dim != target_dim {
                if current_dim == 1 {
                    // This dimension was expanded from 1, expand along it.
                    axes_to_expand.push(i);
                } else {
                    // Shapes mismatch in a non-broadcastable way (current is not 1)
                    return Err(NeuraRustError::InternalError(format!(
                        "Cannot expand gradient shape {:?} to target {:?}: Incompatible dimension {} (size {} vs target size {}).",
                        current_shape, target_shape, i, current_dim, target_dim
                    )));
                }
            }
        }

        if axes_to_expand.is_empty() {
             // This case should ideally be caught by the initial `current_shape == target_shape` check,
             // but we add a safeguard.
            if current_shape == target_shape { // Double check shapes if no axes found
                Ok(self.clone())
            } else {
                 Err(NeuraRustError::InternalError(format!(
                     "Cannot expand gradient shape {:?} to {:?}: Shapes differ but no expansion axes identified.",
                     current_shape, target_shape
                 )))
            }
        } else {
            // Perform expansion along the identified axes.
            // keep_dims=true simplifies reshaping later.
            let expanded_grad = sum_axes(self, &axes_to_expand, true)?;

            // Reshape the result to match the target shape EXACTLY.
            // sum_axes with keep_dims=true will keep the rank, but dimensions summed will be 1.
            // Reshape is needed to remove these dimensions if the target_shape doesn't have them
            // (e.g., expanding [2, 5] to [2, 5] should result in shape [2, 5], not [1, 2, 5]).
            // However, reshape might fail if the tensor is not contiguous after sum_axes.
            // TODO: Ensure sum_axes output is contiguous or handle non-contiguous reshape.
            // For now, assume sum_axes output is contiguous or reshape can handle it.
            if expanded_grad.shape() == target_shape {
                 Ok(expanded_grad)
            } else {
                // If shapes don't match after sum_axes(keep_dims=true), it means we need to expand
                // the dimensions that were 1 in the current shape.
                // Let's try reshaping directly.
                 crate::ops::view::reshape_op(&expanded_grad, target_shape.to_vec())
            }
        }
    }
}

// --- Expand Kernel (Private) ---

/// Private kernel for expanding/broadcasting tensor data to a target shape.
///
/// Iterates through the target shape, calculating the corresponding index in the source
/// tensor based on broadcasting rules (dimensions of size 1 in the source are repeated).
///
/// # Arguments
/// * `target_shape`: The desired output shape.
/// * `source_data`: Slice containing the data of the source tensor.
/// * `source_shape`: Shape of the source tensor.
/// * `source_strides`: Strides of the source tensor.
/// * `source_offset`: Offset of the source tensor's data within its buffer.
///
/// # Returns
/// * `Ok(Vec<T>)` containing the expanded data.
/// * `Err(NeuraRustError)` if shapes are incompatible or an index is out of bounds.
pub(crate) fn expand_kernel<T>(
    target_shape: &[usize],
    source_data: &[T],
    source_shape: &[usize],
    source_strides: &[usize],
    source_offset: usize,
) -> Result<Vec<T>, NeuraRustError>
where
    T: Copy + Zero + Debug,
{
    let target_rank = target_shape.len();
    let source_rank = source_shape.len();

    // Validation (Should ideally be done before calling the kernel, but double-check)
    if source_rank > target_rank {
         return Err(NeuraRustError::InternalError(format!(
            "Expand kernel error: Source rank ({}) > Target rank ({}).",
            source_rank, target_rank
        )));
    }
    let rank_diff = target_rank - source_rank;
    for i in 0..source_rank {
        if source_shape[i] != 1 && source_shape[i] != target_shape[rank_diff + i] {
            return Err(NeuraRustError::ShapeMismatch {
                expected: target_shape.to_vec(), // Simplified expected shape indication
                actual: source_shape.to_vec(),
                operation: format!(
                    "expand_kernel (dimension {} mismatch: source {} vs target {})",
                     i, source_shape[i], target_shape[rank_diff + i]
                 )
            });
        }
    }

    // Prepare output buffer
    let target_numel = target_shape.iter().product::<usize>();
    let mut output_data = vec![T::zero(); target_numel];
    let mut current_target_indices = vec![0; target_rank];

    // Iterate through the target shape (linearly)
    for target_linear_idx in 0..target_numel {
        // Calculate corresponding source physical index based on broadcasting rules
        let mut source_physical_offset = source_offset;
        for source_dim_idx in 0..source_rank {
            let target_dim_idx = rank_diff + source_dim_idx;
            let source_index_for_dim = if source_shape[source_dim_idx] == 1 && target_shape[target_dim_idx] > 1 {
                0 // Dimension was broadcasted, use index 0 from source
            } else {
                current_target_indices[target_dim_idx] // Dimension matched, use target index
            };
            source_physical_offset += source_index_for_dim * source_strides[source_dim_idx];
        }

        // Bounds check (important!)
        if source_physical_offset >= source_data.len() {
            return Err(NeuraRustError::InternalError(format!(
                "Expand kernel source index out of bounds. TargetCoords: {:?}, SourceOffset: {}, SourceDataLen: {}",
                current_target_indices,
                source_physical_offset,
                source_data.len()
            )));
        }

        // Copy the value (output data is contiguous, use linear index)
        output_data[target_linear_idx] = source_data[source_physical_offset];

        // Increment target indices for the next iteration
        if target_numel > 0 && target_linear_idx < target_numel - 1 {
            let mut dim_to_increment = target_rank;
            while dim_to_increment > 0 {
                dim_to_increment -= 1;
                current_target_indices[dim_to_increment] += 1;
                if current_target_indices[dim_to_increment] < target_shape[dim_to_increment] {
                    break;
                }
                current_target_indices[dim_to_increment] = 0;
            }
        }
    }

    Ok(output_data)
} 