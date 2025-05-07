use crate::{error::NeuraRustError, ops::reduction::sum_op, ops::view::reshape_op, tensor::Tensor};
use std::fmt::Debug;
use num_traits::Zero;

impl Tensor {
    /// Reduces the tensor (gradient) to match a target shape by summing along broadcasted dimensions.
    ///
    /// Crucial for backward pass of broadcasting ops. Assumes input is F32 CPU for now.
    pub fn reduce_to_shape(
        &self,
        target_shape: &[usize],
    ) -> Result<Tensor, NeuraRustError> {
        let current_shape = self.shape(); // Use non-generic accessor

        if current_shape == target_shape {
            return Ok(self.clone());
        }

        // Handle reduction to scalar
        if target_shape.is_empty() {
            if self.numel() == 1 { // Use non-generic accessor
                 return reshape_op(self, vec![]); // Use non-generic op
            }
            let axes_to_reduce: Vec<usize> = (0..current_shape.len()).collect();
            // Use sum_op, pass axes as Option<&[usize]>
            return sum_op(self, Some(&axes_to_reduce), false);
        }

        let current_rank = current_shape.len();
        let target_rank = target_shape.len();

        if current_rank < target_rank {
            return Err(NeuraRustError::InternalError(format!(
                "Cannot reduce shape {:?} to target {:?}: Current rank < target rank.",
                current_shape, target_shape
            )));
        }

        let rank_diff = current_rank - target_rank;
        let mut axes_to_reduce: Vec<usize> = (0..rank_diff).collect();

        for i in 0..target_rank {
            let current_dim = current_shape[rank_diff + i];
            let target_dim = target_shape[i];

            if current_dim != target_dim {
                // target_dim must be 1 for broadcast reduction
                if target_dim == 1 {
                    axes_to_reduce.push(rank_diff + i);
                } else {
                    return Err(NeuraRustError::InternalError(format!(
                        "Cannot reduce shape {:?} to target {:?}: Incompatible dim {} ({} vs target {}).",
                        current_shape, target_shape, i, current_dim, target_dim
                    )));
                }
            }
        }

        if axes_to_reduce.is_empty() {
            if current_shape == target_shape {
                Ok(self.clone())
            } else {
                 Err(NeuraRustError::InternalError(format!(
                     "Cannot reduce shape {:?} to {:?}: Shapes differ but no reduction axes.",
                     current_shape, target_shape
                 )))
            }
        } else {
            // Use sum_op, pass axes as Option<&[usize]>, keep_dims=true
            let reduced_grad = sum_op(self, Some(&axes_to_reduce), true)?;

            // Reshape to remove squeezed dims if necessary
            if reduced_grad.shape() == target_shape {
                 Ok(reduced_grad)
            } else {
                // TODO: Check contiguity before reshape? sum_op needs to guarantee output contiguity?
                reshape_op(&reduced_grad, target_shape.to_vec())
            }
        }
    }

    /// Expands a gradient tensor to match the shape of the original tensor it corresponds to.
    /// Placeholder - Use expand_op directly for now.
    /// TODO: Implement proper gradient expansion logic if needed, potentially using expand_kernel.
    pub fn expand_gradient_to_shape(
        &self, // The gradient tensor
        target_shape: &[usize],
    ) -> Result<Tensor, NeuraRustError> {
        let current_shape = self.shape();
        if current_shape == target_shape {
            return Ok(self.clone());
        }
        let target_shape_isize: Vec<isize> = target_shape.iter().map(|&d| d as isize).collect();
        crate::ops::view::expand_op(self, &target_shape_isize)
    }

    /// Expands the tensor to match a target shape by adding new dimensions (size 1) or repeating existing dimensions of size 1.
    /// This is the counterpart to `reduce_to_shape` for broadcasting.
    ///
    /// Used in backward passes where a gradient needs to be expanded to the shape of an original input.
    pub fn expand_to_match_nd(
        &self,
        target_shape: &[usize],
    ) -> Result<Tensor, NeuraRustError> {
        let current_shape = self.shape(); // Use non-generic accessor
        if current_shape == target_shape {
            return Ok(self.clone());
        }

        // Example: current_shape [3], target_shape [2, 3]
        // We need to call expand_op with a target_shape that reflects the desired view.
        // expand_op itself handles the logic of adding new axes (size 1) or expanding existing size 1 axes.

        // Convert target_shape from &[usize] to &[isize] for expand_op
        let target_shape_isize: Vec<isize> = target_shape.iter().map(|&dim| dim as isize).collect();

        // Call the actual expand_op. This op should handle adding new leading dimensions if necessary.
        crate::ops::view::expand_op(self, &target_shape_isize)
    }
}

// expand_kernel can remain generic for now
pub(crate) fn expand_kernel<T>(
    target_shape: &[usize],
    source_data: &[T],
    source_shape: &[usize],
    source_strides: &[usize],
    source_offset: usize,
) -> Result<Vec<T>, NeuraRustError>
where
    T: Copy + Zero + Debug, // Keep necessary bounds for the kernel logic
{
    let target_numel = target_shape.iter().product::<usize>();
    let mut expanded_data = vec![T::zero(); target_numel];

    let source_rank = source_shape.len();
    let target_rank = target_shape.len();
    let rank_diff = target_rank.saturating_sub(source_rank);

    let mut current_target_indices = vec![0; target_rank];

    for i in 0..target_numel {
        // Calculate corresponding source indices based on target indices
        let mut current_source_indices = vec![0; source_rank];
        let mut use_source = true;
        for j in 0..target_rank {
            let source_dim_index = j.checked_sub(rank_diff);

            if let Some(src_idx) = source_dim_index {
                // Dimension exists in source
                let source_dim_size = source_shape[src_idx];
                if source_dim_size == 1 {
                    // Broadcasting from dim size 1, source index is always 0
                    current_source_indices[src_idx] = 0;
                } else if current_target_indices[j] < source_dim_size {
                    // Target index is within source bounds
                    current_source_indices[src_idx] = current_target_indices[j];
                } else {
                    // This case theoretically shouldn't be reachable if shapes are broadcast-compatible
                    // and we are expanding correctly. But as a safeguard:
                    use_source = false; // Index out of bounds for non-broadcast dim
                    break;
                }
            } else {
                // Dimension only exists in target (was added by broadcasting)
                // No corresponding source index, but doesn't invalidate access
                // if the source dimensions align correctly.
            }
        }

        if use_source {
            // Calculate the linear offset in the source buffer
            let mut source_linear_offset = source_offset;
            for k in 0..source_rank {
                source_linear_offset += current_source_indices[k] * source_strides[k];
            }

            // Check bounds for source_data slice
            if source_linear_offset < source_data.len() {
                expanded_data[i] = source_data[source_linear_offset];
            } else {
                // Error: Calculated offset is out of bounds for the provided source data slice
                return Err(NeuraRustError::InternalError(format!(
                    "Source buffer access out of bounds during expand_kernel. Offset: {}, Len: {}. Src Shape: {:?}, Tgt Shape: {:?}, Tgt Idx: {:?}, Src Idx: {:?}",
                    source_linear_offset, source_data.len(), source_shape, target_shape, current_target_indices, current_source_indices
                )));
            }
        } // else: use_source is false, keep the zero value

        // Increment target indices (standard C-order iteration)
        if target_numel > 0 {
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

    Ok(expanded_data)
} 