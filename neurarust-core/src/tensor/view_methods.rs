use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::error::NeuraRustError;
use crate::device::StorageDevice; // Needed for contiguous
use std::sync::{Arc, RwLockReadGuard}; // Keep Arc for contiguous helper, RwLockReadGuard for helper signature
use std::fmt::Debug; // Keep Debug for T bound and contiguous helper
use std::marker::Copy; // Keep Copy for T bound and contiguous helper
use std::iter::Product; // Keep Product for T bound in contiguous

// Helper function for recursive multidimensional iteration used by contiguous()
// Moved here from mod.rs
fn copy_non_contiguous_recursive<T: Clone + Debug + Copy + 'static>(
    original_guard: &RwLockReadGuard<'_, TensorData<T>>,
    original_cpu_data: &Arc<Vec<T>>,
    new_buffer: &mut Vec<T>,
    current_indices: &mut Vec<usize>,
    current_dim: usize,
) {
    if current_dim == original_guard.shape.len() {
        let original_offset = original_guard.get_offset(current_indices);
        new_buffer.push(original_cpu_data[original_offset].clone());
    } else {
        for i in 0..original_guard.shape[current_dim] {
            current_indices[current_dim] = i;
            copy_non_contiguous_recursive(
                original_guard,
                original_cpu_data,
                new_buffer,
                current_indices,
                current_dim + 1,
            );
        }
    }
}

impl<T: 'static + Debug + Copy> Tensor<T> {
    /// Checks if the tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.read_data().is_contiguous()
    }

    /// Creates a view of the tensor by slicing along specified dimensions.
    pub fn slice(&self, ranges: &[crate::ops::view_ops::SliceArg]) -> Result<Self, NeuraRustError>
    where
        T: Default + Send + Sync,
    {
        crate::ops::view_ops::slice_op(self, ranges)
    }

    /// Creates a view of the tensor with two dimensions transposed.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, NeuraRustError>
    where
        T: Default + Send + Sync,
    {
        crate::ops::view_ops::transpose_op(self, dim1, dim2)
    }

    /// Creates a view of the tensor with dimensions permuted according to the specified order.
    pub fn permute(&self, dims: &[usize]) -> Result<Self, NeuraRustError>
    where
        T: Default + Send + Sync,
    {
        crate::ops::view_ops::permute_op(self, dims)
    }

    /// Creates a view of the tensor with a different shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuraRustError>
    where
        T: Default + Send + Sync,
    {
        crate::ops::view_ops::reshape_op(self, new_shape)
    }

    /// Returns a contiguous version of the tensor.
    pub fn contiguous(&self) -> Result<Self, NeuraRustError>
    where
        T: Default + Send + Sync + Product,
    {
        if self.is_contiguous() {
            Ok(self.clone()) // Use Self::clone() which is defined in traits.rs
        } else {
            let guard = self.read_data();
            let device = guard.device;
            let shape = guard.shape.clone();
            let numel = guard.numel();

            let mut new_buffer_vec = Vec::with_capacity(numel);

            match device {
                StorageDevice::CPU => {
                    let original_cpu_data = guard.data.cpu_data()?;
                    let mut current_indices = vec![0; shape.len()];
                    copy_non_contiguous_recursive::<T>( // Explicit type annotation
                        &guard,
                        original_cpu_data,
                        &mut new_buffer_vec,
                        &mut current_indices,
                        0,
                    );
                }
                StorageDevice::GPU => {
                    return Err(NeuraRustError::UnsupportedOperation("GPU contiguous copy not yet implemented".to_string()));
                }
            }
            drop(guard);
            // Use Self::new which is defined in create.rs
            Self::new(new_buffer_vec, shape)
        }
    }
} 