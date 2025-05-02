// src/tensor/traits.rs

use crate::device::StorageDevice;
use crate::tensor::Tensor;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::marker::Copy;
use std::sync::Arc;
use std::cmp::PartialEq;

// --- Trait Implementations ---

// Add bounds here
impl<T: 'static + Debug + Copy> Clone for Tensor<T> {
    /// Clones the Tensor. This is a shallow clone that increases the reference count
    /// of the underlying shared data. Modifications through one clone will be visible
    /// through others.
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
        }
    }
}

// Add bounds here, merge with existing Debug
impl<T: 'static + Debug + Copy> Debug for Tensor<T> {
    /// Formats the Tensor for debugging. Shows shape, device, strides, offset,
    /// and a preview of the data if it's on the CPU.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Need read_data access
        let td = self.read_data();

        write!(
            f,
            "Tensor(shape={:?}, device={:?}, strides={:?}, offset={}, data=",
            td.shape, td.device, td.strides, td.offset
        )?;

        match td.device {
            StorageDevice::CPU => {
                match td.data.cpu_data() {
                    Ok(_cpu_data) => {
                        // Simplified display
                        write!(f, "[... ~{} elements on CPU ...]", td.numel())?;
                    }
                    Err(_) => write!(f, "<Error getting CPU data>")?,
                }
            }
            StorageDevice::GPU => write!(f, "<GPU Buffer>")?,
        }

        write!(f, ")")
    }
}

// Add bounds here, merge with existing PartialEq
impl<T> PartialEq for Tensor<T>
where
    T: 'static + Debug + Copy + PartialEq,
{
    /// Checks for logical equality between two Tensors.
    ///
    /// Two tensors are considered equal if they have the same device, shape, strides,
    /// offset, and underlying data content according to their view parameters.
    ///
    /// Gradient information (`grad`, `requires_grad`, `grad_fn`) is NOT considered
    /// in this equality check.
    ///
    /// Note: This currently only implements comparison for tensors residing on the CPU.
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.data, &other.data) {
            return true;
        }

        // Use read lock directly via the data field
        let self_guard = match self.data.read() {
            Ok(guard) => guard,
            Err(_) => return false, // Poisoned lock, consider unequal
        };
        let other_guard = match other.data.read() {
            Ok(guard) => guard,
            Err(_) => return false,
        };

        // Compare metadata
        if self_guard.device != other_guard.device
            || self_guard.shape != other_guard.shape
            || self_guard.strides != other_guard.strides
            || self_guard.offset != other_guard.offset
        {
            return false;
        }

        // Compare buffer contents for CPU tensors
        if self_guard.device == StorageDevice::CPU {
            let self_buffer_arc = match self_guard.data.cpu_data() {
                Ok(arc) => arc,
                Err(_) => return false, // Failed to get CPU data
            };
            let other_buffer_arc = match other_guard.data.cpu_data() {
                Ok(arc) => arc,
                Err(_) => return false,
            };

            // Optimization: If underlying data Arcs are the same, data is equal
            if Arc::ptr_eq(self_buffer_arc, other_buffer_arc) {
                return true;
            }

            let self_data_slice = self_buffer_arc.as_slice();
            let other_data_slice = other_buffer_arc.as_slice();

            let numel = self_guard.shape.iter().product();
            if numel == 0 {
                return true; // Empty tensors are equal
            }

            // Use nd_iter or direct index calculation for element comparison
            // Need index_to_coord and get_offset from TensorData
            let shape = &self_guard.shape;
            let self_strides = &self_guard.strides;
            // No need for other_strides if shapes/strides already matched

            for i in 0..numel {
                 // Need to qualify index_to_coord if it's in utils
                let coords = super::utils::index_to_coord(i, self_strides, shape);
                let self_offset = self_guard.get_offset(&coords);
                let other_offset = other_guard.get_offset(&coords);

                if self_offset >= self_data_slice.len() || other_offset >= other_data_slice.len() {
                    return false; // Offset out of bounds
                }

                if self_data_slice[self_offset] != other_data_slice[other_offset] {
                    return false;
                }
            }
            // All elements matched
            true
        } else {
            // For non-CPU tensors, rely on metadata comparison for now
            eprintln!(
                "Warning: PartialEq comparison for non-CPU tensors relies on metadata only."
            );
            true // Metadata already matched
        }
    }
}

// Add bounds here, merge with existing Eq
impl<T: Eq + 'static + Debug + Copy> Eq for Tensor<T> {} // Eq follows from PartialEq

// Add bounds here
impl<T: Hash + 'static + Debug + Copy> Hash for Tensor<T> {
    /// Hashes the Tensor based on the pointer address of the `RwLock<TensorData>`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Need id_ptr access
        self.id_ptr().hash(state);
    }
}

// Note: Implementations for Add, Sub, Mul, Neg etc. using the op functions
// should also go in this file or a dedicated ops_impl.rs file.
