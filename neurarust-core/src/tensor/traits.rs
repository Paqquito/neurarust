// src/tensor/traits.rs

use crate::tensor::Tensor;
use crate::device::StorageDevice;
use std::fmt::{self, Debug};
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::marker::Copy;

// --- Trait Implementations ---

// Add bounds here
impl<T: 'static + Debug + Copy> Clone for Tensor<T> {
    /// Clones the Tensor. This is a shallow clone that increases the reference count
    /// of the underlying shared data. Modifications through one clone will be visible
    /// through others.
    fn clone(&self) -> Self {
        Tensor { data: Arc::clone(&self.data) }
    }
}

// Add bounds here, merge with existing Debug
impl<T: 'static + Debug + Copy> Debug for Tensor<T> {
    /// Formats the Tensor for debugging. Shows shape, device, strides, offset,
    /// and a preview of the data if it's on the CPU.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Need read_data access
        let td = self.read_data();

        write!(f, "Tensor(shape={:?}, device={:?}, strides={:?}, offset={}, data=",
               td.shape, td.device, td.strides, td.offset)?;

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
impl<T: PartialEq + 'static + Debug + Copy> PartialEq for Tensor<T> {
    /// Checks for tensor equality.
    fn eq(&self, other: &Self) -> bool {
        // Need read_data access
        if Arc::ptr_eq(&self.data, &other.data) {
            return true;
        }
        let self_guard = self.read_data();
        let other_guard = other.read_data();

        if self_guard.shape != other_guard.shape ||
           self_guard.device != other_guard.device ||
           self_guard.offset != other_guard.offset ||
           self_guard.strides != other_guard.strides
        {
            return false;
        }

        match (self_guard.device, other_guard.device) {
            (StorageDevice::CPU, StorageDevice::CPU) => {
                match (self_guard.data.cpu_data(), other_guard.data.cpu_data()) {
                    (Ok(self_cpu_data), Ok(other_cpu_data)) => {
                        if Arc::ptr_eq(&self_guard.data, &other_guard.data) {
                            return true;
                        }
                        if self_guard.is_contiguous() && other_guard.is_contiguous() && self_guard.offset == other_guard.offset {
                            let numel = self_guard.numel();
                            let self_slice = &self_cpu_data[self_guard.offset..self_guard.offset + numel];
                            let other_slice = &other_cpu_data[other_guard.offset..other_guard.offset + numel];
                            self_slice == other_slice
                        } else {
                            eprintln!(
                                "Warning: PartialEq comparing non-contiguous CPU Tensors or views with different offsets. Returning false. Implement element-wise comparison."
                            );
                            false
                        }
                    }
                    _ => false,
                }
            }
            (StorageDevice::GPU, StorageDevice::GPU) => {
                 Arc::ptr_eq(&self_guard.data, &other_guard.data)
            }
            _ => false,
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