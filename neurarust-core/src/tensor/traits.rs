// src/tensor/traits.rs

use crate::device::StorageDevice;
use crate::tensor::Tensor;
 // Import DType
use crate::buffer::{Buffer, CpuBuffer}; // Import Buffer types
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
// use std::marker::Copy; // No longer needed for impls
use std::sync::Arc;
use std::cmp::PartialEq;

// --- Trait Implementations ---

// Remove <T> and bounds
impl Clone for Tensor {
    /// Clones the Tensor. This is a shallow clone that increases the reference count
    /// of the underlying shared data. Modifications through one clone will be visible
    /// through others.
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data), // Clone the Arc
        }
    }
}

// Remove <T> and bounds
impl Debug for Tensor {
    /// Formats the Tensor for debugging. Shows shape, device, strides, offset,
    /// dtype, and a preview of the data if it's on the CPU.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use read lock on self.data
        let td_guard = self.data.read().map_err(|_| fmt::Error)?; // Handle poisoned lock
        let td = &*td_guard;

        write!(
            f,
            "Tensor(shape={:?}, device={:?}, dtype={:?}, strides={:?}, offset={}, data=",
            td.shape, td.device, td.dtype, td.strides, td.offset // Access fields from guard
        )?;

        // Match on buffer type and device
        match (&*td.buffer, td.device) {
            (Buffer::Cpu(cpu_buffer), StorageDevice::CPU) => {
                // Match on specific CPU buffer type (only F32 for now)
                match cpu_buffer {
                    CpuBuffer::F32(data_arc) => {
                        // Get slice for preview (limit number of elements shown)
                        let data_slice: &Vec<f32> = data_arc;
                        let numel = td.numel();
                        let preview_len = std::cmp::min(numel, 10); // Show max 10 elements
                        write!(f, "[... ~{} elements (F32): ", numel)?;
                        // TODO: This preview ignores strides/offset for simplicity
                        for i in 0..preview_len {
                            write!(f, "{:?}{}", data_slice.get(i).unwrap_or(&f32::NAN), if i < preview_len - 1 { ", " } else { "" })?;
                        }
                        if numel > preview_len { write!(f, ", ...")?; }
                        write!(f, "]")?;
                    }
                    // Add arms for other CpuBuffer types later
                    // _ => write!(f, "<Other CPU Buffer Type>")?,
                }
            }
            (Buffer::Gpu{..}, StorageDevice::GPU) => write!(f, "<GPU Buffer>")?,
            // Handle inconsistent state (e.g., CPU buffer on GPU device marker?)
            _ => write!(f, "<Inconsistent Buffer/Device State>")?,
        }

        // Add requires_grad info
        write!(f, ", requires_grad={})", td.requires_grad)?;

        Ok(())
    }
}

// Remove <T> and bounds
impl PartialEq for Tensor {
    /// Checks for logical equality between two Tensors.
    ///
    /// Two tensors are considered equal if they have the same device, dtype, shape,
    /// strides, offset, and underlying data content according to their view parameters.
    /// Gradient information is NOT considered.
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.data, &other.data) {
            return true; // Same underlying TensorData instance
        }

        let self_guard = match self.data.read() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        let other_guard = match other.data.read() {
            Ok(guard) => guard,
            Err(_) => return false,
        };

        // Compare metadata first (including dtype)
        if self_guard.device != other_guard.device
            || self_guard.dtype != other_guard.dtype // Add dtype check
            || self_guard.shape != other_guard.shape
            || self_guard.strides != other_guard.strides
            || self_guard.offset != other_guard.offset
        {
            return false;
        }

        // Compare buffer contents based on device and dtype
        match ((&*self_guard.buffer, self_guard.device), (&*other_guard.buffer, other_guard.device)) {
            ((Buffer::Cpu(CpuBuffer::F32(self_arc)), StorageDevice::CPU),
             (Buffer::Cpu(CpuBuffer::F32(other_arc)), StorageDevice::CPU)) => {
                // Both are F32 CPU buffers
                if Arc::ptr_eq(self_arc, other_arc) {
                    return true; // Same underlying data Arc
                }
                let self_data_slice: &Vec<f32> = self_arc;
                let other_data_slice: &Vec<f32> = other_arc;

                let numel = self_guard.numel();
                if numel == 0 {
                    return true; // Empty tensors are equal
                }

                // Element-wise comparison respecting strides/offset
                let shape = &self_guard.shape;
                let self_strides = &self_guard.strides;
                // No need for other_strides if shapes/strides already matched

                for i in 0..numel {
                    // Use the utility function from the tensor module directly
                    let coords = crate::tensor::utils::index_to_coord(i, self_strides, shape);
                    let self_abs_offset = self_guard.get_offset(&coords);
                    let other_abs_offset = other_guard.get_offset(&coords);

                    // Bounds check before accessing slices
                    if self_abs_offset >= self_data_slice.len() || other_abs_offset >= other_data_slice.len() {
                         eprintln!("Warning: Offset out of bounds during PartialEq check. This might indicate incorrect strides or shape.");
                        return false;
                    }

                    // Compare f32 values (consider using approx::relative_eq! for tolerance later?)
                    if self_data_slice[self_abs_offset] != other_data_slice[other_abs_offset] {
                        return false;
                    }
                }
                true // All elements matched
            }
            // TODO: Add arms for other matching DTypes (I64 == I64, etc.) later
            // ((Buffer::Cpu(CpuBuffer::I64(..)), StorageDevice::CPU), (Buffer::Cpu(CpuBuffer::I64(..)), StorageDevice::CPU)) => { ... }

            // Add arms for GPU comparison later if possible/meaningful
            // ((Buffer::Gpu{..}, StorageDevice::GPU), (Buffer::Gpu{..}, StorageDevice::GPU)) => { ... }

            // All other combinations (different devices, different dtypes, CPU vs GPU) are considered unequal
            _ => false,
        }
    }
}

// Remove <T> and bounds
impl Eq for Tensor {} // Eq follows from PartialEq

// Remove <T> and bounds
impl Hash for Tensor {
    /// Hashes the Tensor based on the pointer address of the `RwLock<TensorData>`.
    /// This allows using Tensors as keys in HashMaps/HashSets where object identity matters.
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the address of the Arc's contained RwLock<TensorData>
        Arc::as_ptr(&self.data).hash(state);
    }
}

// Note: Implementations for Add, Sub, Mul, Neg etc. using the op functions
// should also go in this file or a dedicated ops_impl.rs file.
