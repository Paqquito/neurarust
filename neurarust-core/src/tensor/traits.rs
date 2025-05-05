// src/tensor/traits.rs

//! This module implements standard Rust traits for the main [`Tensor`](../struct.Tensor.html) structure.
//!
//! It provides implementations for:
//! - [`Clone`]: Shallow cloning using `Arc` for shared ownership.
//! - [`Debug`]: Debug formatting, displaying tensor metadata and a preview of CPU data.
//! - [`PartialEq`] and [`Eq`]: Logical equality check based on shape, strides, dtype, device, offset, and data content (ignoring gradients).
//! - [`Hash`]: Hashing based on the memory address of the underlying data for identity comparison.
//! - `TensorImpl`: A trait (and its implementation for `Tensor`) providing basic accessor methods.

use crate::device::StorageDevice;
use crate::tensor::Tensor;
use crate::types::DType; // Import DType
use crate::buffer::{Buffer, CpuBuffer}; // Import Buffer types
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
// use std::marker::Copy; // No longer needed for impls
use std::sync::Arc;
use std::cmp::PartialEq;

/// A trait defining the core properties and accessors for a tensor implementation.
///
/// This trait is intended to abstract the basic characteristics of a tensor,
/// such as its shape, data type, and storage details. While `Tensor` itself
/// implements this, it could potentially be used for other tensor-like structures
/// in the future.
#[allow(dead_code)] // Keep trait definition for potential future use or reference
pub trait TensorImpl {
    /// Returns the shape of the tensor as a `Vec<usize>`.
    fn shape(&self) -> Vec<usize>;
    /// Returns the strides of the tensor as a `Vec<usize>`.
    fn strides(&self) -> Vec<usize>;
    /// Returns the storage device where the tensor data resides.
    fn device(&self) -> StorageDevice;
    /// Returns the data type (`DType`) of the tensor elements.
    fn dtype(&self) -> DType;
    /// Returns the rank (number of dimensions) of the tensor.
    fn rank(&self) -> usize;
    /// Returns the total number of elements in the tensor.
    fn numel(&self) -> usize;
    /// Checks if the tensor is contiguous in memory.
    fn is_contiguous(&self) -> bool;
    // Add other methods that were part of the trait if necessary
}

// --- Trait Implementations ---

// Note: No <T> generic parameter needed anymore for Tensor
impl Clone for Tensor {
    /// Creates a shallow clone of the `Tensor`.
    ///
    /// This operation is inexpensive as it only increments the reference count
    /// of the underlying shared [`TensorData`](../tensor_data/struct.TensorData.html).
    /// Both the original and the cloned `Tensor` will point to the same data.
    /// Modifications made through one clone will be visible through the other.
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data), // Clone the Arc
        }
    }
}

// Note: No <T> generic parameter needed anymore for Tensor
impl Debug for Tensor {
    /// Formats the `Tensor` for debugging purposes.
    ///
    /// The output includes:
    /// - Shape
    /// - Storage device
    /// - Data type (`DType`)
    /// - Strides
    /// - Offset
    /// - A preview of the data elements (limited count) if the tensor is on the CPU.
    /// - Whether the tensor requires gradient computation (`requires_grad`).
    ///
    /// For GPU tensors, it currently indicates `<GPU Buffer>` instead of showing data.
    /// The data preview for CPU tensors currently ignores strides and offset for simplicity.
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
                    CpuBuffer::F64(data_arc) => {
                        // Get slice for preview (limit number of elements shown)
                        let data_slice: &Vec<f64> = data_arc;
                        let numel = td.numel();
                        let preview_len = std::cmp::min(numel, 10); // Show max 10 elements
                        write!(f, "[... ~{} elements (F64): ", numel)?;
                        // TODO: This preview ignores strides/offset for simplicity
                        for i in 0..preview_len {
                            write!(f, "{:?}{}", data_slice.get(i).unwrap_or(&f64::NAN), if i < preview_len - 1 { ", " } else { "" })?;
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

// Note: No <T> generic parameter needed anymore for Tensor
impl PartialEq for Tensor {
    /// Checks for logical equality between two `Tensor` instances.
    ///
    /// Two tensors are considered equal if they meet all the following criteria:
    /// 1. They reside on the same [`StorageDevice`].
    /// 2. They have the same [`DType`].
    /// 3. They have the same shape.
    /// 4. They have the same strides.
    /// 5. They have the same offset.
    /// 6. Their underlying data elements are equal, considering their respective views
    ///    (shape, strides, offset).
    ///
    /// **Important:**
    /// - This comparison **ignores** gradient information (`grad` and `grad_fn`).
    /// - Currently, element-wise comparison is only implemented for `f32` tensors residing on the CPU.
    ///   Comparisons involving other data types or GPU tensors will likely return `false` unless
    ///   they are pointer-equal (the exact same `Arc<RwLock<TensorData>>`).
    /// - Floating-point comparisons use direct `!=`. For tolerance-based comparisons, use dedicated functions.
    ///
    /// This performs a deep comparison of the data if the metadata matches and the tensors
    /// don't share the exact same underlying `TensorData` instance.
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
                let _self_strides = &self_guard.strides;
                // No need for other_strides if shapes/strides already matched

                for i in 0..numel {
                    // Corrected: Use the utility function with only index and shape
                    let coords = crate::tensor::utils::index_to_coord(i, shape); // Pass only shape
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

// Note: No <T> generic parameter needed anymore for Tensor
/// Marker trait implementation indicating that `Tensor` has a total equality relation
/// when `PartialEq` returns true.
impl Eq for Tensor {}

// Note: No <T> generic parameter needed anymore for Tensor
impl Hash for Tensor {
    /// Hashes the `Tensor` based on the memory address of its underlying shared data structure.
    ///
    /// This implementation ensures that two `Tensor` clones (pointing to the same
    /// `Arc<RwLock<TensorData>>`) will produce the same hash value.
    ///
    /// **Note:** This hash function reflects *object identity*, not *value equality*.
    /// Two logically equal but distinct tensors (e.g., created separately with the same data)
    /// will likely have different hash values.
    /// Use this for scenarios where you need to track specific tensor instances in hash-based
    /// collections (like `HashMap` or `HashSet`).
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the address of the Arc's contained RwLock<TensorData>
        Arc::as_ptr(&self.data).hash(state);
    }
}

// Note: Implementations for Add, Sub, Mul, Neg etc. using the op functions
// should also go in this file or a dedicated ops_impl.rs file.

// --- TensorImpl Implementation ---

/// Implementation of the `TensorImpl` trait for the main `Tensor` struct.
/// Provides access to the core properties stored within the underlying `TensorData`.
impl TensorImpl for Tensor {
    /// Returns a clone of the tensor's shape.
    /// Acquires a read lock on the internal data.
    fn shape(&self) -> Vec<usize> {
        self.data.read().unwrap().shape.clone()
    }

    /// Returns a clone of the tensor's strides.
    /// Acquires a read lock on the internal data.
    fn strides(&self) -> Vec<usize> {
        self.data.read().unwrap().strides.clone()
    }

    /// Returns the tensor's storage device.
    /// Acquires a read lock on the internal data.
    fn device(&self) -> StorageDevice {
        self.data.read().unwrap().device
    }

    /// Returns the tensor's data type (`DType`).
    /// Acquires a read lock on the internal data.
    fn dtype(&self) -> DType {
        self.data.read().unwrap().dtype
    }

    /// Returns the rank (number of dimensions) of the tensor.
    /// Acquires a read lock on the internal data.
    fn rank(&self) -> usize {
        self.data.read().unwrap().shape.len()
    }

    /// Returns the total number of elements in the tensor.
    /// Acquires a read lock on the internal data.
    fn numel(&self) -> usize {
        self.data.read().unwrap().shape.iter().product()
    }

    /// Checks if the tensor's data layout is contiguous in memory.
    /// Acquires a read lock on the internal data.
    fn is_contiguous(&self) -> bool {
        let self_guard = self.data.read().unwrap();
        let self_shape = &self_guard.shape;
        let rank = self_shape.len();

        if rank == 0 {
            return true;
        }

        // Use actual strides for calculation
        let self_strides = &self_guard.strides;
        // Recalculate expected strides
        let mut expected_stride = 1;
        for i in (0..rank).rev() {
            let current_shape = self_shape[i];
            if current_shape == 0 {
                // Tensor with dimension 0 is considered contiguous
                continue;
            }
            // Skip dimension with size 1 for contiguity check, but its stride must be consistent if not the last dim
            if current_shape == 1 {
                // We don't update expected_stride here for size 1 dims
                continue;
            }

            if self_strides[i] != expected_stride {
                return false;
            }
            expected_stride *= current_shape;
        }
        true
    }
}
