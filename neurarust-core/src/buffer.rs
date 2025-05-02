use std::fmt::Debug;
use std::sync::Arc;

use crate::device::StorageDevice;
use crate::error::NeuraRustError;

/// Represents the actual storage for tensor data.
/// Can hold data either on CPU or GPU (or other devices in the future).
#[derive(Debug)] // Avoid deriving Clone, Eq, PartialEq directly as GPU buffers might not support it easily.
pub enum Buffer<T> {
    /// Data stored on the CPU as a standard Rust vector.
    /// Wrapped in Arc for cheap cloning (needed for shared ownership).
    Cpu(Arc<Vec<T>>),
    /// Placeholder for GPU buffer. Stores device and maybe size.
    /// The actual GPU buffer handle (e.g., wgpu::Buffer) would be stored elsewhere
    /// or managed by a dedicated GPU memory manager.
    Gpu { device: StorageDevice, len: usize }, // Store len for consistency
}

impl<T> Buffer<T> {
    /// Returns the number of elements the buffer can hold.
    pub fn len(&self) -> usize {
        match self {
            Buffer::Cpu(vec_arc) => vec_arc.len(),
            Buffer::Gpu { len, .. } => *len,
        }
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the device where the buffer is stored.
    pub fn device(&self) -> StorageDevice {
        match self {
            Buffer::Cpu(_) => StorageDevice::CPU,
            Buffer::Gpu { device, .. } => *device,
        }
    }

    // --- CPU Specific Methods ---

    /// Creates a new CPU buffer from a vector.
    pub fn new_cpu(data: Vec<T>) -> Self {
        Buffer::Cpu(Arc::new(data))
    }

    /// Provides access to the underlying CPU data vector if the buffer is on the CPU.
    /// Returns an error if the buffer is not on the CPU.
    pub fn cpu_data(&self) -> Result<&Arc<Vec<T>>, NeuraRustError> {
        match self {
            Buffer::Cpu(data_arc) => Ok(data_arc),
            Buffer::Gpu { .. } => Err(NeuraRustError::DataNotAvailableError {
                expected: StorageDevice::CPU,
                actual: self.device(),
            }),
        }
    }

    // --- GPU Specific Methods (Placeholders) ---
    // pub fn gpu_buffer(&self) -> Result<GpuBufferHandle, NeuraRustError> { ... }
}

// Implement Clone manually to ensure Arc is cloned correctly
impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        match self {
            Buffer::Cpu(arc_vec) => Buffer::Cpu(Arc::clone(arc_vec)),
            Buffer::Gpu { device, len } => Buffer::Gpu {
                device: *device,
                len: *len,
            }, // Clone metadata
        }
    }
}

// Equality check might be complex, especially for GPU.
// For now, let's compare based on CPU data pointers or GPU metadata.
impl<T> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Buffer::Cpu(arc_a), Buffer::Cpu(arc_b)) => Arc::ptr_eq(arc_a, arc_b),
            (
                Buffer::Gpu {
                    device: dev_a,
                    len: len_a,
                },
                Buffer::Gpu {
                    device: dev_b,
                    len: len_b,
                },
            ) => dev_a == dev_b && len_a == len_b, // Simple metadata check for now
            _ => false, // Buffers on different devices are not equal
        }
    }
}
impl<T> Eq for Buffer<T> {}
