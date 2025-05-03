use std::fmt::Debug;
use std::sync::Arc;

use crate::device::StorageDevice;
use crate::error::NeuraRustError;

/// Enum representing different buffer types based on device and data type.
/// This allows TensorData to hold different kinds of data buffers.
#[derive(Debug, Clone)] // Clone is needed if TensorData needs to be cloned
pub enum Buffer {
    /// Data resides on the CPU.
    Cpu(CpuBuffer),
    /// Placeholder for GPU buffer. Stores device and maybe size.
    /// The actual GPU buffer handle (e.g., wgpu::Buffer) would be stored elsewhere
    /// or managed by a dedicated GPU memory manager.
    Gpu { device: StorageDevice, len: usize }, // Store len for consistency
}

/// Enum for CPU-specific buffer types.
#[derive(Debug, Clone)]
pub enum CpuBuffer {
    /// Buffer holding f32 data on the CPU.
    F32(Arc<Vec<f32>>),
    // Add other CPU types like I64, F64 etc. here later
    // e.g., I64(Arc<Vec<i64>>),
}

impl Buffer {
    /// Attempts to get a reference to the underlying `Arc<Vec<f32>>` if this is a CPU F32 buffer.
    ///
    /// Returns an error if the buffer is not a CPU buffer or not of type F32.
    pub fn try_get_cpu_f32(&self) -> Result<&Arc<Vec<f32>>, NeuraRustError> {
        match self {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => Ok(data_arc),
            Buffer::Cpu(_) => Err(NeuraRustError::UnsupportedOperation(
                "Buffer is CPU but not F32 type".to_string(),
            )),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: "try_get_cpu_f32".to_string(),
            }),
        }
    }
}

/* // Temporarily comment out old methods causing errors

impl<T: Clone + Debug + Default> Buffer<T> {
    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        match self {
            Buffer::Cpu(data) => data.len(),
            // Buffer::Gpu(info) => info.size, // Assuming GpuBufferInfo has size
            // Add other backends
        }
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a new CPU buffer from a vector.
    pub fn new_cpu(data: Vec<T>) -> Self {
        Buffer::Cpu(Arc::new(data))
    }

    /// Attempts to get a reference to the CPU data Arc<Vec<T>>.
    /// Returns None if the buffer is not on the CPU.
    pub fn cpu_data(&self) -> Option<&Arc<Vec<T>>> {
        match self {
            Buffer::Cpu(data) => Some(data),
            // Add other backends
            // _ => None,
        }
    }

    /// Clones the CPU data into a new Vec<T>.
    /// Panics if the buffer is not on the CPU.
    pub fn to_cpu_vec(&self) -> Vec<T> {
        match self.cpu_data() {
            Some(data_arc) => data_arc.as_ref().clone(),
            None => panic!("Cannot convert non-CPU buffer to Vec"),
        }
    }

    // Add methods like `gpu_data()`, `to_gpu_buffer()`, etc. later
}

*/

/* // Temporarily comment out old trait implementations

// Implement Clone manually to ensure Arc is cloned correctly
impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        match self {
            Buffer::Cpu(data_arc) => Buffer::Cpu(Arc::clone(data_arc)),
            // Buffer::Gpu(info) => Buffer::Gpu(info.clone()), // Assuming GpuBufferInfo is Clone
        }
    }
}

// Implement Eq and PartialEq manually if needed, comparing underlying data
// Note: Comparing GPU buffers might be complex or nonsensical.
impl<T: Eq> Eq for Buffer<T> {}

impl<T: PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Buffer::Cpu(self_data), Buffer::Cpu(other_data)) => self_data == other_data,
            // Add comparisons for other combinations (CPU vs GPU, GPU vs GPU)
            // This might involve transferring data or simply returning false.
            // For now, only CPU-CPU comparison is shown.
            _ => false, // Buffers on different devices or types are not equal
        }
    }
}

*/
