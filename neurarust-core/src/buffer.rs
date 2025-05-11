use std::fmt::Debug;
use std::sync::Arc;

use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::types::DType;

/// Abstract representation of a tensor's underlying data storage.
///
/// This enum acts as a dispatcher, holding different buffer types based on the
/// device (`StorageDevice`) and potentially the data type (`DType`).
/// It allows `TensorData` to manage data residing on CPU or GPU (future) uniformly.
#[derive(Debug, Clone)] // Clone is needed if TensorData needs to be cloned
pub enum Buffer {
    /// Data resides on the CPU, managed by a `CpuBuffer`.
    Cpu(CpuBuffer),
    /// Placeholder for data residing on a GPU.
    ///
    /// Currently, this variant only stores metadata (`device`, `len`).
    /// Actual GPU memory management (e.g., holding buffer handles from libraries
    /// like `wgpu` or CUDA bindings) will be implemented in future phases.
    Gpu { 
        /// The specific GPU device.
        device: StorageDevice, 
        /// The number of elements in the buffer.
        len: usize 
    }, 
}

/// Enum representing concrete CPU buffer types, specialized by data type.
///
/// Each variant holds an `Arc<Vec<T>>` for cheap cloning (sharing ownership)
/// of the underlying data vector.
#[derive(Debug, Clone)]
pub enum CpuBuffer {
    /// Buffer holding `f32` (32-bit floating-point) values.
    F32(Arc<Vec<f32>>),
    /// Buffer holding `f64` (64-bit floating-point) values.
    F64(Arc<Vec<f64>>),
    /// Buffer holding `i32` (32-bit integer) values.
    I32(Arc<Vec<i32>>),
    /// Buffer holding `i64` (64-bit integer) values.
    I64(Arc<Vec<i64>>),
    /// Buffer holding `bool` values.
    Bool(Arc<Vec<bool>>),
    // TODO: Add other CPU buffer types like I64, I32, U8, Bool later
}

impl Buffer {
    /// Attempts to get an immutable reference to the underlying `Arc<Vec<f32>>`.
    ///
    /// Returns `Ok(&Arc<Vec<f32>>)` if the buffer is a `Buffer::Cpu(CpuBuffer::F32)`.
    /// Returns `Err(NeuraRustError)` if the buffer is on the GPU or has a different data type.
    pub fn try_get_cpu_f32(&self) -> Result<&Arc<Vec<f32>>, NeuraRustError> {
        match self {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => Ok(data_arc),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::F64,
                operation: "try_get_cpu_f32".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::I32,
                operation: "try_get_cpu_f32".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::I64,
                operation: "try_get_cpu_f32".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::Bool,
                operation: "try_get_cpu_f32".to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: "try_get_cpu_f32".to_string(),
            }),
        }
    }

    /// Attempts to get an immutable reference to the underlying `Arc<Vec<f64>>`.
    ///
    /// Returns `Ok(&Arc<Vec<f64>>)` if the buffer is a `Buffer::Cpu(CpuBuffer::F64)`.
    /// Returns `Err(NeuraRustError)` if the buffer is on the GPU or has a different data type.
    pub fn try_get_cpu_f64(&self) -> Result<&Arc<Vec<f64>>, NeuraRustError> {
        match self {
            Buffer::Cpu(CpuBuffer::F64(data_arc)) => Ok(data_arc),
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::F32,
                operation: "try_get_cpu_f64".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::I32,
                operation: "try_get_cpu_f64".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::I64,
                operation: "try_get_cpu_f64".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::Bool,
                operation: "try_get_cpu_f64".to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: "try_get_cpu_f64".to_string(),
            }),
        }
    }
    
    /// Attempts to get a mutable reference to the underlying `Vec<f32>`.
    ///
    /// This operation requires exclusive access to the buffer. It will fail if the buffer
    /// is shared (i.e., referenced by multiple tensors/views).
    ///
    /// Returns `Ok(&mut Vec<f32>)` if the buffer is `Cpu(F32)` and not shared.
    /// Returns `Err(NeuraRustError)` otherwise (shared buffer, wrong device, wrong dtype).
    pub fn try_get_cpu_f32_mut(&mut self) -> Result<&mut Vec<f32>, NeuraRustError> {
        let op_name = "try_get_cpu_f32_mut";
        match self {
            Buffer::Cpu(CpuBuffer::F32(data_arc)) => {
                match Arc::get_mut(data_arc) {
                    Some(vec_mut) => Ok(vec_mut),
                    None => Err(NeuraRustError::BufferSharedError { operation: op_name.to_string() })
                }
            }
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::F64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::I32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::I64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F32,
                actual: DType::Bool,
                operation: op_name.to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: op_name.to_string(),
            }),
        }
    }

    /// Attempts to get a mutable reference to the underlying `Vec<f64>`.
    ///
    /// This operation requires exclusive access to the buffer. It will fail if the buffer
    /// is shared (i.e., referenced by multiple tensors/views).
    ///
    /// Returns `Ok(&mut Vec<f64>)` if the buffer is `Cpu(F64)` and not shared.
    /// Returns `Err(NeuraRustError)` otherwise (shared buffer, wrong device, wrong dtype).
    pub fn try_get_cpu_f64_mut(&mut self) -> Result<&mut Vec<f64>, NeuraRustError> {
        let op_name = "try_get_cpu_f64_mut";
        match self {
            Buffer::Cpu(CpuBuffer::F64(data_arc)) => {
                 match Arc::get_mut(data_arc) {
                    Some(vec_mut) => Ok(vec_mut),
                    None => Err(NeuraRustError::BufferSharedError { operation: op_name.to_string() })
                 }
            }
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::F32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::I32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::I64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: DType::Bool,
                operation: op_name.to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: op_name.to_string(),
            }),
        }
    }

    /// Tente d'obtenir une référence immuable vers l'Arc<Vec<i32>> sous-jacent.
    pub fn try_get_cpu_i32(&self) -> Result<&Arc<Vec<i32>>, NeuraRustError> {
        match self {
            Buffer::Cpu(CpuBuffer::I32(data_arc)) => Ok(data_arc),
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::F32,
                operation: "try_get_cpu_i32".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::F64,
                operation: "try_get_cpu_i32".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::I64,
                operation: "try_get_cpu_i32".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::Bool,
                operation: "try_get_cpu_i32".to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: "try_get_cpu_i32".to_string(),
            }),
        }
    }

    /// Tente d'obtenir une référence immuable vers l'Arc<Vec<i64>> sous-jacent.
    pub fn try_get_cpu_i64(&self) -> Result<&Arc<Vec<i64>>, NeuraRustError> {
        match self {
            Buffer::Cpu(CpuBuffer::I64(data_arc)) => Ok(data_arc),
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::F32,
                operation: "try_get_cpu_i64".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::F64,
                operation: "try_get_cpu_i64".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::I32,
                operation: "try_get_cpu_i64".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::Bool,
                operation: "try_get_cpu_i64".to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: "try_get_cpu_i64".to_string(),
            }),
        }
    }

    /// Tente d'obtenir une référence immuable vers l'Arc<Vec<bool>> sous-jacent.
    pub fn try_get_cpu_bool(&self) -> Result<&Arc<Vec<bool>>, NeuraRustError> {
        match self {
            Buffer::Cpu(CpuBuffer::Bool(data_arc)) => Ok(data_arc),
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::F32,
                operation: "try_get_cpu_bool".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::F64,
                operation: "try_get_cpu_bool".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::I32,
                operation: "try_get_cpu_bool".to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::I64,
                operation: "try_get_cpu_bool".to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: "try_get_cpu_bool".to_string(),
            }),
        }
    }

    pub fn try_get_cpu_i32_mut(&mut self) -> Result<&mut Vec<i32>, NeuraRustError> {
        let op_name = "try_get_cpu_i32_mut";
        match self {
            Buffer::Cpu(CpuBuffer::I32(data_arc)) => {
                match Arc::get_mut(data_arc) {
                    Some(vec_mut) => Ok(vec_mut),
                    None => Err(NeuraRustError::BufferSharedError { operation: op_name.to_string() })
                }
            }
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::F32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::F64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::I64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: DType::Bool,
                operation: op_name.to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: op_name.to_string(),
            }),
        }
    }
    pub fn try_get_cpu_i64_mut(&mut self) -> Result<&mut Vec<i64>, NeuraRustError> {
        let op_name = "try_get_cpu_i64_mut";
        match self {
            Buffer::Cpu(CpuBuffer::I64(data_arc)) => {
                match Arc::get_mut(data_arc) {
                    Some(vec_mut) => Ok(vec_mut),
                    None => Err(NeuraRustError::BufferSharedError { operation: op_name.to_string() })
                }
            }
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::F32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::F64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::I32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::Bool(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: DType::Bool,
                operation: op_name.to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: op_name.to_string(),
            }),
        }
    }
    pub fn try_get_cpu_bool_mut(&mut self) -> Result<&mut Vec<bool>, NeuraRustError> {
        let op_name = "try_get_cpu_bool_mut";
        match self {
            Buffer::Cpu(CpuBuffer::Bool(data_arc)) => {
                match Arc::get_mut(data_arc) {
                    Some(vec_mut) => Ok(vec_mut),
                    None => Err(NeuraRustError::BufferSharedError { operation: op_name.to_string() })
                }
            }
            Buffer::Cpu(CpuBuffer::F32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::F32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::F64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::F64,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I32(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::I32,
                operation: op_name.to_string(),
            }),
            Buffer::Cpu(CpuBuffer::I64(_)) => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: DType::I64,
                operation: op_name.to_string(),
            }),
            Buffer::Gpu { device, .. } => Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: *device,
                operation: op_name.to_string(),
            }),
        }
    }

    // TODO: Add similar try_get methods for other DTypes (I64, Bool, etc.) when added.
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
