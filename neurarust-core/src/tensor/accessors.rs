// src/tensor/accessors.rs
use crate::{
    buffer::{Buffer, CpuBuffer},
    device::StorageDevice,
    error::NeuraRustError,
    tensor::Tensor,
    types::DType,
};

impl Tensor {
    /// Returns a clone of the tensor's shape (dimensions).
    pub fn shape(&self) -> Vec<usize> {
        self.read_data().shape.clone()
    }

    /// Returns a clone of the tensor's strides.
    pub fn strides(&self) -> Vec<usize> {
        self.read_data().strides.clone()
    }

    /// Returns the storage device where the tensor data resides.
    pub fn device(&self) -> StorageDevice {
        self.read_data().device
    }

    /// Returns the data type (`DType`) of the tensor elements.
    pub fn dtype(&self) -> DType {
        self.read_data().dtype
    }

    /// Returns the rank (number of dimensions) of the tensor.
    pub fn rank(&self) -> usize {
        self.read_data().shape.len()
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        // Reuse shape() ? No, direct access is fine.
        self.read_data().shape.iter().product()
    }

     /// Checks if the tensor is contiguous in memory.
     /// A tensor is contiguous if its elements are laid out in memory sequentially
     /// according to the standard row-major order (C order).
     pub fn is_contiguous(&self) -> bool {
         self.read_data().is_contiguous() // Delegate to TensorData
     }


    /// Extracts the single scalar value as f32 from a tensor containing exactly one element.
    ///
    /// Returns an error if the tensor is not scalar, not F32, or if offset is invalid.
    pub fn item_f32(&self) -> Result<f32, NeuraRustError> {
        let guard = self.read_data();
        if guard.numel() != 1 {
            return Err(NeuraRustError::ShapeMismatch {
                operation: "item_f32()".to_string(),
                expected: "1 element".to_string(),
                actual: format!("{} elements (shape {:?})", guard.numel(), guard.shape),
            });
        }
        let physical_offset = guard.offset;

        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::F32(arc_vec)) => {
                let data: &Vec<f32> = arc_vec;
                if physical_offset >= data.len() {
                     return Err(NeuraRustError::IndexOutOfBounds { index: vec![physical_offset], shape: vec![data.len()] });
                }
                Ok(data[physical_offset]) // Direct access
            }
            Buffer::Cpu(CpuBuffer::F64(_)) => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: DType::F64,
                    operation: "item_f32() called on F64 tensor".to_string(),
                })
            }
            Buffer::Gpu { .. } => {
                 Err(NeuraRustError::DeviceMismatch { expected: StorageDevice::CPU, actual: StorageDevice::GPU, operation: "item_f32() - GPU not yet supported".to_string() })
            }
        }
    }

    /// Extracts the single scalar value as f64 from a tensor containing exactly one element.
    ///
    /// Performs F32 -> F64 conversion if necessary.
    /// Returns an error if the tensor is not scalar or if offset is invalid.
    pub fn item_f64(&self) -> Result<f64, NeuraRustError> {
        let guard = self.read_data();
        if guard.numel() != 1 {
            return Err(NeuraRustError::ShapeMismatch { operation: "item_f64()".to_string(), expected: "1 element".to_string(), actual: format!("{} elements (shape {:?})", guard.numel(), guard.shape) });
        }
        let physical_offset = guard.offset;

        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::F32(arc_vec)) => {
                let data: &Vec<f32> = arc_vec;
                if physical_offset >= data.len() {
                     return Err(NeuraRustError::IndexOutOfBounds { index: vec![physical_offset], shape: vec![data.len()] });
                }
                // Conversion F32 -> F64
                Ok(data[physical_offset] as f64)
            }
            Buffer::Cpu(CpuBuffer::F64(arc_vec)) => {
                let data: &Vec<f64> = arc_vec;
                 if physical_offset >= data.len() {
                     return Err(NeuraRustError::IndexOutOfBounds { index: vec![physical_offset], shape: vec![data.len()] });
                }
                Ok(data[physical_offset]) // Direct access
            }
            Buffer::Gpu { .. } => {
                Err(NeuraRustError::DeviceMismatch { expected: StorageDevice::CPU, actual: StorageDevice::GPU, operation: "item_f64() - GPU not yet supported".to_string() })
            }
        }
    }

    // NOTE: get_f32_data / get_f64_data are currently still in mod.rs
    // We can decide later if they belong here.

    /// Acquires a read lock on the tensor's data.
    ///
    /// This allows reading the `TensorData` fields immutably.
    /// The lock is automatically released when the guard goes out of scope.
    /// Panics if the RwLock is poisoned.
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<'_, crate::tensor_data::TensorData> {
        self.data.read().expect("RwLock poisoned")
    }

    /// Acquires a write lock on the tensor's data.
    ///
    /// This allows modifying the `TensorData` fields mutably.
    /// The lock is automatically released when the guard goes out of scope.
    /// Panics if the RwLock is poisoned.
    pub fn write_data(&self) -> std::sync::RwLockWriteGuard<'_, crate::tensor_data::TensorData> {
        self.data.write().expect("RwLock poisoned")
    }

    /// Attempts to get the tensor data as a `Vec<f32>`.
    /// Returns an error if the tensor is not on the CPU or not F32.
    /// This method now correctly handles non-contiguous tensors by creating a new Vec.
    pub fn get_f32_data(&self) -> Result<Vec<f32>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_f32_data".to_string(),
            });
        }
        if guard.dtype != DType::F32 {
            return Err(NeuraRustError::UnsupportedOperation(
                format!("get_f32_data requires DType::F32, got {:?}", guard.dtype)
            ));
        }

        let buffer_arc = guard.buffer().try_get_cpu_f32()?;
        let underlying_data: &Vec<f32> = buffer_arc;

        let numel = guard.numel();
        let mut result_vec = Vec::with_capacity(numel);

        if numel == 0 {
            return Ok(result_vec);
        }

        // Need to use the index_to_coord utility
        // Assuming utils is accessible, otherwise need crate::tensor::utils::index_to_coord
        for i in 0..numel {
            let coords = crate::tensor::utils::index_to_coord(i, &guard.shape);
            let physical_offset = guard.get_offset(&coords);

            if physical_offset >= underlying_data.len() {
                return Err(NeuraRustError::InternalError(format!(
                    "Calculated physical offset {} is out of bounds for buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                    physical_offset,
                    underlying_data.len(),
                    i,
                    coords,
                    guard.shape,
                    guard.strides,
                    guard.offset
                )));
            }
            result_vec.push(underlying_data[physical_offset]);
        }

        Ok(result_vec)
    }

    /// Attempts to get the tensor data as a `Vec<f64>`.
    /// Returns an error if the tensor is not on the CPU or not F64.
    pub fn get_f64_data(&self) -> Result<Vec<f64>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_f64_data".to_string(),
            });
        }
        if guard.dtype != DType::F64 {
            return Err(NeuraRustError::DataTypeMismatch {
                expected: DType::F64,
                actual: guard.dtype,
                operation: "get_f64_data".to_string(),
            });
        }
        
        let buffer_arc = guard.buffer().try_get_cpu_f64()?;
        let underlying_data: &Vec<f64> = buffer_arc;

        let numel = guard.numel();
        let mut result_vec = Vec::with_capacity(numel);

        if numel == 0 {
            return Ok(result_vec);
        }

        for i in 0..numel {
            let coords = crate::tensor::utils::index_to_coord(i, &guard.shape);
            let physical_offset = guard.get_offset(&coords);

            if physical_offset >= underlying_data.len() {
                 return Err(NeuraRustError::InternalError(format!(
                    "Calculated physical offset {} is out of bounds for F64 buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                    physical_offset,
                    underlying_data.len(),
                    i,
                    coords,
                    guard.shape,
                    guard.strides,
                    guard.offset
                )));
            }
            result_vec.push(underlying_data[physical_offset]);
        }
        
        Ok(result_vec)
    }
}