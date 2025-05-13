// src/tensor/accessors.rs
use crate::{
    buffer::{Buffer, CpuBuffer},
    device::StorageDevice,
    error::NeuraRustError,
    tensor::Tensor,
    types::DType,
};

/// This `impl` block provides methods for accessing the properties and data of a `Tensor`.
impl Tensor {
    /// Returns a clone of the tensor's shape (dimensions).
    ///
    /// Acquires a read lock internally.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(t.shape(), vec![2, 2]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        self.read_data().shape.clone()
    }

    /// Returns a clone of the tensor's strides.
    ///
    /// Strides define the number of elements to jump in the underlying storage
    /// to move one step along each dimension.
    /// Acquires a read lock internally.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(t.strides(), vec![2, 1]); // Contiguous strides for [2, 2]
    /// ```
    pub fn strides(&self) -> Vec<usize> {
        self.read_data().strides.clone()
    }

    /// Returns the storage device (`CPU` or `GPU`) where the tensor data resides.
    ///
    /// Acquires a read lock internally.
    pub fn device(&self) -> StorageDevice {
        self.read_data().device
    }

    /// Returns the data type (`DType::F32` or `DType::F64`) of the tensor elements.
    ///
    /// Acquires a read lock internally.
    pub fn dtype(&self) -> DType {
        self.read_data().dtype
    }

    /// Returns the rank (number of dimensions) of the tensor.
    ///
    /// Equal to `tensor.shape().len()`.
    /// Acquires a read lock internally.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(t.rank(), 2);
    /// let s = Tensor::new(vec![5.0f32], vec![1]).unwrap();
    /// assert_eq!(s.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        self.read_data().shape.len()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// Equal to the product of the sizes of all dimensions.
    /// Acquires a read lock internally.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.numel(), 6);
    /// let empty = Tensor::new(vec![], vec![2, 0, 3]).unwrap();
    /// assert_eq!(empty.numel(), 0);
    /// ```
    pub fn numel(&self) -> usize {
        // Reuse shape() ? No, direct access is fine.
        self.read_data().shape.iter().product()
    }

     /// Checks if the tensor is contiguous in memory.
     ///
     /// A tensor is considered contiguous if its elements are laid out in memory
     /// sequentially in row-major order (C order), without gaps, considering its
     /// shape and strides. Operations on contiguous tensors are often more efficient.
     /// Views created by operations like `transpose` or slicing might result in non-contiguous tensors.
     ///
     /// Acquires a read lock internally.
     pub fn is_contiguous(&self) -> bool {
         self.read_data().is_contiguous() // Delegate to TensorData
     }

    /// Extracts the single scalar value as `f32` from a tensor containing exactly one element.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The tensor does not contain exactly one element (`numel() != 1`).
    /// - The tensor's data type is not `DType::F32`.
    /// - The tensor is stored on the GPU (currently unsupported for this operation).
    /// - The internal offset points outside the bounds of the underlying buffer.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let scalar = Tensor::new(vec![42.0f32], vec![]).unwrap();
    /// assert_eq!(scalar.item_f32().unwrap(), 42.0);
    ///
    /// let vec_t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    /// assert!(vec_t.item_f32().is_err()); // Not a scalar
    /// ```
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
            &Buffer::Cpu(CpuBuffer::I32(_)) | &Buffer::Cpu(CpuBuffer::I64(_)) | &Buffer::Cpu(CpuBuffer::Bool(_)) => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: self.dtype(),
                    operation: "item_f32() appelé sur un tenseur non-f32".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "item_f32() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Extracts the single scalar value as `f64` from a tensor containing exactly one element.
    ///
    /// If the tensor's data type is `DType::F32`, the value is converted to `f64`.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The tensor does not contain exactly one element (`numel() != 1`).
    /// - The tensor is stored on the GPU (currently unsupported for this operation).
    /// - The internal offset points outside the bounds of the underlying buffer.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let scalar_f64 = Tensor::new_f64(vec![99.9], vec![]).unwrap();
    /// assert_eq!(scalar_f64.item_f64().unwrap(), 99.9);
    ///
    /// let scalar_f32 = Tensor::new(vec![3.14f32], vec![]).unwrap();
    /// // Compare against the f32 literal cast to f64 for robust comparison
    /// assert!((scalar_f32.item_f64().unwrap() - (3.14f32 as f64)).abs() < 1e-9);
    ///
    /// let vec_t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    /// assert!(vec_t.item_f64().is_err()); // Not a scalar
    /// ```
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
            &Buffer::Cpu(CpuBuffer::I32(_)) | &Buffer::Cpu(CpuBuffer::I64(_)) | &Buffer::Cpu(CpuBuffer::Bool(_)) => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F64,
                    actual: self.dtype(),
                    operation: "item_f64() appelé sur un tenseur non-f64".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "item_f64() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Accesses a single element at the specified indices as `f32`.
    ///
    /// Performs bounds checking for each dimension.
    ///
    /// # Arguments
    /// * `indices`: A slice representing the coordinates of the element to access.
    ///   The length of the slice must match the rank of the tensor.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The number of indices does not match the tensor's rank (`RankMismatch`).
    /// - Any index is out of bounds for its corresponding dimension (`IndexOutOfBounds`).
    /// - The tensor's data type is not `DType::F32` (`DataTypeMismatch`).
    /// - The tensor is stored on the `GPU` (`DeviceMismatch`).
    /// - An internal error occurs during offset calculation.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.at_f32(&[0, 1]).unwrap(), 2.0);
    /// assert_eq!(t.at_f32(&[1, 2]).unwrap(), 6.0);
    /// assert!(t.at_f32(&[0, 3]).is_err()); // Index out of bounds
    /// assert!(t.at_f32(&[0]).is_err());    // Rank mismatch
    /// ```
    pub fn at_f32(&self, indices: &[usize]) -> Result<f32, NeuraRustError> {
        let guard = self.read_data();
        let rank = guard.shape.len();

        // 1. Check rank
        if indices.len() != rank {
            return Err(NeuraRustError::RankMismatch {
                expected: rank,
                actual: indices.len(),
            });
        }

        // 2. Check bounds for each index
        for (dim, &index) in indices.iter().enumerate() {
            if index >= guard.shape[dim] {
                return Err(NeuraRustError::IndexOutOfBounds {
                    index: indices.to_vec(),
                    shape: guard.shape.clone(),
                });
            }
        }

        // 3. Calculate physical offset
        let mut physical_offset = guard.offset;
        for (dim, &index) in indices.iter().enumerate() {
            physical_offset += index * guard.strides[dim];
        }

        // 4. Access data in buffer
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::F32(arc_vec)) => {
                let data: &Vec<f32> = arc_vec;
                // Double-check physical offset (should be unlikely after coord check, but safe)
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Calculated physical offset {} is out of buffer bounds {} in at_f32", physical_offset, data.len())
                    ));
                 }
                Ok(data[physical_offset])
            }
            Buffer::Cpu(CpuBuffer::F64(_)) => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: DType::F64,
                    operation: "at_f32() called on F64 tensor".to_string(),
                })
            }
            Buffer::Gpu { .. } => {
                Err(NeuraRustError::DeviceMismatch { expected: StorageDevice::CPU, actual: StorageDevice::GPU, operation: "at_f32() - GPU not yet supported".to_string() })
            }
            &Buffer::Cpu(CpuBuffer::I32(_)) | &Buffer::Cpu(CpuBuffer::I64(_)) | &Buffer::Cpu(CpuBuffer::Bool(_)) => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: self.dtype(),
                    operation: "at_f32() appelé sur un tenseur non-f32".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "at_f32() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Accesses a single element at the specified indices as `f64`.
    ///
    /// Performs bounds checking for each dimension.
    /// If the tensor's `DType` is `F32`, the value is converted to `f64`.
    ///
    /// # Arguments
    /// * `indices`: A slice representing the coordinates of the element to access.
    ///   The length of the slice must match the rank of the tensor.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The number of indices does not match the tensor's rank (`RankMismatch`).
    /// - Any index is out of bounds for its corresponding dimension (`IndexOutOfBounds`).
    /// - The tensor's data type is neither `DType::F32` nor `DType::F64` (`DataTypeMismatch`).
    /// - The tensor is stored on the `GPU` (`DeviceMismatch`).
    /// - An internal error occurs during offset calculation.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t_f64 = Tensor::new_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t_f64.at_f64(&[0, 1]).unwrap(), 2.0);
    /// assert_eq!(t_f64.at_f64(&[1, 2]).unwrap(), 6.0);
    ///
    /// let t_f32 = Tensor::new(vec![1.5f32, 2.5f32], vec![2]).unwrap();
    /// assert!((t_f32.at_f64(&[0]).unwrap() - 1.5).abs() < 1e-9);
    /// ```
    pub fn at_f64(&self, indices: &[usize]) -> Result<f64, NeuraRustError> {
        let guard = self.read_data();
        let rank = guard.shape.len();

        if indices.len() != rank {
            return Err(NeuraRustError::RankMismatch {
                expected: rank,
                actual: indices.len(),
            });
        }

        for (dim, &index) in indices.iter().enumerate() {
            if index >= guard.shape[dim] {
                return Err(NeuraRustError::IndexOutOfBounds {
                    index: indices.to_vec(),
                    shape: guard.shape.clone(),
                });
            }
        }

        let mut physical_offset = guard.offset;
        for (dim, &index) in indices.iter().enumerate() {
            physical_offset += index * guard.strides[dim];
        }

        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::F64(arc_vec)) => {
                let data: &Vec<f64> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Calculated physical offset {} is out of buffer bounds {} in at_f64 for F64 tensor", physical_offset, data.len())
                    ));
                }
                Ok(data[physical_offset])
            }
            Buffer::Cpu(CpuBuffer::F32(arc_vec)) => {
                let data: &Vec<f32> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Calculated physical offset {} is out of buffer bounds {} in at_f64 for F32 tensor", physical_offset, data.len())
                    ));
                }
                Ok(data[physical_offset] as f64) // Convert F32 to F64
            }
            Buffer::Gpu { .. } => {
                Err(NeuraRustError::DeviceMismatch { expected: StorageDevice::CPU, actual: StorageDevice::GPU, operation: "at_f64() - GPU not yet supported".to_string() })
            }
            &Buffer::Cpu(CpuBuffer::I32(_)) | &Buffer::Cpu(CpuBuffer::I64(_)) | &Buffer::Cpu(CpuBuffer::Bool(_)) => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F64,
                    actual: self.dtype(),
                    operation: "at_f64() appelé sur un tenseur non-f64".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "at_f64() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Acquires a read lock on the tensor's internal [`TensorData`](../tensor_data/struct.TensorData.html).
    ///
    /// This returns a read guard, allowing immutable access to the fields of `TensorData`
    /// (like shape, strides, buffer, requires_grad, etc.).
    /// The lock is automatically released when the returned guard goes out of scope.
    /// Useful for accessing multiple properties without repeated locking.
    ///
    /// # Panics
    /// Panics if the internal `RwLock` is poisoned (i.e., a previous write access panicked).
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<'_, crate::tensor_data::TensorData> {
        self.data.read().expect("RwLock poisoned")
    }

    /// Acquires a write lock on the tensor's internal [`TensorData`](../tensor_data/struct.TensorData.html).
    ///
    /// This returns a write guard, allowing mutable access to the fields of `TensorData`.
    /// **Caution:** Modifying internal fields directly can lead to an inconsistent state
    /// if not done carefully (e.g., changing shape without updating strides or buffer).
    /// Prefer using dedicated tensor operations when possible.
    /// The lock is automatically released when the returned guard goes out of scope.
    ///
    /// # Panics
    /// Panics if the internal `RwLock` is poisoned.
    pub fn write_data(&self) -> std::sync::RwLockWriteGuard<'_, crate::tensor_data::TensorData> {
        self.data.write().expect("RwLock poisoned")
    }

    /// Copies the tensor's logical data into a new `Vec<f32>`.
    ///
    /// This method correctly handles tensors that are views (non-contiguous)
    /// by iterating through the logical elements according to shape, strides, and offset,
    /// and copying them into a newly allocated, contiguous `Vec<f32>`.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The tensor is not stored on the `CPU`.
    /// - The tensor's data type is not `DType::F32`.
    /// - An internal error occurs during index calculation (e.g., offset out of bounds).
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// assert_eq!(t.get_f32_data().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///
    /// // Example with a view (e.g., transpose - needs transpose impl)
    /// // let transposed = t.transpose(0, 1).unwrap(); // Assume shape [3, 2], non-contiguous
    /// // assert_eq!(transposed.get_f32_data().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // Logical order
    /// ```
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

    /// Copies the tensor's logical data into a new `Vec<f64>`.
    ///
    /// This method correctly handles tensors that are views (non-contiguous)
    /// by iterating through the logical elements according to shape, strides, and offset,
    /// and copying them into a newly allocated, contiguous `Vec<f64>`.
    /// If the tensor's `DType` is `F32`, the values are converted to `f64` during the copy.
    ///
    /// # Errors
    /// Returns `NeuraRustError` if:
    /// - The tensor is not stored on the `CPU`.
    /// - The tensor's data type is neither `DType::F32` nor `DType::F64`.
    /// - An internal error occurs during index calculation.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::tensor::Tensor;
    /// let t_f64 = Tensor::new_f64(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// assert_eq!(t_f64.get_f64_data().unwrap(), vec![1.0, 2.0, 3.0]);
    ///
    /// let t_f32 = Tensor::new(vec![1.5f32, 2.5f32], vec![2]).unwrap();
    /// let data_f64 = t_f32.get_f64_data().unwrap();
    /// assert!((data_f64[0] - 1.5).abs() < 1e-9);
    /// assert!((data_f64[1] - 2.5).abs() < 1e-9);
    /// ```
    pub fn get_f64_data(&self) -> Result<Vec<f64>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_f64_data".to_string(),
            });
        }

        let numel = guard.numel();
        let mut result_vec = Vec::with_capacity(numel);
        if numel == 0 {
            return Ok(result_vec);
        }

        match guard.dtype {
            DType::F64 => {
                let buffer_arc = guard.buffer().try_get_cpu_f64()?;
                let underlying_data: &Vec<f64> = buffer_arc;
                for i in 0..numel {
                    let coords = crate::tensor::utils::index_to_coord(i, &guard.shape);
                    let physical_offset = guard.get_offset(&coords);
                    if physical_offset >= underlying_data.len() {
                        return Err(NeuraRustError::InternalError(format!(
                            "Calculated physical offset {} is out of bounds for F64 buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                            physical_offset, underlying_data.len(), i, coords, guard.shape, guard.strides, guard.offset
                        )));
                    }
                    result_vec.push(underlying_data[physical_offset]);
                }
            }
            DType::F32 => {
                let buffer_arc = guard.buffer().try_get_cpu_f32()?;
                let underlying_data: &Vec<f32> = buffer_arc;
                 for i in 0..numel {
                    let coords = crate::tensor::utils::index_to_coord(i, &guard.shape);
                    let physical_offset = guard.get_offset(&coords);
                    if physical_offset >= underlying_data.len() {
                        return Err(NeuraRustError::InternalError(format!(
                            "Calculated physical offset {} is out of bounds for F32 buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                            physical_offset, underlying_data.len(), i, coords, guard.shape, guard.strides, guard.offset
                        )));
                    }
                    // Convert F32 to F64
                    result_vec.push(underlying_data[physical_offset] as f64);
                 }
            }
            DType::I32 | DType::I64 | DType::Bool => {
                return Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F64,
                    actual: guard.dtype,
                    operation: "get_f64_data requires F64 or F32".to_string(),
                });
            }
        }
        // Gestion explicite du buffer CUDA (hors match sur dtype)
        // if let Buffer::Cuda(_) = &*guard.buffer {
        //     return Err(NeuraRustError::DeviceMismatch {
        //         expected: StorageDevice::CPU,
        //         actual: StorageDevice::Cuda(0),
        //         operation: "get_f64_data - accès CPU sur buffer CUDA non supporté".to_string(),
        //     });
        // }

        Ok(result_vec)
    }

    /// Extrait la valeur scalaire unique comme `i32` d'un tenseur contenant exactement un élément.
    pub fn item_i32(&self) -> Result<i32, NeuraRustError> {
        let guard = self.read_data();
        if guard.numel() != 1 {
            return Err(NeuraRustError::ShapeMismatch {
                operation: "item_i32()".to_string(),
                expected: "1 element".to_string(),
                actual: format!("{} elements (shape {:?})", guard.numel(), guard.shape),
            });
        }
        let physical_offset = guard.offset;
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::I32(arc_vec)) => {
                let data: &Vec<i32> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::IndexOutOfBounds { index: vec![physical_offset], shape: vec![data.len()] });
                }
                Ok(data[physical_offset])
            }
            Buffer::Cpu(CpuBuffer::F32(_)) | Buffer::Cpu(CpuBuffer::F64(_)) | Buffer::Cpu(CpuBuffer::I64(_)) | Buffer::Cpu(CpuBuffer::Bool(_)) | Buffer::Gpu { .. } => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::I32,
                    actual: guard.dtype,
                    operation: "item_i32()".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "item_i32() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Extrait la valeur scalaire unique comme `i64` d'un tenseur contenant exactement un élément.
    pub fn item_i64(&self) -> Result<i64, NeuraRustError> {
        let guard = self.read_data();
        if guard.numel() != 1 {
            return Err(NeuraRustError::ShapeMismatch {
                operation: "item_i64()".to_string(),
                expected: "1 element".to_string(),
                actual: format!("{} elements (shape {:?})", guard.numel(), guard.shape),
            });
        }
        let physical_offset = guard.offset;
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::I64(arc_vec)) => {
                let data: &Vec<i64> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::IndexOutOfBounds { index: vec![physical_offset], shape: vec![data.len()] });
                }
                Ok(data[physical_offset])
            }
            Buffer::Cpu(CpuBuffer::F32(_)) | Buffer::Cpu(CpuBuffer::F64(_)) | Buffer::Cpu(CpuBuffer::I32(_)) | Buffer::Cpu(CpuBuffer::Bool(_)) | Buffer::Gpu { .. } => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::I64,
                    actual: guard.dtype,
                    operation: "item_i64()".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "item_i64() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Extrait la valeur scalaire unique comme `bool` d'un tenseur contenant exactement un élément.
    pub fn item_bool(&self) -> Result<bool, NeuraRustError> {
        let guard = self.read_data();
        if guard.numel() != 1 {
            return Err(NeuraRustError::ShapeMismatch {
                operation: "item_bool()".to_string(),
                expected: "1 element".to_string(),
                actual: format!("{} elements (shape {:?})", guard.numel(), guard.shape),
            });
        }
        let physical_offset = guard.offset;
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::Bool(arc_vec)) => {
                let data: &Vec<bool> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::IndexOutOfBounds { index: vec![physical_offset], shape: vec![data.len()] });
                }
                Ok(data[physical_offset])
            }
            Buffer::Cpu(CpuBuffer::F32(_)) | Buffer::Cpu(CpuBuffer::F64(_)) | Buffer::Cpu(CpuBuffer::I32(_)) | Buffer::Cpu(CpuBuffer::I64(_)) | Buffer::Gpu { .. } => {
                Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::Bool,
                    actual: guard.dtype,
                    operation: "item_bool()".to_string(),
                })
            }
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "item_bool() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Accède à l'élément à l'index donné comme `i32` (pour DType::I32)
    pub fn at_i32(&self, index: &[usize]) -> Result<i32, NeuraRustError> {
        let guard = self.read_data();
        let rank = guard.shape.len();
        if index.len() != rank {
            return Err(NeuraRustError::RankMismatch {
                expected: rank,
                actual: index.len(),
            });
        }
        for (dim, &idx) in index.iter().enumerate() {
            if idx >= guard.shape[dim] {
                return Err(NeuraRustError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: guard.shape.clone(),
                });
            }
        }
        let mut physical_offset = guard.offset;
        for (dim, &idx) in index.iter().enumerate() {
            physical_offset += idx * guard.strides[dim];
        }
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::I32(arc_vec)) => {
                let data: &Vec<i32> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Calculated physical offset {} is out of buffer bounds {} in at_i32", physical_offset, data.len())
                    ));
                }
                Ok(data[physical_offset])
            }
            _ => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I32,
                actual: guard.dtype,
                operation: "at_i32()".to_string(),
            }),
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "at_i32() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Accède à l'élément à l'index donné comme `i64` (pour DType::I64)
    pub fn at_i64(&self, index: &[usize]) -> Result<i64, NeuraRustError> {
        let guard = self.read_data();
        let rank = guard.shape.len();
        if index.len() != rank {
            return Err(NeuraRustError::RankMismatch {
                expected: rank,
                actual: index.len(),
            });
        }
        for (dim, &idx) in index.iter().enumerate() {
            if idx >= guard.shape[dim] {
                return Err(NeuraRustError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: guard.shape.clone(),
                });
            }
        }
        let mut physical_offset = guard.offset;
        for (dim, &idx) in index.iter().enumerate() {
            physical_offset += idx * guard.strides[dim];
        }
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::I64(arc_vec)) => {
                let data: &Vec<i64> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Calculated physical offset {} is out of buffer bounds {} in at_i64", physical_offset, data.len())
                    ));
                }
                Ok(data[physical_offset])
            }
            _ => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::I64,
                actual: guard.dtype,
                operation: "at_i64()".to_string(),
            }),
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "at_i64() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Accède à l'élément à l'index donné comme `bool` (pour DType::Bool)
    pub fn at_bool(&self, index: &[usize]) -> Result<bool, NeuraRustError> {
        let guard = self.read_data();
        let rank = guard.shape.len();
        if index.len() != rank {
            return Err(NeuraRustError::RankMismatch {
                expected: rank,
                actual: index.len(),
            });
        }
        for (dim, &idx) in index.iter().enumerate() {
            if idx >= guard.shape[dim] {
                return Err(NeuraRustError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: guard.shape.clone(),
                });
            }
        }
        let mut physical_offset = guard.offset;
        for (dim, &idx) in index.iter().enumerate() {
            physical_offset += idx * guard.strides[dim];
        }
        match &*guard.buffer {
            Buffer::Cpu(CpuBuffer::Bool(arc_vec)) => {
                let data: &Vec<bool> = arc_vec;
                if physical_offset >= data.len() {
                    return Err(NeuraRustError::InternalError(
                        format!("Calculated physical offset {} is out of buffer bounds {} in at_bool", physical_offset, data.len())
                    ));
                }
                Ok(data[physical_offset])
            }
            _ => Err(NeuraRustError::DataTypeMismatch {
                expected: DType::Bool,
                actual: guard.dtype,
                operation: "at_bool()".to_string(),
            }),
            // Buffer::Cuda(_) => Err(NeuraRustError::DeviceMismatch {
            //     expected: StorageDevice::CPU,
            //     actual: StorageDevice::Cuda(0),
            //     operation: "at_bool() - accès CPU sur buffer CUDA non supporté".to_string(),
            // }),
        }
    }

    /// Copie les données logiques du tenseur dans un nouveau `Vec<i32>`.
    ///
    /// Gère correctement les vues (non contiguës) en itérant selon shape, strides et offset.
    /// Retourne une erreur si le tenseur n'est pas sur le CPU ou n'est pas de type I32.
    pub fn get_i32_data(&self) -> Result<Vec<i32>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_i32_data".to_string(),
            });
        }
        if guard.dtype != DType::I32 {
            return Err(NeuraRustError::UnsupportedOperation(
                format!("get_i32_data requires DType::I32, got {:?}", guard.dtype)
            ));
        }
        let buffer_arc = guard.buffer().try_get_cpu_i32()?;
        let underlying_data: &Vec<i32> = buffer_arc;
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
                    "Calculated physical offset {} is out of bounds for I32 buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                    physical_offset, underlying_data.len(), i, coords, guard.shape, guard.strides, guard.offset
                )));
            }
            result_vec.push(underlying_data[physical_offset]);
        }
        Ok(result_vec)
    }

    /// Copie les données logiques du tenseur dans un nouveau `Vec<i64>`.
    pub fn get_i64_data(&self) -> Result<Vec<i64>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_i64_data".to_string(),
            });
        }
        if guard.dtype != DType::I64 {
            return Err(NeuraRustError::UnsupportedOperation(
                format!("get_i64_data requires DType::I64, got {:?}", guard.dtype)
            ));
        }
        let buffer_arc = guard.buffer().try_get_cpu_i64()?;
        let underlying_data: &Vec<i64> = buffer_arc;
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
                    "Calculated physical offset {} is out of bounds for I64 buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                    physical_offset, underlying_data.len(), i, coords, guard.shape, guard.strides, guard.offset
                )));
            }
            result_vec.push(underlying_data[physical_offset]);
        }
        Ok(result_vec)
    }

    /// Copie les données logiques du tenseur dans un nouveau `Vec<bool>`.
    pub fn get_bool_data(&self) -> Result<Vec<bool>, NeuraRustError> {
        let guard = self.read_data();
        if guard.device != StorageDevice::CPU {
            return Err(NeuraRustError::DeviceMismatch {
                expected: StorageDevice::CPU,
                actual: guard.device,
                operation: "get_bool_data".to_string(),
            });
        }
        if guard.dtype != DType::Bool {
            return Err(NeuraRustError::UnsupportedOperation(
                format!("get_bool_data requires DType::Bool, got {:?}", guard.dtype)
            ));
        }
        let buffer_arc = guard.buffer().try_get_cpu_bool()?;
        let underlying_data: &Vec<bool> = buffer_arc;
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
                    "Calculated physical offset {} is out of bounds for Bool buffer len {} (logical index {}, coords {:?}, shape {:?}, strides {:?}, offset {})",
                    physical_offset, underlying_data.len(), i, coords, guard.shape, guard.strides, guard.offset
                )));
            }
            result_vec.push(underlying_data[physical_offset]);
        }
        Ok(result_vec)
    }
}