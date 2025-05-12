use std::fmt::Debug;

/// Represents the physical location where tensor data is stored.
///
/// Tensors can reside on different devices, influencing where computations
/// are performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
pub enum StorageDevice {
    /// Data is stored in main system memory (RAM).
    /// This is the default device.
    #[default]
    CPU,
    /// Data is stored on a CUDA-enabled NVIDIA GPU (with device ID)
    Cuda(u32),
    /// Data is stored on a generic GPU (legacy, à déprécier)
    GPU,
    // TODO: Potentially add other devices like TPUs, Metal (Apple Silicon) in the future.
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_storage_device_cuda() {
        let dev = StorageDevice::Cuda(0);
        assert_eq!(format!("{:?}", dev), "Cuda(0)");
        let dev1 = StorageDevice::Cuda(1);
        assert_eq!(format!("{:?}", dev1), "Cuda(1)");
    }
}

