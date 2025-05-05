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
    /// Data is stored on a CUDA-enabled NVIDIA GPU.
    ///
    /// **Note:** GPU support is planned for future phases and currently operations
    /// primarily target the CPU.
    /// TODO: Add device ID/index when multiple GPUs are supported.
    GPU,
    // TODO: Potentially add other devices like TPUs, Metal (Apple Silicon) in the future.
}

