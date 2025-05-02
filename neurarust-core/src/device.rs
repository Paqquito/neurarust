use std::fmt::Debug;

/// Represents the physical location where tensor data is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageDevice {
    /// Data is stored in main system memory (RAM).
    CPU,
    /// Data is stored on a specific GPU.
    /// TODO: Add device ID/index when multiple GPUs are supported.
    GPU,
    // TODO: Potentially add other devices like TPUs in the future.
}

impl Default for StorageDevice {
    fn default() -> Self {
        StorageDevice::CPU
    }
} 