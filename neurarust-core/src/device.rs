use std::fmt::Debug;

/// Represents the physical location where tensor data is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
pub enum StorageDevice {
    /// Data is stored in main system memory (RAM).
    #[default]
    CPU,
    /// Data is stored on a specific GPU.
    /// TODO: Add device ID/index when multiple GPUs are supported.
    GPU,
    // TODO: Potentially add other devices like TPUs in the future.
}

