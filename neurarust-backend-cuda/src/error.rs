use thiserror::Error;
use rustacuda::error::CudaError as RustCudaError;

#[derive(Error, Debug)]
pub enum CudaBackendError {
    #[error("CUDA backend is not initialized. Call init_cuda() first.")]
    NotInitialized,
    #[error("CUDA runtime initialization failed: {0}")]
    InitializationError(RustCudaError),
    #[error("CUDA device error: {0}")]
    DeviceError(RustCudaError),
    #[error("CUDA context error: {0}")]
    ContextError(RustCudaError),
    #[error("CUDA driver error: {source}")]
    DriverError { #[from] source: RustCudaError },
    #[error("No CUDA devices found")]
    NoDevicesFound,
    #[error("Invalid CUDA device ID: {0}")]
    InvalidDeviceId(u32),
    #[error("Failed to get device properties for device {0}: {1}")]
    DevicePropertiesError(u32, RustCudaError),
    #[error("Feature not yet implemented: {0}")]
    NotImplemented(String),
    #[error("Failed to set current CUDA context: {0}")]
    SetCurrentContextError(RustCudaError),
    #[error("Failed to pop current CUDA context: {0}")]
    PopCurrentContextError(RustCudaError),
    #[error("Failed to create CUDA context for device {0}: {1}")]
    ContextCreationError(u32, RustCudaError),
    #[error("Logger initialization failed: {0}")]
    LoggerInitError(String),
} 