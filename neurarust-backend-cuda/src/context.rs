use crate::error::CudaBackendError;
use rustacuda::context::{Context, ContextFlags, CurrentContext};
use rustacuda::device::Device as CudaDeviceDrv;
use rustacuda::error::CudaError as RustCudaError;
use rustacuda::CudaFlags;
use once_cell::sync::OnceCell;
use std::sync::{Mutex, Arc};
use log::{debug, error, info, warn};

// Global static for ensuring CUDA is initialized only once.
static CUDA_INITIALIZED: OnceCell<Result<(), RustCudaError>> = OnceCell::new();

/// Initializes the CUDA Driver API and the logging system.
/// This function is idempotent and thread-safe.
/// It must be called before any other CUDA operations.
///
/// The logging level can be controlled by the `RUST_LOG` environment variable
/// (e.g., `RUST_LOG=neurarust_backend_cuda=debug`).
pub fn initialize_cuda_with_logging() -> Result<(), CudaBackendError> {
    // Initialize env_logger. This can be called multiple times, but only the first call has effect.
    // We wrap it in a OnceCell to ensure it's attempted at least once if this function is called.
    // Errors during logger init are reported but don't stop CUDA init.
    static LOGGER_INITIALIZED: OnceCell<()> = OnceCell::new();
    LOGGER_INITIALIZED.get_or_init(|| {
        match env_logger::builder().is_test(false).try_init() {
            Ok(_) => info!("NeuraRust CUDA Backend Logger initialized."),
            Err(e) => eprintln!("Failed to initialize logger: {}. Logging might not work as expected.", e),
        };
    });

    match CUDA_INITIALIZED.get_or_try_init(|| {
        debug!("Attempting to initialize CUDA runtime...");
        rustacuda::init(CudaFlags::empty())
    }) {
        Ok(Ok(_)) => {
            info!("CUDA runtime initialized successfully.");
            Ok(())
        }
        Ok(Err(e)) => {
            error!("CUDA runtime initialization failed: {}", e);
            Err(CudaBackendError::InitializationError(*e))
        }
        Err(e) => {
            // This case should ideally not happen if OnceCell is used correctly,
            // but we handle it for robustness.
            error!("CUDA runtime initialization failed due to OnceCell error: {}", e);
            Err(CudaBackendError::InitializationError(*e))
        }
    }
}

/// Represents a CUDA device with its properties.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub id: u32,
    pub name: String,
    pub total_memory: usize,
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    // Keep a reference to the rustacuda Device object for direct operations if needed
    // Note: CudaDeviceDrv is not Clone, so we can't store it directly if CudaDevice is Clone.
    // For now, we re-acquire it when needed for context creation.
}

impl CudaDevice {
    fn from_rustacuda_device(device_drv: CudaDeviceDrv, id: u32) -> Result<Self, CudaBackendError> {
        let name = device_drv.name().map_err(|e| {
            error!("Failed to get name for device {}: {}", id, e);
            CudaBackendError::DevicePropertiesError(id, e)
        })?;
        let total_memory = device_drv.total_memory().map_err(|e| {
            error!("Failed to get total memory for device {}: {}", id, e);
            CudaBackendError::DevicePropertiesError(id, e)
        })?;
        let compute_capability = device_drv.compute_capability().map_err(|e| {
            error!("Failed to get compute capability for device {}: {}", id, e);
            CudaBackendError::DevicePropertiesError(id, e)
        })?;
        debug!("Found device {}: {} with {}MB VRAM, CC {}.{}", id, name, total_memory / (1024 * 1024), compute_capability.0, compute_capability.1);
        Ok(CudaDevice {
            id,
            name,
            total_memory,
            compute_capability,
        })
    }
}

/// Lists all available CUDA devices.
/// Ensures CUDA is initialized before listing devices.
///
/// # Returns
/// A `Result` containing a `Vec` of `CudaDevice` or a `CudaBackendError`.
pub fn list_devices() -> Result<Vec<CudaDevice>, CudaBackendError> {
    initialize_cuda_with_logging()?;

    debug!("Querying for number of CUDA devices...");
    let num_devices = CudaDeviceDrv::count().map_err(|e| {
        error!("Failed to count CUDA devices: {}", e);
        CudaBackendError::DeviceError(e)
    })?;

    if num_devices == 0 {
        warn!("No CUDA devices found.");
        return Err(CudaBackendError::NoDevicesFound);
    }
    info!("Found {} CUDA device(s).", num_devices);

    let mut devices = Vec::with_capacity(num_devices as usize);
    for i in 0..num_devices {
        let device_id = i as u32;
        match CudaDeviceDrv::get_device(device_id) {
            Ok(rust_device) => {
                devices.push(CudaDevice::from_rustacuda_device(rust_device, device_id)?);
            }
            Err(e) => {
                error!("Failed to get device {}: {}", device_id, e);
                return Err(CudaBackendError::DeviceError(e));
            }
        }
    }
    Ok(devices)
}

/// Represents a CUDA context.
/// The context is destroyed when this struct goes out of scope.
#[derive(Debug)]
pub struct CudaContext {
    inner: Arc<Context>,
    pub device_id: u32, // Store device ID for reference
}

impl CudaContext {
    /// Creates a new CUDA context for the specified device and makes it current for the calling thread.
    /// Ensure `initialize_cuda_with_logging()` has been called before this.
    ///
    /// # Arguments
    /// * `device`: The `CudaDevice` to create the context on.
    /// * `flags`: `ContextFlags` for context creation.
    ///
    /// # Returns
    /// A `Result` containing the new `CudaContext` or a `CudaBackendError`.
    pub fn new(device: &CudaDevice, flags: ContextFlags) -> Result<Self, CudaBackendError> {
        initialize_cuda_with_logging()?; // Ensure initialized
        debug!("Creating new CUDA context for device ID: {}", device.id);
        let rust_device = CudaDeviceDrv::get_device(device.id).map_err(|e|{
            error!("Failed to get CudaDeviceDrv for device ID {}: {}", device.id, e);
            CudaBackendError::DeviceError(e)
        })?;
        let context = Context::create_and_push(flags, rust_device).map_err(|e| {
            error!("Failed to create and push context for device {}: {}", device.id, e);
            CudaBackendError::ContextCreationError(device.id, e)
        })?;
        info!("Successfully created and pushed CUDA context for device ID: {}", device.id);
        Ok(CudaContext {
            inner: Arc::new(context),
            device_id: device.id,
        })
    }

    /// Sets this context as the current context for the calling thread.
    pub fn set_current(&self) -> Result<(), CudaBackendError> {
        debug!("Setting current context for device ID: {} (Thread: {:?})", self.device_id, std::thread::current().id());
        CurrentContext::set_current(&self.inner).map_err(|e| {
            error!("Failed to set current context (device {}): {}", self.device_id, e);
            CudaBackendError::SetCurrentContextError(e)
        })
    }

    /// Gets the underlying `rustacuda::context::Context`.
    /// This is useful for operations that require a direct reference to the `rustacuda` context.
    pub fn rust_context(&self) -> &Arc<Context> {
        &self.inner
    }

    /// Gets the ID of the device associated with this context.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

// `Context` from `rustacuda` handles its own destruction (cuCtxDestroy) when it's dropped.
// `CurrentContext::pop()` is called when `CurrentContextGuard` is dropped after `create_and_push` or `set_current`.
// Our `CudaContext` wraps an `Arc<Context>`, so the actual CUDA context is destroyed when the last Arc is dropped.

/// Retrieves the primary context for a device, creating it if necessary, and retains it.
/// This context is then set as the current context for the calling thread.
/// The primary context is intended for simple, single-context-per-device applications.
/// Note: The context will be destroyed when the returned `CudaContext` is dropped.
pub fn get_primary_context(device_id: u32) -> Result<CudaContext, CudaBackendError> {
    initialize_cuda_with_logging()?;
    debug!("Attempting to get primary context for device ID: {}", device_id);
    let rust_device = CudaDeviceDrv::get_device(device_id).map_err(|e| {
        error!("Failed to get CudaDeviceDrv for device ID {}: {}", device_id, e);
        CudaBackendError::DeviceError(e)
    })?;

    // Using primary_context_retain, which increases the ref count if it exists, or creates and sets.
    // This context should be active on the calling thread after this call.
    let pctx = rust_device.primary_context_retain().map_err(|e| {
        error!("Failed to retain primary context for device {}: {}", device_id, e);
        CudaBackendError::ContextCreationError(device_id, e)
    })?;
    info!("Successfully retained and set primary CUDA context for device ID: {}", device_id);
    Ok(CudaContext{
        inner: Arc::new(pctx), // Wrap in Arc for consistent API, though primary_context_retain already handles lifetime.
        device_id,
    })
}

/// Gets information about the current CUDA context on the calling thread, if any.
pub fn current_context_info() -> Result<Option<(u32, ContextFlags)>, CudaBackendError> {
    initialize_cuda_with_logging()?;
    match CurrentContext::get_current() {
        Ok(Some(context_ref)) => {
            let device = CurrentContext::get_device().map_err(CudaBackendError::DeviceError)?;
            let flags = Context::get_flags().map_err(CudaBackendError::ContextError)?;
            debug!("Current context on device: {}, flags: {:?}", device.ordinal(), flags);
            Ok(Some((device.ordinal() as u32, flags)))
        }
        Ok(None) => {
            debug!("No current CUDA context on this thread.");
            Ok(None)
        }
        Err(e) => {
            error!("Error getting current context: {}", e);
            Err(CudaBackendError::ContextError(e))
        }
    }
}

// Placeholder for CudaContext. Will be developed further.
// Per the roadmap: "centralized CUDA context management: Ensure clear creation,
// activation (setting current), and explicit destruction of contexts, ideally associated with device IDs."
// This will likely involve thread-local contexts or a more sophisticated manager.

// Remove old dummy placeholders if they exist
/*
lazy_static::lazy_static! {
    static ref CUDA_CONTEXTS: RwLock<HashMap<u32, CudaContext>> = RwLock::new(HashMap::new());
}

pub fn old_initialize_cuda() -> Result<(), String> {
    println!("CUDA Placeholder: Initializing CUDA...");
    Ok(())
}

pub fn old_list_devices() -> Result<Vec<CudaDevice>, String> {
    println!("CUDA Placeholder: Listing devices...");
    Ok(vec![CudaDevice {
        id: 0,
        name: "Dummy CUDA Device".to_string(),
        total_memory: 1024 * 1024 * 1024, // 1GB
        compute_capability: (0,0), // Added to match new struct
    }])
}
*/ 