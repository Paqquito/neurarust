//! Backend CUDA pour NeuraRust : gestion du contexte et des devices CUDA
//!
//! Fournit les fonctions d'initialisation, d'énumération et d'interrogation des devices CUDA.

use rustacuda::prelude::*;
use rustacuda::device::Device;
use rustacuda::error::CudaError;
use rustacuda::device::DeviceAttribute;
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::context::{Context, ContextFlags, CurrentContext};

/// Initialise le contexte CUDA (doit être appelé avant toute opération CUDA)
///
/// # Errors
/// Retourne une erreur si l'initialisation CUDA échoue.
pub fn init_cuda() -> Result<(), CudaError> {
    rustacuda::init(CudaFlags::empty())
}

/// Retourne le nombre de devices CUDA disponibles sur la machine.
///
/// # Errors
/// Retourne une erreur si la requête CUDA échoue.
pub fn device_count() -> Result<usize, CudaError> {
    Device::num_devices().map(|n| n as usize)
}

/// Récupère un device CUDA par son index.
///
/// # Arguments
/// * `index` - L'index du device (0 <= index < device_count())
///
/// # Errors
/// Retourne une erreur si l'index est invalide ou si la requête CUDA échoue.
pub fn get_device(index: usize) -> Result<Device, CudaError> {
    Device::get_device(index as u32)
}

/// Propriétés d'un device CUDA (nom, mémoire, compute capability)
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
}

/// Récupère les propriétés d'un device CUDA par son index.
///
/// # Arguments
/// * `index` - L'index du device (0 <= index < device_count())
///
/// # Errors
/// Retourne une erreur si l'index est invalide ou si la requête CUDA échoue.
pub fn device_properties(index: usize) -> Result<DeviceProperties, CudaError> {
    let device = get_device(index)?;
    let name = device.name()?;
    let total_memory = device.total_memory()? as usize;
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    Ok(DeviceProperties {
        name,
        total_memory,
        compute_capability_major: major,
        compute_capability_minor: minor,
    })
}

/// Crée et active un contexte CUDA sur le device donné (par défaut device 0)
///
/// # Arguments
/// * `device_index` - L'index du device CUDA (0 par défaut)
///
/// # Errors
/// Retourne une erreur si la création ou l'activation du contexte échoue
pub fn create_cuda_context(device_index: usize) -> Result<Context, CudaError> {
    let device = get_device(device_index)?;
    Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
}

/// Wrapper sûr pour un stream CUDA
///
/// ⚠️ Un contexte CUDA doit être actif (CurrentContext) lors de la création/destruction du stream !
#[derive(Debug)]
pub struct CudaStream {
    stream: Stream,
}

impl CudaStream {
    /// Crée un nouveau stream CUDA (par défaut, non-blocking si précisé)
    ///
    /// # Arguments
    /// * `non_blocking` - Si vrai, crée un stream non-bloquant
    ///
    /// # Errors
    /// Retourne une erreur si la création du stream échoue
    pub fn new(non_blocking: bool) -> Result<Self, CudaError> {
        let flags = if non_blocking {
            StreamFlags::NON_BLOCKING
        } else {
            StreamFlags::DEFAULT
        };
        let stream = Stream::new(flags, None)?;
        Ok(CudaStream { stream })
    }

    /// Synchronise ce stream CUDA (attend la fin de toutes les opérations soumises)
    ///
    /// # Errors
    /// Retourne une erreur si la synchronisation échoue
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.stream.synchronize()
    }

    /// Accès interne au stream rustacuda (pour usage avancé)
    pub fn inner(&self) -> &Stream {
        &self.stream
    }
}

// D'autres fonctions viendront ici (énumération, propriétés, etc.)

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_init_and_device_count() {
        // L'init doit réussir même si aucune carte n'est présente
        init_cuda().expect("CUDA init failed");
        let count = device_count().expect("device_count failed");
        // On ne peut pas garantir qu'il y a un GPU, mais la fonction doit marcher
        println!("CUDA device count: {}", count);
        // Si au moins un device, on teste get_device
        if count > 0 {
            let dev = get_device(0).expect("get_device(0) failed");
            println!("Device 0: {}", dev.name().unwrap_or("unknown".to_string()));
        }
    }

    #[test]
    fn test_device_properties() {
        init_cuda().expect("CUDA init failed");
        let count = device_count().expect("device_count failed");
        if count > 0 {
            let props = device_properties(0).expect("device_properties(0) failed");
            println!("Device 0 properties: {:?}", props);
            assert!(!props.name.is_empty());
            assert!(props.total_memory > 0);
        }
    }

    #[test]
    fn test_cuda_stream_create_and_sync() {
        init_cuda().expect("CUDA init failed");
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        // Création d'un stream par défaut
        let s = CudaStream::new(false).expect("Stream creation failed");
        s.synchronize().expect("Stream synchronize failed");
        // Création d'un stream non-bloquant
        let s2 = CudaStream::new(true).expect("Non-blocking stream creation failed");
        s2.synchronize().expect("Non-blocking stream synchronize failed");
        // Le contexte sera détruit à la fin du scope
    }

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
