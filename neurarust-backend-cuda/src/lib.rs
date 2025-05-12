//! Backend CUDA pour NeuraRust : gestion du contexte et des devices CUDA
//!
//! Fournit les fonctions d'initialisation, d'énumération et d'interrogation des devices CUDA.

use rustacuda::prelude::*;
use rustacuda::device::Device;
use rustacuda::error::CudaError;
use rustacuda::device::DeviceAttribute;
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::context::{Context, ContextFlags, CurrentContext};
use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use std::ptr::null_mut;
use rustacuda::memory::DeviceBuffer;
unsafe extern "C" {
    pub fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> u32;
    pub fn cuMemFree_v2(dptr: u64) -> u32;
}
use std::ffi::c_void;

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

/// Bloc mémoire alloué sur le device CUDA (pointeur brut, taille, device)
#[derive(Debug)]
pub struct CudaMemoryBlock {
    pub device_ptr: u64, // device pointer (as u64 for FFI)
    pub size_bytes: usize,
    pub device_id: u32,
}

/// Allocateur mémoire CUDA avec cache par device (PyTorch-like)
#[derive(Debug)]
pub struct CudaCachingAllocator {
    // Pour chaque device, une hashmap: taille -> Vec<device_ptr libres>
    cache: Mutex<HashMap<u32, HashMap<usize, Vec<u64>>>>,
}

impl CudaCachingAllocator {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Alloue un bloc mémoire sur le device CUDA (avec cache)
    pub fn alloc(&self, size_bytes: usize, device_id: u32) -> Result<CudaMemoryBlock, CudaError> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(device_cache) = cache.get_mut(&device_id) {
            if let Some(blocks) = device_cache.get_mut(&size_bytes) {
                if let Some(ptr) = blocks.pop() {
                    return Ok(CudaMemoryBlock { device_ptr: ptr, size_bytes, device_id });
                }
            }
        }
        drop(cache);
        let _ctx = create_cuda_context(device_id as usize)?;
        let mut device_ptr: u64 = 0;
        unsafe {
            cuMemAlloc_v2(&mut device_ptr as *mut u64, size_bytes);
        }
        Ok(CudaMemoryBlock { device_ptr, size_bytes, device_id })
    }

    /// Libère (ou met en cache) un bloc mémoire CUDA
    pub fn free(&self, block: CudaMemoryBlock) -> Result<(), CudaError> {
        let mut cache = self.cache.lock().unwrap();
        let device_cache = cache.entry(block.device_id).or_insert_with(HashMap::new);
        let blocks = device_cache.entry(block.size_bytes).or_insert_with(Vec::new);
        blocks.push(block.device_ptr);
        Ok(())
    }

    /// Vide complètement le cache (libère toute la mémoire GPU)
    pub fn empty_cache(&self) -> Result<(), CudaError> {
        let mut cache = self.cache.lock().unwrap();
        for (_device_id, device_cache) in cache.iter_mut() {
            for (_size, blocks) in device_cache.iter_mut() {
                for ptr in blocks.drain(..) {
                    unsafe {
                        cuMemFree_v2(ptr as u64 as u64);
                    }
                }
            }
        }
        cache.clear();
        Ok(())
    }
}

// Singleton global pour l'allocateur (Arc pour partage thread-safe)
lazy_static::lazy_static! {
    pub static ref CUDA_ALLOCATOR: Arc<CudaCachingAllocator> = Arc::new(CudaCachingAllocator::new());
}

#[cfg(test)]
mod cuda_alloc_tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_alloc_and_free() {
        init_cuda().expect("CUDA init failed");
        let count = device_count().expect("device_count failed");
        if count == 0 { return; }
        let allocator = CudaCachingAllocator::new();
        let block = allocator.alloc(1024, 0).expect("alloc failed");
        assert_eq!(block.size_bytes, 1024);
        allocator.free(block).expect("free failed");
        // Réalloue, doit venir du cache
        let block2 = allocator.alloc(1024, 0).expect("alloc2 failed");
        assert_eq!(block2.size_bytes, 1024);
        allocator.free(block2).expect("free2 failed");
    }

    #[test]
    fn test_empty_cache() {
        init_cuda().expect("CUDA init failed");
        let count = device_count().expect("device_count failed");
        if count == 0 { return; }
        let allocator = CudaCachingAllocator::new();
        let block = allocator.alloc(2048, 0).expect("alloc failed");
        allocator.free(block).expect("free failed");
        allocator.empty_cache().expect("empty_cache failed");
    }

    #[test]
    fn test_thread_safety() {
        init_cuda().expect("CUDA init failed");
        let count = device_count().expect("device_count failed");
        if count == 0 { return; }
        let allocator = Arc::new(CudaCachingAllocator::new());
        let mut handles = vec![];
        for _ in 0..4 {
            let alloc = allocator.clone();
            handles.push(thread::spawn(move || {
                let block = alloc.alloc(4096, 0).expect("alloc failed");
                alloc.free(block).expect("free failed");
            }));
        }
        for h in handles { h.join().unwrap(); }
    }
}

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
