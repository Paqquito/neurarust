//! Allocateur mémoire CUDA avec cache (PyTorch-like)

use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use rustacuda::error::CudaError;
use crate::context::create_cuda_context;

unsafe extern "C" {
    pub fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> u32;
    pub fn cuMemFree_v2(dptr: u64) -> u32;
}

#[derive(Debug)]
pub struct CudaMemoryBlock {
    pub device_ptr: u64, // device pointer (as u64 for FFI)
    pub size_bytes: usize,
    pub device_id: u32,
}

#[derive(Debug)]
pub struct CudaCachingAllocator {
    cache: Mutex<HashMap<u32, HashMap<usize, Vec<u64>>>>,
}

impl CudaCachingAllocator {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

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

    pub fn free(&self, block: CudaMemoryBlock) -> Result<(), CudaError> {
        let mut cache = self.cache.lock().unwrap();
        let device_cache = cache.entry(block.device_id).or_insert_with(HashMap::new);
        let blocks = device_cache.entry(block.size_bytes).or_insert_with(Vec::new);
        blocks.push(block.device_ptr);
        Ok(())
    }

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

lazy_static::lazy_static! {
    pub static ref CUDA_ALLOCATOR: Arc<CudaCachingAllocator> = Arc::new(CudaCachingAllocator::new());
} 