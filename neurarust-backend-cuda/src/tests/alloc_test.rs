//! Tests unitaires pour alloc.rs

use super::super::alloc::*;
use super::super::context::*;
use std::sync::Arc;
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