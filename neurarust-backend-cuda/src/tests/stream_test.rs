//! Tests unitaires pour stream.rs

use super::super::stream::*;
use super::super::context::*;

#[test]
fn test_cuda_stream_create_and_sync() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        let s = CudaStream::new(false).expect("Stream creation failed");
        s.synchronize().expect("Stream synchronize failed");
        let s2 = CudaStream::new(true).expect("Non-blocking stream creation failed");
        s2.synchronize().expect("Non-blocking stream synchronize failed");
    }
} 