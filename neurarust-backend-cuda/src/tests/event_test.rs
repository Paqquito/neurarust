//! Tests unitaires pour event.rs

use super::super::event::*;
use super::super::stream::*;
use super::super::context::*;
use std::time::Duration;
use std::thread::sleep;

#[test]
fn test_cuda_event_create_and_sync() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        let stream = CudaStream::new(false).expect("Stream creation failed");
        let event = CudaEvent::new(true).expect("Event creation failed");
        event.record(&stream).expect("Event record failed");
        stream.synchronize().expect("Stream sync failed");
        event.synchronize().expect("Event sync failed");
        assert!(event.is_completed().unwrap());
    }
}

#[test]
fn test_cuda_event_timing() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        let stream = CudaStream::new(false).expect("Stream creation failed");
        let start = CudaEvent::new(true).expect("Start event failed");
        let end = CudaEvent::new(true).expect("End event failed");
        start.record(&stream).expect("Start record failed");
        sleep(Duration::from_millis(10));
        end.record(&stream).expect("End record failed");
        stream.synchronize().expect("Stream sync failed");
        let elapsed = CudaEvent::elapsed_time_ms(&start, &end).expect("Elapsed time failed");
        println!("[TEST] Temps mesuré entre deux événements : {} ms", elapsed);
    }
}

#[test]
fn test_massive_event_creation_and_destruction() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        let mut events = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let e = CudaEvent::new(true).expect("Event creation failed");
            events.push(e);
        }
        drop(events);
    }
}

#[test]
fn test_cross_stream_synchronization() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        let stream1 = CudaStream::new(false).expect("Stream1 creation failed");
        let stream2 = CudaStream::new(false).expect("Stream2 creation failed");
        let event = CudaEvent::new(true).expect("Event creation failed");
        event.record(&stream1).expect("Event record failed");
        stream1.synchronize().expect("Stream1 sync failed");
        event.synchronize().expect("Event sync failed");
        stream2.synchronize().expect("Stream2 sync failed");
    }
}

#[test]
fn test_event_profiling_hook() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
        let stream = CudaStream::new(false).expect("Stream creation failed");
        let start = CudaEvent::new(true).expect("Start event failed");
        let end = CudaEvent::new(true).expect("End event failed");
        start.record(&stream).expect("Start record failed");
        std::thread::sleep(Duration::from_millis(5));
        end.record(&stream).expect("End record failed");
        stream.synchronize().expect("Stream sync failed");
        let elapsed = CudaEvent::elapsed_time_ms(&start, &end).expect("Elapsed time failed");
        println!("[PROFILER] Opération GPU simulée : {} ms", elapsed);
    }
} 