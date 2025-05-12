//! Tests unitaires pour context.rs

use super::super::context::*;

#[test]
fn test_cuda_init_and_device_count() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    println!("CUDA device count: {}", count);
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
fn test_create_cuda_context() {
    init_cuda().expect("CUDA init failed");
    let count = device_count().expect("device_count failed");
    if count > 0 {
        let _ctx = create_cuda_context(0).expect("Context creation failed");
    }
} 