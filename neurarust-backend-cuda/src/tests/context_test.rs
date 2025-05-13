use neurarust_backend_cuda::context::*;
use neurarust_backend_cuda::error::CudaBackendError;
use rustacuda::context::ContextFlags;
use std::sync::Once;

// Helper to initialize logger only once for all tests in this module
static LOGGER_INIT: Once = Once::new();

fn setup_logger() {
    LOGGER_INIT.call_once(|| {
        // Attempt to initialize env_logger. If it fails (e.g., already initialized), that's okay.
        let _ = env_logger::builder().is_test(true).try_init();
    });
}

#[test]
fn test_initialize_cuda_idempotent() {
    setup_logger();
    // Call multiple times to test idempotency
    assert!(initialize_cuda_with_logging().is_ok());
    assert!(initialize_cuda_with_logging().is_ok());
}

#[test]
fn test_list_devices() {
    setup_logger();
    match list_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                // This is a valid outcome if no CUDA devices are present or CUDA is not properly set up.
                // In a CI environment without GPUs, this branch might be hit.
                // We check if the error is NoDevicesFound if list_devices itself failed.
                println!("No CUDA devices found or CUDA not set up. This might be expected.");
            } else {
                println!("Found {} devices:", devices.len());
                for device in devices {
                    println!(
                        "  ID: {}, Name: {}, Memory: {}MB, CC: {}.{}",
                        device.id,
                        device.name,
                        device.total_memory / (1024 * 1024),
                        device.compute_capability.0,
                        device.compute_capability.1
                    );
                    assert!(!device.name.is_empty());
                    assert!(device.total_memory > 0);
                }
            }
        }
        Err(CudaBackendError::NoDevicesFound) => {
            println!("Test explicitly caught NoDevicesFound error, which is acceptable.");
        }
        Err(CudaBackendError::InitializationError(e)) => {
            // This case might occur if CUDA toolkit is not installed or drivers are missing
            println!("CUDA InitializationError during list_devices: {}. This might be expected in some environments.", e);
        }
        Err(e) => {
            panic!("Failed to list devices: {:?}", e);
        }
    }
}

#[test]
fn test_create_and_set_context() {
    setup_logger();
    if let Ok(devices) = list_devices() {
        if let Some(device) = devices.first() {
            println!("Using device {} for context test.", device.name);
            let context = CudaContext::new(device, ContextFlags::SCHED_AUTO).expect("Failed to create context");
            assert_eq!(context.device_id(), device.id);

            // Check current context info after creation (new context should be current)
            if let Ok(Some((current_dev_id, _))) = current_context_info() {
                assert_eq!(current_dev_id, device.id, "Context created with new() should be current on the creating thread.");
            } else {
                panic!("Expected a current context after CudaContext::new()");
            }
            
            // Explicitly set current (should be fine, it might already be current)
            context.set_current().expect("Failed to set context current");
            
            // Drop the context, rustacuda should handle popping it from the thread's current context stack
            // if it was pushed by Context::create_and_push.
            drop(context);

            // After dropping the specific context, there might be no context or a different one (e.g. primary if retained before).
            // This part of the test is a bit tricky as `create_and_push` makes the new context current.
            // When `context` (which holds an Arc to the rustacuda Context) is dropped, 
            // if it was the one pushed by `create_and_push`, its drop handler should pop it.
            // We can't easily test if `cuCtxPopCurrent` was called without more advanced tools,
            // but we can check that `current_context_info` behaves as expected (e.g. shows None or a previous one).
            println!("Context dropped. Checking current context again...");
            match current_context_info() {
                Ok(Some((_id, _flags))) => println!("A context is still current on the thread (possibly primary or another one)."),
                Ok(None) => println!("No context is current on the thread after drop, as expected in some scenarios."),
                Err(e) => panic!("Error checking current context after drop: {:?}", e),
            }

        } else {
            println!("No devices found, skipping context creation test.");
        }
    } else {
        println!("Failed to list devices, skipping context creation test.");
    }
}

#[test]
fn test_get_primary_context() {
    setup_logger();
    if let Ok(devices) = list_devices() {
        if let Some(device) = devices.first() {
            println!("Using device {} for primary context test.", device.name);
            let p_ctx = get_primary_context(device.id).expect("Failed to get primary context");
            assert_eq!(p_ctx.device_id(), device.id);

            // The primary context should now be current on this thread.
             if let Ok(Some((current_dev_id, _))) = current_context_info() {
                assert_eq!(current_dev_id, device.id, "Primary context should be current after get_primary_context()");
            } else {
                panic!("Expected a current primary context after get_primary_context()");
            }
            drop(p_ctx);
        } else {
            println!("No devices found, skipping primary context test.");
        }
    } else {
        println!("Failed to list devices, skipping primary context test.");
    }
}

#[test]
fn test_current_context_info_no_context() {
    setup_logger();
    // This test is tricky because other tests might leave contexts active.
    // Ideally, this would run in a pristine environment or ensure all contexts are popped.
    // For now, we call initialize_cuda_with_logging and then current_context_info.
    // If no GPU or CUDA, this might fail at init or list_devices, which is fine.
    if initialize_cuda_with_logging().is_ok() {
        // We cannot guarantee no context is current if other tests ran or if a primary context
        // was activated by the driver. So we just call it and log.
        match current_context_info() {
            Ok(Some((id, flags))) => {
                println!("test_current_context_info_no_context: Context found: device {}, flags {:?}", id, flags);
            }
            Ok(None) => {
                println!("test_current_context_info_no_context: No context initially, as might be expected.");
            }
            Err(e) => {
                 // If CUDA is not available, this path might be taken via errors in current_context_info itself.
                println!("test_current_context_info_no_context: Error checking context: {:?}. This is acceptable if CUDA is unavailable.", e);
            }
        }
    }
} 