//! Gestion du contexte et des devices CUDA

use rustacuda::prelude::*;
use rustacuda::device::Device;
use rustacuda::error::CudaError;
use rustacuda::device::DeviceAttribute;
use rustacuda::context::{Context, ContextFlags};

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
}

pub fn init_cuda() -> Result<(), CudaError> {
    rustacuda::init(CudaFlags::empty())
}

pub fn device_count() -> Result<usize, CudaError> {
    Device::num_devices().map(|n| n as usize)
}

pub fn get_device(index: usize) -> Result<Device, CudaError> {
    Device::get_device(index as u32)
}

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

pub fn create_cuda_context(device_index: usize) -> Result<Context, CudaError> {
    let device = get_device(device_index)?;
    Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
} 