//! Gestion des streams CUDA

// use rustacuda::prelude::*;
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::error::CudaError;

#[derive(Debug)]
pub struct CudaStream {
    stream: Stream,
}

impl CudaStream {
    pub fn new(non_blocking: bool) -> Result<Self, CudaError> {
        let flags = if non_blocking {
            StreamFlags::NON_BLOCKING
        } else {
            StreamFlags::DEFAULT
        };
        let stream = Stream::new(flags, None)?;
        Ok(CudaStream { stream })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.stream.synchronize()
    }

    pub fn inner(&self) -> &Stream {
        &self.stream
    }
} 