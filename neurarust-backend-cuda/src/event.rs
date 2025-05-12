//! Gestion des événements CUDA

// use rustacuda::prelude::*;
use rustacuda::event::{Event, EventFlags};
use rustacuda::error::CudaError;
use crate::stream::CudaStream;

#[derive(Debug)]
pub struct CudaEvent {
    event: Event,
}

impl CudaEvent {
    pub fn new(enable_timing: bool) -> Result<Self, CudaError> {
        let flags = if enable_timing { EventFlags::DEFAULT } else { EventFlags::DISABLE_TIMING };
        let event = Event::new(flags)?;
        Ok(CudaEvent { event })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.event.record(&stream.inner())
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.event.synchronize()
    }

    pub fn is_completed(&self) -> Result<bool, CudaError> {
        match self.event.query() {
            Ok(rustacuda::event::EventStatus::Ready) => Ok(true),
            Ok(rustacuda::event::EventStatus::NotReady) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn elapsed_time_ms(start: &CudaEvent, end: &CudaEvent) -> Result<f32, CudaError> {
        Event::elapsed_time_f32(&start.event, &end.event)
    }
} 