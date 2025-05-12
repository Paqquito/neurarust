//! Backend CUDA pour NeuraRust : gestion du contexte et des devices CUDA
//!
//! Fournit les fonctions d'initialisation, d'énumération et d'interrogation des devices CUDA.

use rustacuda::prelude::*;
use rustacuda::device::Device;
use rustacuda::error::CudaError;

/// Initialise le contexte CUDA (doit être appelé avant toute opération CUDA)
pub fn init_cuda() -> Result<(), CudaError> {
    rustacuda::init(CudaFlags::empty())
}

// D'autres fonctions viendront ici (énumération, propriétés, etc.)

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
