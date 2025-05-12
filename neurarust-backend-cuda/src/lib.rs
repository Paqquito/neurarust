//! Backend CUDA pour NeuraRust : point d'entrée du module
//!
//! Structure des modules :
//! - context.rs : gestion du contexte et des devices CUDA
//! - stream.rs  : gestion des streams CUDA
//! - event.rs   : gestion des événements CUDA
//! - alloc.rs   : allocateur mémoire et cache
//! - tests/     : tests unitaires par fonctionnalité

pub mod context;
pub mod stream;
pub mod event;
pub mod alloc;

#[cfg(test)]
mod tests;
