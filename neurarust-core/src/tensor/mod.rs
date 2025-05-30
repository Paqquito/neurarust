#![allow(clippy::needless_borrow)]
// src/tensor/mod.rs

//!
//! # Tensor Module
//! 
//! This module defines the core `Tensor` structure, the primary data structure
//! for representing multi-dimensional arrays in NeuraRust.
//! 
//! It also declares submodules containing tensor creation functions (`create`),
//! utility functions (`utils`, `broadcast_utils`, `iter_utils`), and implementations
//! for various tensor functionalities (methods for autograd, accessors, reductions, views, traits).

use crate::tensor_data::TensorData;
use std::sync::{Arc, RwLock};
use crate::error::NeuraRustError;

// --- Define existing implementation modules ---
// Documentation for methods within these modules should be added in the respective files.
mod autograd_methods;
mod traits; // Keep traits module declared
mod accessors; // Ensure accessors module is declared
mod reduction_methods; // Declare the new module
mod view_methods; // Declare view methods
mod inplace_arithmetic_methods; // Added inplace_arithmetic_methods module
pub mod inplace_ops; // Declare the new module for inplace operations logic

#[cfg(test)]
mod inplace_ops_tests;

pub mod create; // Make the create module public

// --- Declare utility modules ---
pub mod utils; // Declare public utils
pub mod broadcast_utils; // Declare the new public broadcast utils module
pub mod iter_utils; // Declare the new iterator utils module

// Re-export creation functions to make them public and easily accessible
// e.g., `neurarust_core::zeros(...)` instead of `neurarust_core::tensor::create::zeros(...)`
pub use create::{zeros, ones, full, zeros_like, ones_like,
                 zeros_f64, ones_f64, full_f64,
                 from_vec_f32, from_vec_f64,
                 from_vec_i32, from_vec_i64, from_vec_bool};

/// Represents a multi-dimensional array (tensor) in NeuraRust.
///
/// `Tensor` acts as a lightweight handle to the underlying tensor data and metadata,
/// stored in `TensorData`. It uses `Arc<RwLock<TensorData>>` internally to enable:
/// 
/// 1.  **Shared Ownership & Cheap Clones:** Multiple `Tensor` instances can point to the same
///     `TensorData` without copying the potentially large data buffer. Cloning a `Tensor`
///     is inexpensive as it only clones the `Arc`.
/// 2.  **Interior Mutability:** Metadata within `TensorData` (like `requires_grad` or `grad`)
///     can be modified safely through an immutable `Tensor` reference (`&Tensor`) via the `RwLock`.
///     This is essential for autograd operations.
///
/// Tensors are type-aware via the `DType` enum stored in `TensorData` and manage their
/// data storage through the `Buffer` enum, abstracting over CPU and future GPU backends.
///
/// Most tensor operations are implemented as methods or associated functions, organized into
/// submodules like `ops`, `autograd_methods`, `view_methods`, etc.
/// Creation functions are available in the `create` submodule and re-exported here.
pub struct Tensor {
    /// Arc for shared ownership, RwLock for interior mutability of TensorData.
    /// Marked `pub(crate)` to allow internal modules direct access if needed,
    /// but external users interact via methods.
    pub(crate) data: Arc<RwLock<TensorData>>,
}

// --- Test Module ---
// Tests are now in separate files linked via `#[path = ...]` in respective modules.

// --- Re-exports ---
// pub use tensor_data::TensorData; // Commented out
// pub use storage::{Buffer, Storage, StorageDevice}; // Commented out
// pub use create::{zeros, ones, full}; // Commented out

// pub use methods::{Tensor, TensorMethods}; // Commented out

// pub use crate::error::NeuraRustError; // Commented out

// --- Tensor Implementation --- 
impl Tensor {
    /// Creates a new CPU tensor with the given `f32` data and shape.
    ///
    /// This is a primary constructor. It takes ownership of the data vector,
    /// calculates contiguous strides automatically, and initializes the tensor
    /// on the CPU with `requires_grad` set to `false`.
    ///
    /// # Example
    /// ```
    /// use neurarust_core::{Tensor, NeuraRustError};
    /// 
    /// fn main() -> Result<(), NeuraRustError> {
    ///     let data = vec![1.0, 2.0, 3.0, 4.0];
    ///     let shape = vec![2, 2];
    ///     let t = Tensor::new(data, shape)?;
    ///     assert_eq!(t.shape(), &[2, 2]);
    ///     assert_eq!(t.dtype(), neurarust_core::DType::F32);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Arguments
    /// * `data_vec`: A `Vec<f32>` containing the tensor data in row-major order.
    /// * `shape`: A `Vec<usize>` defining the tensor dimensions.
    ///
    /// # Errors
    /// Returns `NeuraRustError::TensorCreationError` if the number of elements in `data_vec`
    /// does not match the total number of elements specified by `shape`.
    pub fn new(data_vec: Vec<f32>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Creates a new CPU tensor with the given `f64` data and shape.
    ///
    /// Similar to `Tensor::new`, but creates a tensor with `DType::F64`.
    ///
    /// # Arguments
    /// * `data_vec`: A `Vec<f64>` containing the tensor data.
    /// * `shape`: A `Vec<usize>` defining the tensor dimensions.
    ///
    /// # Errors
    /// Returns `NeuraRustError::TensorCreationError` if data length mismatches shape numel.
    pub fn new_f64(data_vec: Vec<f64>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new_f64(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Crée un tenseur CPU avec les données i32 et la forme données.
    pub fn new_i32(data_vec: Vec<i32>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new_i32(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Crée un tenseur CPU avec les données i64 et la forme données.
    pub fn new_i64(data_vec: Vec<i64>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new_i64(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Crée un tenseur CPU avec les données bool et la forme données.
    pub fn new_bool(data_vec: Vec<bool>, shape: Vec<usize>) -> Result<Self, NeuraRustError> {
        let tensor_data = TensorData::new_bool(data_vec, shape)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Teste l'égalité élément par élément avec un autre tenseur. Retourne un tenseur Bool.
    pub fn eq(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::equal_op(self, other)
    }

    /// Teste la différence élément par élément avec un autre tenseur. Retourne un tenseur Bool.
    pub fn ne(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::ne_op(self, other)
    }

    /// Teste l'infériorité stricte élément par élément avec un autre tenseur. Retourne un tenseur Bool.
    pub fn lt(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::lt_op(self, other)
    }

    /// Teste la supériorité stricte élément par élément avec un autre tenseur. Retourne un tenseur Bool.
    pub fn gt(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::gt_op(self, other)
    }

    /// Teste l'infériorité ou égalité élément par élément avec un autre tenseur. Retourne un tenseur Bool.
    pub fn le(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::le_op(self, other)
    }

    /// Teste la supériorité ou égalité élément par élément avec un autre tenseur. Retourne un tenseur Bool.
    pub fn ge(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::ge_op(self, other)
    }

    /// Effectue un ET logique élément par élément avec un autre tenseur Bool. Retourne un tenseur Bool.
    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::logical_and_op(self, other)
    }

    /// Effectue un OU logique élément par élément avec un autre tenseur Bool. Retourne un tenseur Bool.
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::logical_or_op(self, other)
    }

    /// Effectue un XOR logique élément par élément avec un autre tenseur Bool. Retourne un tenseur Bool.
    pub fn logical_xor(&self, other: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::logical_xor_op(self, other)
    }

    /// Effectue un NON logique élément par élément sur ce tenseur Bool. Retourne un tenseur Bool.
    pub fn logical_not(&self) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::logical_not_op(self)
    }

    /// Renvoie un tenseur où chaque élément provient de `self` si `condition` est vrai, sinon de `y` (broadcast supporté).
    pub fn where_cond(&self, condition: &Tensor, y: &Tensor) -> Result<Tensor, NeuraRustError> {
        crate::ops::comparison::where_op(condition, self, y)
    }

    /// Compte la fréquence des valeurs dans un tenseur d'entiers (I32/I64).
    pub fn bincount(&self, weights: Option<&Tensor>, minlength: usize) -> Result<Tensor, crate::error::NeuraRustError> {
        crate::ops::reduction::bincount_op(self, weights, minlength)
    }
}