//! Exemple d'utilisation des DTypes avancés (I32, I64, Bool) et des opérations associées.
//! Ce script illustre la création de tenseurs, les opérations arithmétiques et logiques, et les fonctions avancées.

use neurarust_core::{DType, StorageDevice, NeuraRustError};
use neurarust_core::tensor::create::{zeros_dtype, ones_dtype, full_dtype_i32, full_dtype_i64, full_dtype_bool, randint, bernoulli_scalar};

fn main() -> Result<(), NeuraRustError> {
    println!("--- Création de tenseurs Integer et Bool ---");
    // Tenseur I32
    let t_i32 = zeros_dtype(&[3, 2], DType::I32)?;
    println!("I32 zeros : {:?}", t_i32);
    let t_i32_ones = ones_dtype(&[2, 2], DType::I32)?;
    println!("I32 ones : {:?}", t_i32_ones);
    let t_i32_full = full_dtype_i32(&[2, 3], 7i32)?;
    println!("I32 full(7) : {:?}", t_i32_full);
    let t_i32_rand = randint(0, 10, vec![4, 2], DType::I32, StorageDevice::CPU)?;
    println!("I32 randint [0,10) : {:?}", t_i32_rand);

    // Tenseur I64
    let t_i64 = zeros_dtype(&[2, 2], DType::I64)?;
    println!("I64 zeros : {:?}", t_i64);
    let t_i64_full = full_dtype_i64(&[2, 2], 42i64)?;
    println!("I64 full(42) : {:?}", t_i64_full);
    let t_i64_rand = randint(5, 15, vec![3, 2], DType::I64, StorageDevice::CPU)?;
    println!("I64 randint [5,15) : {:?}", t_i64_rand);

    // Tenseur Bool
    let t_bool = zeros_dtype(&[2, 3], DType::Bool)?;
    println!("Bool zeros : {:?}", t_bool);
    let t_bool_full = full_dtype_bool(&[2, 2], true)?;
    println!("Bool full(true) : {:?}", t_bool_full);
    let t_bool_bernoulli = bernoulli_scalar(0.3, vec![3, 3], DType::Bool, StorageDevice::CPU)?;
    println!("Bool bernoulli(p=0.3) : {:?}", t_bool_bernoulli);

    Ok(())
} 