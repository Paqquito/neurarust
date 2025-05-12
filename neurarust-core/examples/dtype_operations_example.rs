//! Exemple d'utilisation des DTypes avancés (I32, I64, Bool) et des opérations associées.
//! Ce script illustre la création de tenseurs, les opérations arithmétiques et logiques, et les fonctions avancées.

use neurarust_core::{DType, StorageDevice, NeuraRustError};
use neurarust_core::tensor::create::{zeros_dtype, ones_dtype, full_dtype_i32, full_dtype_i64, full_dtype_bool, randint, bernoulli_scalar};
use neurarust_core::ops::arithmetic::{add_op, sub_op, mul_op, div_op};

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

    println!("\n--- Opérations arithmétiques sur Integer (I32/I64) ---");
    let a = full_dtype_i32(&[2, 2], 5)?;
    let b = full_dtype_i32(&[2, 2], 3)?;
    let add = add_op(&a, &b)?;
    let sub = sub_op(&a, &b)?;
    let mul = mul_op(&a, &b)?;
    let div = div_op(&a, &b)?;
    println!("5 + 3 = {:?}", add.get_i32_data()?);
    println!("5 - 3 = {:?}", sub.get_i32_data()?);
    println!("5 * 3 = {:?}", mul.get_i32_data()?);
    println!("5 / 3 = {:?}", div.get_i32_data()?);
    assert_eq!(add.get_i32_data()?, vec![8,8,8,8]);
    assert_eq!(sub.get_i32_data()?, vec![2,2,2,2]);
    assert_eq!(mul.get_i32_data()?, vec![15,15,15,15]);
    assert_eq!(div.get_i32_data()?, vec![1,1,1,1]); // division entière

    // Cas limite : division par zéro
    let c = full_dtype_i32(&[2, 2], 0)?;
    let div_zero = div_op(&a, &c);
    assert!(div_zero.is_err());
    println!("Division par zéro (attendu: erreur): {:?}", div_zero);

    // I64
    let a64 = full_dtype_i64(&[2, 2], 10)?;
    let b64 = full_dtype_i64(&[2, 2], 4)?;
    let add64 = add_op(&a64, &b64)?;
    let mul64 = mul_op(&a64, &b64)?;
    println!("10 + 4 (I64) = {:?}", add64.get_i64_data()?);
    println!("10 * 4 (I64) = {:?}", mul64.get_i64_data()?);
    assert_eq!(add64.get_i64_data()?, vec![14,14,14,14]);
    assert_eq!(mul64.get_i64_data()?, vec![40,40,40,40]);

    println!("\n--- Opérations logiques sur Bool ---");
    let t_true = full_dtype_bool(&[2, 2], true)?;
    let t_false = full_dtype_bool(&[2, 2], false)?;
    let and = t_true.logical_and(&t_false)?;
    let or = t_true.logical_or(&t_false)?;
    let xor = t_true.logical_xor(&t_false)?;
    let not = t_true.logical_not()?;
    println!("true AND false = {:?}", and.get_bool_data()?);
    println!("true OR false = {:?}", or.get_bool_data()?);
    println!("true XOR false = {:?}", xor.get_bool_data()?);
    println!("NOT true = {:?}", not.get_bool_data()?);
    assert_eq!(and.get_bool_data()?, vec![false, false, false, false]);
    assert_eq!(or.get_bool_data()?, vec![true, true, true, true]);
    assert_eq!(xor.get_bool_data()?, vec![true, true, true, true]);
    assert_eq!(not.get_bool_data()?, vec![false, false, false, false]);

    Ok(())
} 