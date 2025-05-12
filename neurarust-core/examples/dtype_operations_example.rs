//! Exemple d'utilisation des DTypes avancés (I32, I64, Bool) et des opérations associées.
//! Ce script illustre la création de tenseurs, les opérations arithmétiques et logiques, et les fonctions avancées.

use neurarust_core::{DType, StorageDevice, NeuraRustError};
use neurarust_core::tensor::create::{zeros_dtype, ones_dtype, full_dtype_i32, full_dtype_i64, full_dtype_bool, randint, bernoulli_scalar};
use neurarust_core::ops::arithmetic::{add_op, sub_op, mul_op, div_op};
use neurarust_core::tensor::Tensor;

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

    // --- Opérations avancées : index_select, masked_select, masked_fill_ ---
    println!("\n--- Opérations avancées sur Integer (I32) ---");
    let t = randint(0, 10, vec![4, 4], DType::I32, StorageDevice::CPU)?;
    println!("Tenseur de base : {:?}", t.get_i32_data()?);

    // Exemple : sélectionner la 2e ligne d'un tenseur I32
    let indices = full_dtype_i64(&[1], 1)?; // Indice 1 (2e ligne)
    match t.index_select(0, &indices) {
        Ok(selected) => println!("Sélection de la 2e ligne : {:?}", selected),
        Err(e) => println!("Erreur lors de l'index_select (2e ligne) : {:?}", e),
    }

    // Masque des éléments pairs (I32)
    let t = match full_dtype_i32(&[4], 1) {
        Ok(t) => t,
        Err(e) => { println!("Erreur lors de la création du tenseur t : {:?}", e); return Err(e); }
    };
    let incr = match full_dtype_i32(&[4], 0) {
        Ok(i) => i,
        Err(e) => { println!("Erreur lors de la création du tenseur incr : {:?}", e); return Err(e); }
    };
    let mut incr_data = match incr.get_i32_data() {
        Ok(d) => d,
        Err(e) => { println!("Erreur lors de l'accès aux données incr : {:?}", e); return Err(e); }
    };
    for (i, v) in incr_data.iter_mut().enumerate() { *v = i as i32; }
    let incr = match Tensor::new_i32(incr_data, vec![4]) {
        Ok(t) => t,
        Err(e) => { println!("Erreur lors de la création du tenseur incr final : {:?}", e); return Err(e); }
    };
    let t = match add_op(&t, &incr) {
        Ok(t) => t,
        Err(e) => { println!("Erreur lors de l'addition pour t : {:?}", e); return Err(e); }
    }; // t = [1,2,3,4]
    // Création du masque booléen des éléments pairs
    let t_data = match t.get_i32_data() {
        Ok(d) => d,
        Err(e) => { println!("Erreur lors de l'accès aux données t : {:?}", e); return Err(e); }
    };
    let mask_data: Vec<bool> = t_data.iter().map(|x| x % 2 == 0).collect();
    let mask = match Tensor::new_bool(mask_data, vec![4]) {
        Ok(m) => m,
        Err(e) => { println!("Erreur lors de la création du masque : {:?}", e); return Err(e); }
    };
    println!("Masque des éléments pairs : {:?}", mask);

    // Sélection par masque
    match t.masked_select(&mask) {
        Ok(masked) => println!("Éléments pairs sélectionnés : {:?}", masked),
        Err(e) => println!("Erreur lors du masked_select (pairs) : {:?}", e),
    }

    // Gestion d'erreur : index hors bornes
    let indices_out = full_dtype_i64(&[1], 10)?;
    let res = t.index_select(0, &indices_out);
    assert!(res.is_err());
    println!("index_select avec indice hors borne : {:?}", res);

    // Gestion d'erreur : masque de mauvaise forme
    let bad_mask = zeros_dtype(&[2, 2], DType::Bool)?;
    let res = t.masked_select(&bad_mask);
    assert!(res.is_err());
    println!("masked_select avec masque de mauvaise forme : {:?}", res);

    Ok(())
} 