use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;

/// Compte la fréquence des valeurs dans un tenseur d'entiers (I32/I64).
pub fn bincount_op(input: &Tensor, weights: Option<&Tensor>, minlength: usize) -> Result<Tensor, NeuraRustError> {
    // Vérifier que input est 1D, dtype I32 ou I64, non-négatif
    let shape = input.shape();
    if shape.len() != 1 {
        return Err(NeuraRustError::ShapeMismatch {
            expected: format!("[1] (1D tensor)"),
            actual: format!("{:?}", shape),
            operation: "bincount_op".to_string(),
        });
    }
    let dtype = input.dtype();
    let _n = shape[0];
    if dtype == DType::I32 {
        let values = input.get_i32_data()?;
        if let Some(w) = weights {
            let w_data = w.get_f32_data()?;
            if w_data.len() != values.len() {
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("weights.len() == input.len() == {}", values.len()),
                    actual: format!("weights.len() == {}", w_data.len()),
                    operation: "bincount_op".to_string(),
                });
            }
            let maxval = values.iter().copied().max().unwrap_or(0).max(minlength as i32 - 1);
            if values.iter().any(|&v| v < 0) {
                return Err(NeuraRustError::UnsupportedOperation("bincount: valeurs négatives non supportées".to_string()));
            }
            let mut out = vec![0f32; (maxval as usize + 1).max(minlength)];
            for (i, &v) in values.iter().enumerate() {
                out[v as usize] += w_data[i];
            }
            let len = out.len();
            Tensor::new(out, vec![len])
        } else {
            let maxval = values.iter().copied().max().unwrap_or(0).max(minlength as i32 - 1);
            if values.iter().any(|&v| v < 0) {
                return Err(NeuraRustError::UnsupportedOperation("bincount: valeurs négatives non supportées".to_string()));
            }
            let mut out = vec![0i32; (maxval as usize + 1).max(minlength)];
            for &v in &values {
                out[v as usize] += 1;
            }
            let len = out.len();
            Tensor::new_i32(out, vec![len])
        }
    } else if dtype == DType::I64 {
        let values = input.get_i64_data()?;
        if let Some(w) = weights {
            let w_data = w.get_f64_data()?;
            if w_data.len() != values.len() {
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("weights.len() == input.len() == {}", values.len()),
                    actual: format!("weights.len() == {}", w_data.len()),
                    operation: "bincount_op".to_string(),
                });
            }
            let maxval = values.iter().copied().max().unwrap_or(0).max(minlength as i64 - 1);
            if values.iter().any(|&v| v < 0) {
                return Err(NeuraRustError::UnsupportedOperation("bincount: valeurs négatives non supportées".to_string()));
            }
            let mut out = vec![0f64; (maxval as usize + 1).max(minlength)];
            for (i, &v) in values.iter().enumerate() {
                out[v as usize] += w_data[i];
            }
            let len = out.len();
            Tensor::new_f64(out, vec![len])
        } else {
            let maxval = values.iter().copied().max().unwrap_or(0).max(minlength as i64 - 1);
            if values.iter().any(|&v| v < 0) {
                return Err(NeuraRustError::UnsupportedOperation("bincount: valeurs négatives non supportées".to_string()));
            }
            let mut out = vec![0i64; (maxval as usize + 1).max(minlength)];
            for &v in &values {
                out[v as usize] += 1;
            }
            let len = out.len();
            Tensor::new_i64(out, vec![len])
        }
    } else {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::I32,
            actual: dtype,
            operation: "bincount_op".to_string(),
        });
    }
} 