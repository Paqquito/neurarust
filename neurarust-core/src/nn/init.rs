// use crate::ops::traits::NeuraNumeric; // Supprimé
use std::sync::Arc; // RwLock supprimé
use crate::buffer::{CpuBuffer, Buffer};
use rand::distributions::{Uniform, Distribution}; // Rng supprimé
use rand_distr::StandardNormal;
use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;
// use crate::device::StorageDevice; // Supprimé
use crate::tensor_data::TensorData;

// TODO: Implement initialization functions (kaiming_uniform_, etc.)

/// Fills the input `Tensor` with the scalar value 0.
///
/// Operates in-place.
///
/// # Arguments
/// * `tensor`: The tensor to fill (mutable reference).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn zeros_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    fill_with_scalar(tensor, 0.0f64)
}

/// Fills the input `Tensor` with the scalar value 1.
///
/// Operates in-place.
///
/// # Arguments
/// * `tensor`: The tensor to fill (mutable reference).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn ones_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    fill_with_scalar(tensor, 1.0f64)
}

/// Fills the input `Tensor` with values according to the Kaiming uniform initialization method.
///
/// Operates in-place. Calculates the bound based on the tensor's fan-in.
///
/// # Arguments
/// * `tensor`: The tensor to initialize (typically a weight tensor).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn kaiming_uniform_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    let fan_in = calculate_fan_in(tensor)?;
    let gain: f64 = (2.0_f64).sqrt(); // gain = sqrt(2) for LeakyReLU with negative_slope=0
    let std = gain / (fan_in as f64).sqrt();
    // Clacul de bound basé sur l'uniforme U(-sqrt(3)*std, sqrt(3)*std)
    let bound = (3.0_f64).sqrt() * std;
    let dist = Uniform::new(-bound, bound); // Distribution Uniforme sur f64
    fill_with_distribution(tensor, &dist) // fill_with_distribution doit gérer le DType
}

/// Fills the input `Tensor` with values according to the Kaiming normal initialization method.
///
/// Operates in-place. Calculates the standard deviation based on the tensor's fan-in.
///
/// # Arguments
/// * `tensor`: The tensor to initialize (typically a weight tensor).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn kaiming_normal_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    let fan_in = calculate_fan_in(tensor)?;
    let gain: f64 = (2.0_f64).sqrt();
    let std = gain / (fan_in as f64).sqrt();
    let dist = StandardNormal; // Distribution Normal(0, 1)
    // Remplir avec N(0, std^2) = std * N(0, 1)
    fill_with_distribution_scaled(tensor, &dist, std)
}

/// Fills the input `Tensor` with values according to the Xavier (Glorot) uniform initialization method.
///
/// Operates in-place. Calculates the bound based on the tensor's fan-in and fan-out.
///
/// # Arguments
/// * `tensor`: The tensor to initialize (typically a weight tensor).
///
/// # Returns
/// A `Result` indicating success or a `NeuraRustError`.
pub fn xavier_uniform_(tensor: &mut Tensor) -> Result<(), NeuraRustError> {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor)?;
    let gain: f64 = 1.0; // gain = 1 for sigmoid, tanh, etc.
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    let bound = (3.0_f64).sqrt() * std;
    let dist = Uniform::new(-bound, bound);
    fill_with_distribution(tensor, &dist)
}

// --- Helper for calculating fan_in/fan_out ---
// Note: This is a simplified version assuming standard weight shapes.
// PyTorch has a more complex _calculate_fan_in_and_fan_out function.
fn calculate_fan_in(tensor: &Tensor) -> Result<usize, NeuraRustError> {
    let shape = tensor.shape();
    if shape.len() < 2 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Fan in calculation requires at least 2 dimensions".to_string(),
        ));
    }
    Ok(shape[1]) // Typical for Linear layers (out_features, in_features)
}

// --- Helper for calculating fan_in/fan_out ---
fn calculate_fan_in_and_fan_out(tensor: &Tensor) -> Result<(usize, usize), NeuraRustError> {
    let shape = tensor.shape();
    if shape.len() < 2 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Fan in/out calculation requires at least 2 dimensions".to_string(),
        ));
    }
    let fan_in = shape[1];
    let fan_out = shape[0];
    Ok((fan_in, fan_out))
}

// --- Helper for filling with a distribution ---

/// Helper function to fill a tensor in-place with values from a distribution.
/// Assumes the Distribution `D` generates values of the correct type `T` matching the tensor buffer.
fn fill_with_distribution<D>(tensor: &mut Tensor, dist: &D) -> Result<(), NeuraRustError>
where
    D: Distribution<f64>, 
{
    if tensor.requires_grad() {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("In-place modification (fill_with_distribution) on a tensor that requires grad is not supported.")
        ));
    }
    let dtype = tensor.dtype();
    let shape = tensor.shape();
    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();

    // Créer le nouveau buffer CPU rempli
    let new_cpu_buffer = match dtype {
        DType::F32 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(dist.sample(&mut rng) as f32);
            }
            CpuBuffer::F32(Arc::new(data_vec))
        }
        DType::F64 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(dist.sample(&mut rng));
            }
            CpuBuffer::F64(Arc::new(data_vec))
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    };

    // Créer le nouveau Buffer (CPU seulement pour l'instant)
    let new_buffer = Buffer::Cpu(new_cpu_buffer);

    // Remplacer l'ancien buffer dans TensorData
    let mut guard = tensor.write_data();
    guard.buffer = Arc::new(new_buffer);
    // TODO: Faut-il aussi mettre à jour offset/strides ? Probablement pas si shape reste identique.
    // Strides devrait être recalculé si nécessaire, mais ici on garde la même shape.
    // Offset devrait être 0 pour un buffer fraîchement créé.
    guard.offset = 0;
    guard.strides = TensorData::calculate_contiguous_strides(&shape); // Recalculer strides contigus

    Ok(())
}

// --- Helper for filling with a scaled distribution ---

/// Helper function to fill a tensor in-place with values from a distribution, scaled by a factor.
/// Assumes the Distribution `D` generates values of type f64 or f32.
fn fill_with_distribution_scaled<D>(
    tensor: &mut Tensor, 
    dist: &D, 
    scale: f64
) -> Result<(), NeuraRustError>
where
    D: Distribution<f64>,
{
    if tensor.requires_grad() {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("In-place modification (fill_with_distribution_scaled) on a tensor that requires grad is not supported.")
        ));
    }
    let dtype = tensor.dtype();
    let shape = tensor.shape();
    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();

    let new_cpu_buffer = match dtype {
        DType::F32 => {
            let _scale_f32 = scale as f32;
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push((dist.sample(&mut rng) * scale) as f32);
            }
            CpuBuffer::F32(Arc::new(data_vec))
        }
        DType::F64 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(dist.sample(&mut rng) * scale);
            }
            CpuBuffer::F64(Arc::new(data_vec))
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    };

    let new_buffer = Buffer::Cpu(new_cpu_buffer);
    let mut guard = tensor.write_data();
    guard.buffer = Arc::new(new_buffer);
    guard.offset = 0;
    guard.strides = TensorData::calculate_contiguous_strides(&shape);

    Ok(())
}

// --- Internal Helper for In-place Fill ---

/// Helper function to fill a tensor in-place with a scalar value.
/// Handles different data types.
fn fill_with_scalar(tensor: &mut Tensor, value: f64) -> Result<(), NeuraRustError> {
    if tensor.requires_grad() {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("In-place modification (fill_with_scalar) on a tensor that requires grad is not supported.")
        ));
    }
    let dtype = tensor.dtype();
    let shape = tensor.shape();
    let numel = shape.iter().product();

    let new_cpu_buffer = match dtype {
        DType::F32 => CpuBuffer::F32(Arc::new(vec![value as f32; numel])),
        DType::F64 => CpuBuffer::F64(Arc::new(vec![value as f64; numel])),
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    };
    
    let new_buffer = Buffer::Cpu(new_cpu_buffer);
    let mut guard = tensor.write_data();
    guard.buffer = Arc::new(new_buffer);
    guard.offset = 0;
    guard.strides = TensorData::calculate_contiguous_strides(&shape);

    Ok(())
}

// --- Tests ---
#[cfg(test)]
#[path = "init_test.rs"]
mod tests; // Link to the test file 