use crate::autograd::graph::NodeId;
use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::device::StorageDevice;
use crate::ops::reduction::sum::sum_kernel;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLock, RwLockReadGuard};
use crate::types::DType;
use crate::ops::arithmetic::mul_op;
use crate::ops::view::expand_op;
use super::utils::{calculate_reduction_output_shape, process_reduction_axes};
// use crate::autograd::grad_check::check_grad; // Keep commented until check_grad is ready

// --- MeanBackward Definition ---

/// Backward operation context for `mean` reduction.
///
/// Stores information needed to compute the gradient of the mean operation:
/// - A reference to the original input tensor's data (`input_node`).
/// - The total number of elements that were reduced (`num_elements_reduced`) to compute the mean.
#[derive(Debug)]
struct MeanBackward {
    input_node: Arc<RwLock<TensorData>>,
    num_elements_reduced: usize,
}

// --- BackwardOp Implementation for MeanBackward ---

impl BackwardOp for MeanBackward {
    /// Computes the gradient for the mean reduction operation.
    ///
    /// The gradient of the mean operation w.r.t. its input is the incoming gradient (`grad_output`)
    /// divided by the number of elements that were reduced (`N`), and then broadcasted back
    /// to the original input shape.
    ///
    /// Gradient = `expand(grad_output / N, original_input_shape)`
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing back from the subsequent operation,
    ///   corresponding to the output of the original mean operation.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Tensor>` with a single element: the gradient
    /// with respect to the original input tensor. Returns an error if expansion or
    /// device operations fail.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // Get original input shape
        let input_guard = self.input_node.read().map_err(|_| {
            // Correct LockError format
            NeuraRustError::LockError{
                lock_type: "read".to_string(),
                reason: "Failed to lock input node in MeanBackward".to_string()
            }
        })?;
        let input_shape = input_guard.shape.clone();
        // Ensure we drop the read guard before potential mutable operations later
        drop(input_guard);

        // Calculate scaling factor
        let n = self.num_elements_reduced;
        if n == 0 {
            // If N is zero, the forward output was likely empty or NaN/inf.
            // The gradient is ill-defined or should be zero. Let's return zero gradient
            // with the correct input shape.
            // TODO: Consider if returning an error or specific handling is better.
             // Assume F32 for now - Correct zeros call
            return Ok(vec![crate::tensor::zeros(&input_shape)?]);
        }
        let scale: f32 = 1.0 / (n as f32);

        // Create scalar tensor for scale factor (assuming F32, CPU)
        let scale_tensor = Tensor::new(vec![scale], vec![])?; // Scalar tensor

        // Multiply grad_output by scale: scaled_grad = grad_output * (1 / N)
        // mul_op handles broadcasting the scalar scale_tensor
        let scaled_grad = mul_op(grad_output, &scale_tensor)?;

        // Expand scaled_grad back to the original input shape
        // expand_op handles the view creation
        let grad_input = expand_op(&scaled_grad, input_shape)?;

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)] // Return NodeId from Arc
    }
}

// --- Kernel de Calcul (F32 CPU) ---

/// Noyau de calcul privé pour la moyenne avec réduction d'axes. F32 CPU.
fn mean_kernel(
    input_guard: &RwLockReadGuard<'_, TensorData>,
    input_data_slice: &[f32],
    axes: &[usize],
    keep_dims: bool,
    output_shape: &[usize],
    n: f32,                   // Use f32 for divisor
) -> Result<Vec<f32>, NeuraRustError>
{
    // 1. Calculer la somme en utilisant le kernel de sum (expects f32)
    let sum_data = sum_kernel(input_guard, input_data_slice, axes, keep_dims, output_shape)?;

    // 2. Diviser chaque élément par N
    if n == 0.0f32 {
        return Err(NeuraRustError::DivisionByZero);
    }

    // Division f32 / f32
    let mean_data: Vec<f32> = sum_data.into_iter().map(|val| val / n).collect();

    Ok(mean_data)
}

// --- mean_op Implementation (Internal API called by Tensor::mean) ---

/// Computes the mean of a tensor along specified axes.
///
/// This is the core implementation called by `Tensor::mean`.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `axes` - An optional slice of `usize` specifying the axes to reduce.
///            If `None`, reduces over all axes.
/// * `keep_dims` - Whether to keep the reduced dimensions (size 1) in the output shape.
///
/// # Returns
///
/// A `Result` containing the tensor with the mean values. Returns an error if
/// axes are invalid, device/dtype is unsupported, or autograd setup fails.
pub(crate) fn mean_op(
    tensor: &Tensor,
    axes: Option<&[usize]>,
    keep_dims: bool,
) -> Result<Tensor, NeuraRustError> {
    let t_guard = tensor.read_data();

    // --- Device Check --- (Garder générique pour l'instant)
    if t_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "mean_op".to_string(),
            expected: StorageDevice::CPU,
            actual: t_guard.device,
        });
    }

    // --- DType Check --- (Garder générique pour l'instant)
    let dtype = t_guard.dtype;
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Mean operation only supports F32 and F64, got {:?}",
            dtype
        )));
    }

    // --- Process Axes --- (Utiliser l'utilitaire)
    let rank = t_guard.shape.len();
    let axes_vec = process_reduction_axes(rank, axes)?;

    // --- Calculate Output Shape --- (Utiliser l'utilitaire)
    let output_shape = calculate_reduction_output_shape(&t_guard.shape, &axes_vec, keep_dims);

    // --- Calculate N (Number of elements reduced) --- (Simplifié)
    let n: usize = if axes_vec.is_empty() {
        t_guard.numel() // Reduce all elements
    } else {
        // Product of the sizes of the dimensions being reduced
        axes_vec.iter().map(|&axis| t_guard.shape[axis]).product()
    };

    if n == 0 {
        // Utiliser UnsupportedOperation car la moyenne de zéro élément n'est pas définie
        return Err(NeuraRustError::UnsupportedOperation(
            "Cannot compute mean over zero elements (dimension size might be 0)".to_string(),
        ));
    }

    // --- Autograd Info ---
    let requires_grad = t_guard.requires_grad;
    // Clone Arc pour BackwardOp si nécessaire
    let input_node_arc = if requires_grad { Some(tensor.data.clone()) } else { None };

    // --- Dispatch Kernel based on DType ---
    let output_tensor = match dtype {
        DType::F32 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f32()?.as_slice();
            // Convertir n en f32 pour le kernel
            let n_f32 = n as f32;
            // Vérifier la conversion usize -> f32
            if n_f32 == 0.0 && n > 0 {
                return Err(NeuraRustError::InternalError(
                    "Element count N is too large to represent accurately as f32 for mean".to_string()
                 ));
            }
            let result_data = mean_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec,
                keep_dims,
                &output_shape,
                n_f32, // Passer n as f32
            )?;
            drop(t_guard);
            Tensor::new(result_data, output_shape)?
        }
        DType::F64 => {
             let input_data_slice = t_guard.buffer().try_get_cpu_f64()?.as_slice();
            // Convertir n en f64 pour le kernel
            let n_f64 = n as f64;
             // Pas de vérification de perte de précision pour f64 ici (moins probable)
            let result_data = mean_kernel_f64( // Appelera une version f64 du kernel (à créer)
                &t_guard,
                input_data_slice,
                &axes_vec,
                keep_dims,
                &output_shape,
                n_f64, // Passer n as f64
            )?;
            drop(t_guard);
            Tensor::new_f64(result_data, output_shape)?
        }
    };

    // --- Autograd Setup ---
    if requires_grad {
        if let Some(node_arc) = input_node_arc {
            let grad_fn = MeanBackward {
                input_node: node_arc,
                num_elements_reduced: n, // Passer le n calculé (usize)
            };
            let mut output_guard = output_tensor.write_data();
            output_guard.requires_grad = true;
            output_guard.grad_fn = Some(Arc::new(grad_fn));
        } else {
            return Err(NeuraRustError::InternalError(
                "Mean op requires grad but input Arc Node unavailable".to_string(),
            ));
        }
    }

    Ok(output_tensor)
}

// TODO: Implémenter mean_kernel_f64
fn mean_kernel_f64(
    input_guard: &RwLockReadGuard<'_, TensorData>,
    input_data_slice: &[f64],
    axes: &[usize],
    keep_dims: bool,
    output_shape: &[usize],
    n: f64,                   // Utiliser f64 pour diviseur
) -> Result<Vec<f64>, NeuraRustError> {
    // 1. Calculer la somme en utilisant le kernel de sum<f64>
    let sum_data = sum_kernel(input_guard, input_data_slice, axes, keep_dims, output_shape)?;

    // 2. Diviser chaque élément par N
    if n == 0.0f64 {
        // Devrait être attrapé par la vérification n==0 dans mean_op, mais double-check ici.
        return Err(NeuraRustError::DivisionByZero);
    }

    // Division f64 / f64
    let mean_data: Vec<f64> = sum_data.into_iter().map(|val| val / n).collect();

    Ok(mean_data)
}

#[cfg(test)]
#[path = "mean_test.rs"]
mod tests; 