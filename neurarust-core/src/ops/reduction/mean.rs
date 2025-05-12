use crate::autograd::BackwardOp;
use crate::autograd::graph::NodeId; // Correct import path
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLockReadGuard};
use crate::types::DType;
use crate::ops::arithmetic::mul_op;
use crate::ops::view::expand_op;
use super::utils::{calculate_reduction_output_shape, process_reduction_axes};
use crate::tensor::create::{zeros, full, full_f64}; // Import specific creation fns
use crate::ops::traits::NeuraNumeric; // Correct import path for the trait
use crate::ops::dtype::cast_op; // Importer cast_op
use crate::ops::reduction::sum::sum_kernel; // Importer sum_kernel
// use crate::autograd::grad_check::check_grad; // Keep commented until check_grad is ready

// --- MeanBackward Definition ---

/// Backward operation context for `mean` reduction.
///
/// Stores information needed to compute the gradient of the mean operation:
/// - A reference to the original input tensor's data (`input_node`).
/// - The total number of elements that were reduced (`num_elements_reduced`) to compute the mean.
#[derive(Debug)]
struct MeanBackward {
    input_node: NodeId,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    keep_dims: bool,
}

// Add unsafe impls for Send + Sync because NodeId is a raw pointer
unsafe impl Send for MeanBackward {}
unsafe impl Sync for MeanBackward {}

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
        let numel_reduced: usize = self.input_shape.iter().product::<usize>() 
            / self.output_shape.iter().product::<usize>();
        if numel_reduced == 0 {
             let zero_grad = zeros(&self.input_shape)?;
             return Ok(vec![cast_op(&zero_grad, grad_output.dtype())?]); // Utiliser cast_op
        }
        let scale_factor = 1.0 / (numel_reduced as f64);

        let scale_tensor = match grad_output.dtype() {
             DType::F32 => full(&[], scale_factor as f32)?,
             DType::F64 => full_f64(&[], scale_factor)?,
             DType::I32 | DType::I64 | DType::Bool => {
                 return Err(NeuraRustError::UnsupportedOperation(
                     "mean_backward n'est pas supporté pour les tenseurs de type I32, I64 ou Bool".to_string())
                 );
             },
        };

        let scaled_grad = mul_op(grad_output, &scale_tensor)?;

        let grad_to_expand = if !self.keep_dims {
            scaled_grad.reshape(self.output_shape.clone())?
        } else {
            scaled_grad
        };

        let target_shape_isize: Vec<isize> = self.input_shape.iter().map(|&d| d as isize).collect();
        let grad_input = expand_op(&grad_to_expand, &target_shape_isize)?;

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input_node]
    }
}

// --- Kernel de Calcul (F32 CPU) ---

/// Noyau de calcul privé pour la moyenne avec réduction d'axes.
fn mean_kernel<T> (
    input_guard: &RwLockReadGuard<TensorData>, // Renommé pour enlever le _
    data: &[T], // Renommé
    axes: &[usize], // Renommé
    keep_dims: bool, // Renommé
    output_shape: &[usize], // Renommé
    n_val: T, // Renommé en n_val pour éviter conflit avec N générique si jamais
) -> Result<Vec<T>, NeuraRustError>
where
    T: NeuraNumeric + std::ops::AddAssign + std::ops::Div<Output = T> + Default + Copy + Debug, // Mise à jour des traits
{
    if n_val == T::zero() {
        // Si N est zéro (et numel > 0), mean_op devrait déjà avoir retourné une erreur.
        // Si numel est aussi zéro, mean_op retourne un tenseur de zéros casté.
        // Ce cas ici est une double sécurité ou pour un N=0 inattendu.
        // Retourner des zéros de la bonne forme est un comportement sûr.
        println!("Warning: mean_kernel called with n_val = 0. Outputting zeros.");
        return Ok(vec![T::zero(); output_shape.iter().product()]);
    }

    // 1. Calculer la somme en utilisant sum_kernel
    let sum_result = sum_kernel(
        input_guard,
        data,
        axes,
        keep_dims, // sum_kernel utilise keep_dims pour déterminer sa logique interne
        output_shape,
    )?;

    // 2. Diviser chaque élément du résultat de la somme par n_val
    let mean_result: Vec<T> = sum_result.into_iter().map(|sum_val| sum_val / n_val).collect();

    Ok(mean_result)
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
    let requires_grad = t_guard.requires_grad;
    let input_node_id = tensor.node_id();
    let input_shape = t_guard.shape.clone();
    let dtype = t_guard.dtype;

    // --- Process Axes & Output Shape ---
    let rank = input_shape.len();
    let axes_vec = process_reduction_axes(rank, axes)?;
    let output_shape = calculate_reduction_output_shape(&input_shape, &axes_vec, keep_dims);
    let output_shape_clone = output_shape.clone();

    // --- Calculate N ---
    let n: usize = if axes_vec.is_empty() {
        t_guard.numel()
    } else {
        axes_vec.iter().map(|&axis| input_shape[axis]).product()
    };

    if n == 0 && t_guard.numel() > 0 { // Reducing over zero elements, but tensor is not empty
        return Err(NeuraRustError::UnsupportedOperation(
            "Cannot compute mean over zero elements (e.g., reducing an axis with size 0)".to_string(),
        ));
    } else if n == 0 && t_guard.numel() == 0 { // Input tensor is empty
         let zeros_tensor = zeros(&output_shape)?;
         return cast_op(&zeros_tensor, dtype); // Utiliser cast_op
    }

    // --- Dispatch Kernel (Placeholder logic) ---
    let output_tensor = match dtype {
        DType::F32 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f32()?.as_slice();
            let n_f32 = n as f32;
            let result_data = mean_kernel::<f32>(
                &t_guard,
                input_data_slice, 
                &axes_vec, 
                keep_dims, 
                &output_shape, 
                n_f32,
            )?;
            drop(t_guard);
            Tensor::new(result_data, output_shape)?
        }
        DType::F64 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_f64()?.as_slice();
            let n_f64 = n as f64;
            let result_data = mean_kernel::<f64>(
                &t_guard,
                input_data_slice, 
                &axes_vec, 
                keep_dims, 
                &output_shape, 
                n_f64,
            )?;
            drop(t_guard);
            Tensor::new_f64(result_data, output_shape)?
        }
        DType::I32 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_i32()?.as_slice();
            let n_f32 = n as f32;
            let sum_data = sum_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec,
                keep_dims,
                &output_shape,
            )?;
            let mean_data: Vec<f32> = sum_data.into_iter().map(|s| s as f32 / n_f32).collect();
            drop(t_guard);
            Tensor::new(mean_data, output_shape)?
        }
        DType::I64 => {
            let input_data_slice = t_guard.buffer().try_get_cpu_i64()?.as_slice();
            let n_f64 = n as f64;
            let sum_data = sum_kernel(
                &t_guard,
                input_data_slice,
                &axes_vec,
                keep_dims,
                &output_shape,
            )?;
            let mean_data: Vec<f64> = sum_data.into_iter().map(|s| s as f64 / n_f64).collect();
            drop(t_guard);
            Tensor::new_f64(mean_data, output_shape)?
        }
        DType::Bool => {
            let input_data_slice = t_guard.buffer().try_get_cpu_bool()?.as_slice();
            let n_f32 = n as f32;
            let input_as_i32: Vec<i32> = input_data_slice.iter().map(|&b| if b { 1 } else { 0 }).collect();
            let sum_data = sum_kernel(
                &t_guard,
                &input_as_i32,
                &axes_vec,
                keep_dims,
                &output_shape,
            )?;
            let mean_data: Vec<f32> = sum_data.into_iter().map(|s| s as f32 / n_f32).collect();
            drop(t_guard);
            Tensor::new(mean_data, output_shape)?
        }
    };

    // --- Autograd Setup ---
    if requires_grad {
        let grad_fn = MeanBackward { // Utilise la structure correcte
            input_node: input_node_id,
            //num_elements_reduced: n, // On recalcule dans backward si besoin
            input_shape: input_shape, 
            output_shape: output_shape_clone, 
            keep_dims: keep_dims,
        };
        let mut output_guard = output_tensor.write_data();
        output_guard.grad_fn = Some(Arc::new(grad_fn));
        output_guard.requires_grad = true;
    }

    Ok(output_tensor)
}

#[cfg(test)]
#[path = "mean_test.rs"]
mod tests; 