use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::tensor::utils::broadcast_shapes;
use crate::ops::arithmetic::mul_op;
use crate::ops::math_elem::ln_op;
use crate::types::DType;
use crate::ops::traits::NeuraNumeric;
use crate::tensor::iter_utils::{NdArrayBroadcastingIter, NdArrayBroadcastingIterF64};

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// --- Simple Generic Kernel ---

/// Generic kernel for element-wise power operation.
fn pow_kernel<T: NeuraNumeric>(base: T, exponent: T) -> T {
    // NeuraNumeric requires Float trait which has powf/powi
    base.powf(exponent) // Use powf for T^T (float^float)
}

// --- PowBackward Definition ---

/// Backward pass structure for the element-wise power operation (`base` ^ `exponent`).
///
/// Stores clones of the base, exponent, and output tensors, along with node references
/// and requirement flags, needed for calculating gradients and managing the graph.
#[derive(Debug)]
struct PowBackward {
    /// Reference counted pointer to the base tensor's data.
    base_node: Arc<RwLock<TensorData>>,
    /// Reference counted pointer to the exponent tensor's data.
    exponent_node: Arc<RwLock<TensorData>>,
    /// Clone of the base tensor.
    base_clone: Tensor,
    /// Clone of the exponent tensor.
    exponent_clone: Tensor,
    /// Clone of the output tensor (result of the forward pass, `z = base^exponent`).
    output_clone: Tensor,
    /// Flag indicating if the base tensor required gradients.
    base_requires_grad: bool,
    /// Flag indicating if the exponent tensor required gradients.
    exponent_requires_grad: bool,
}

// --- BackwardOp Implementation for PowBackward ---

impl BackwardOp for PowBackward {
    /// Computes gradients for the power operation \( z = a^b \) (base `a`, exponent `b`).
    ///
    /// Using the chain rule \( \frac{dL}{dx} = \frac{dL}{dz} \cdot \frac{dz}{dx} \), the gradients are:
    /// \\[ \frac{dL}{da} = \frac{dL}{dz} \cdot \frac{dz}{da} = \frac{dL}{dz} \cdot (b \cdot a^{b-1}) \\]
    /// \\[ \frac{dL}{db} = \frac{dL}{dz} \cdot \frac{dz}{db} = \frac{dL}{dz} \cdot (a^b \cdot \ln(a)) = \frac{dL}{dz} \cdot (z \cdot \ln(a)) \\]
    ///
    /// Where \( \frac{dL}{dz} \) is `grad_output`.
    ///
    /// **Broadcasting Handling:** If broadcasting occurred, gradients are reduced back to the
    /// original input shapes using [`Tensor::reduce_to_shape`](../../tensor/broadcast_utils/struct.Tensor.html#method.reduce_to_shape).
    ///
    /// **Numerical Stability:** The gradient w.r.t. the exponent involves \( \ln(a) \).
    /// If the base \( a \) contains non-positive values, this can result in `NaN` or infinite gradients.
    /// The forward `pow_op` might restrict inputs in the future, but currently relies on `ln_op` propagating `NaN`.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::with_capacity(2);

        // --- Gradient for Base --- 
        if self.base_requires_grad {
            // println!("Calculating grad for pow base"); // Keep commented
            let base_guard = self.base_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock base node".to_string()))?;
            
            // Calculate exponent - 1
            let one = crate::tensor::full(&[], 1.0f32)?; // Use full from create (now exported)
            let exponent_minus_one = crate::ops::arithmetic::sub_op(&self.exponent_clone, &one)?;
            
            // Calculate base^(exponent - 1)
            let base_pow_exp_minus_one = pow_op(&self.base_clone, &exponent_minus_one)?;
            
            // Calculate exponent * base^(exponent - 1)
            let term1 = crate::ops::arithmetic::mul_op(&self.exponent_clone, &base_pow_exp_minus_one)?;
            
            // Calculate grad_output * term1
            let grad_base_unreduced = crate::ops::arithmetic::mul_op(grad_output, &term1)?;
            
            // Reduce gradient to original base shape
            let grad_base = grad_base_unreduced.reduce_to_shape(&base_guard.shape)?;
            grads.push(grad_base); 
        }

        // --- Gradient for Exponent --- 
        if self.exponent_requires_grad {
            // println!("Calculating grad for pow exponent (using ln_op)"); // Keep commented
            let exponent_guard = self.exponent_node.read().map_err(|_| NeuraRustError::InternalError("Failed to lock exponent node".to_string()))?;

            // Calculate ln(base)
            // Need to handle potential errors if base <= 0. 
            // If base <= 0, ln(base) is NaN or error. The gradient might be NaN/Inf.
            // Let ln_op return NaN as implemented, multiplication should propagate it.
            // Consider adding checks in pow_op later to prevent non-positive base if exponent requires grad?
            let ln_base = ln_op(&self.base_clone)?;
            
            // Calculate z * ln(base)  (z is stored in self.output_clone)
            let term2 = mul_op(&self.output_clone, &ln_base)?;
            
            // Calculate grad_output * term2
            let grad_exp_unreduced = mul_op(grad_output, &term2)?;

            // Reduce gradient to original exponent shape
            let grad_exp = grad_exp_unreduced.reduce_to_shape(&exponent_guard.shape)?;
            grads.push(grad_exp);
        }

        Ok(grads)
    }

    /// Returns the identifiers of the input tensor nodes that required gradients.
    /// The order corresponds to the inputs `base` and `exponent`.
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut ids = Vec::new();
        if self.base_requires_grad { ids.push(Arc::as_ptr(&self.base_node)); }
        if self.exponent_requires_grad { ids.push(Arc::as_ptr(&self.exponent_node)); }
        ids
    }
}

// --- pow_op Implementation (Refactored) ---

/// Computes `base` raised to the power of `exponent` element-wise (`base` ^ `exponent`).
pub fn pow_op(base: &Tensor, exponent: &Tensor) -> Result<Tensor, NeuraRustError> {
    let base_guard = base.read_data();
    let exponent_guard = exponent.read_data();

    // --- Device Check ---
    if base_guard.device != StorageDevice::CPU || exponent_guard.device != StorageDevice::CPU {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "pow_op".to_string(),
            expected: StorageDevice::CPU,
            actual: if base_guard.device != StorageDevice::CPU { base_guard.device } else { exponent_guard.device },
        });
    }
    // --- DType Check (Allow matching F32 or F64) ---
    if base_guard.dtype != exponent_guard.dtype {
        return Err(NeuraRustError::DataTypeMismatch {
            operation: "pow_op".to_string(),
            expected: base_guard.dtype,
            actual: exponent_guard.dtype,
        });
    }
    let dtype = base_guard.dtype;

    // --- Broadcasting ---
    let output_shape = broadcast_shapes(&base_guard.shape, &exponent_guard.shape)?;
    let numel = output_shape.iter().product();

    // --- Prepare for Autograd (Clones needed by PowBackward) --- 
    let requires_grad = base_guard.requires_grad || exponent_guard.requires_grad;
    let base_node_arc = if requires_grad { Some(Arc::clone(&base.data)) } else { None };
    let exponent_node_arc = if requires_grad { Some(Arc::clone(&exponent.data)) } else { None };
    // PowBackward specifically needs these clones
    let base_clone = if requires_grad { Some(base.clone()) } else { None }; 
    let exponent_clone = if requires_grad { Some(exponent.clone()) } else { None };

    // --- DType Dispatch for Computation using Broadcasting Iterators --- 
    let result_tensor = match dtype {
        DType::F32 => {
            let base_buffer = base_guard.buffer.try_get_cpu_f32()?;
            let exponent_buffer = exponent_guard.buffer.try_get_cpu_f32()?;
            
            let iter_base = NdArrayBroadcastingIter::new(base_buffer, &base_guard.shape, &base_guard.strides, base_guard.offset, &output_shape)?;
            let iter_exponent = NdArrayBroadcastingIter::new(exponent_buffer, &exponent_guard.shape, &exponent_guard.strides, exponent_guard.offset, &output_shape)?;
            
            // Use the simple pow_kernel
            let output_data_vec: Vec<f32> = iter_base.zip(iter_exponent)
                .map(|(vbase, vexponent)| pow_kernel::<f32>(vbase, vexponent))
                .collect();
            
            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!(
                    "pow_op F32: Output vec len {} mismatch with expected numel {}",
                     output_data_vec.len(), numel
                )));
            }
            
            drop(base_guard); drop(exponent_guard);
            Tensor::new(output_data_vec, output_shape)?
        }
        DType::F64 => {
            let base_buffer = base_guard.buffer.try_get_cpu_f64()?;
            let exponent_buffer = exponent_guard.buffer.try_get_cpu_f64()?;

            let iter_base = NdArrayBroadcastingIterF64::new(base_buffer, &base_guard.shape, &base_guard.strides, base_guard.offset, &output_shape)?;
            let iter_exponent = NdArrayBroadcastingIterF64::new(exponent_buffer, &exponent_guard.shape, &exponent_guard.strides, exponent_guard.offset, &output_shape)?;

            // Use the simple pow_kernel
            let output_data_vec: Vec<f64> = iter_base.zip(iter_exponent)
                .map(|(vbase, vexponent)| pow_kernel::<f64>(vbase, vexponent))
                .collect();

            if output_data_vec.len() != numel {
                 return Err(NeuraRustError::InternalError(format!(
                    "pow_op F64: Output vec len {} mismatch with expected numel {}",
                     output_data_vec.len(), numel
                )));
            }

            drop(base_guard); drop(exponent_guard);
            Tensor::new_f64(output_data_vec, output_shape)?
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    };

    // --- Autograd Setup --- 
    if requires_grad {
        let a_arc = base_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing base_node_arc for pow_op autograd".to_string()))?;
        let e_arc = exponent_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing exponent_node_arc for pow_op autograd".to_string()))?;
        let b_clone = base_clone.ok_or_else(|| NeuraRustError::InternalError("Missing base_clone for pow_op autograd".to_string()))?;
        let exp_clone = exponent_clone.ok_or_else(|| NeuraRustError::InternalError("Missing exponent_clone for pow_op autograd".to_string()))?;
        // Crucially, PowBackward also needs the output clone
        let output_clone = result_tensor.clone();

        let mut output_guard = result_tensor.write_data(); // Use helper method
        output_guard.requires_grad = true;
        let backward_context = PowBackward {
            base_node: a_arc,
            exponent_node: e_arc,
            base_clone: b_clone,
            exponent_clone: exp_clone,
            output_clone, // Pass the output clone
            base_requires_grad: base.requires_grad(), // Read requires_grad again just in case
            exponent_requires_grad: exponent.requires_grad(),
        };
        output_guard.grad_fn = Some(Arc::new(backward_context));
        // println!("PowBackward grad_fn set for pow result."); // Keep commented
    }

    Ok(result_tensor)
}

// --- Tests ---
// Link the external test file
#[cfg(test)]
#[path = "pow_test.rs"]
mod tests;

/* --- REMOVED internal tests module --- 
#[cfg(test)]
mod tests {
    // ... (contenu de l'ancien module de tests) ...
}
*/ 