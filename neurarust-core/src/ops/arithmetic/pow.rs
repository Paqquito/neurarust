use crate::autograd::BackwardOp;
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::tensor::utils::broadcast_shapes;
use crate::ops::arithmetic::mul_op;
use crate::ops::math_elem::ln_op;
use crate::types::DType;

use num_traits::{Float, Zero};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// --- PowBackward Definition ---

/// Backward operation context for `pow_op`.
#[derive(Debug)]
struct PowBackward {
    base_node: Arc<RwLock<TensorData>>,
    exponent_node: Arc<RwLock<TensorData>>,
    base_clone: Tensor,
    exponent_clone: Tensor,
    output_clone: Tensor,
    base_requires_grad: bool,
    exponent_requires_grad: bool,
}

// --- BackwardOp Implementation for PowBackward ---

impl BackwardOp for PowBackward {
    /// Computes gradient for the power operation z = base^exponent.
    /// grad(base) = grad_output * exponent * base^(exponent - 1)
    /// grad(exponent) = grad_output * z * ln(base)
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

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut ids = Vec::new();
        if self.base_requires_grad { ids.push(Arc::as_ptr(&self.base_node)); }
        if self.exponent_requires_grad { ids.push(Arc::as_ptr(&self.exponent_node)); }
        ids
    }
}

// --- pow_kernel (Private Calculation Core) ---

/// Private kernel for element-wise power calculation with broadcasting.
fn pow_kernel<T>(
    output_shape: &[usize],
    base_data: &[T],
    base_shape: &[usize],
    base_strides: &[usize],
    base_offset: usize,
    exponent_data: &[T],
    exponent_shape: &[usize],
    exponent_strides: &[usize],
    exponent_offset: usize,
) -> Result<Vec<T>, NeuraRustError>
where
    T: Float + Copy + Zero + Debug,
{
    let output_numel = output_shape.iter().product::<usize>();
    let mut output_data = vec![T::zero(); output_numel];
    let mut current_coords = vec![0; output_shape.len()];

    for i in 0..output_numel {
        let mut base_physical_idx = base_offset;
        for dim in 0..base_shape.len() {
            let broadcast_dim_offset = output_shape.len() - base_shape.len();
            let coord_idx = broadcast_dim_offset + dim;
            let index = if base_shape[dim] == 1 && output_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
            base_physical_idx += index * base_strides[dim];
        }

        let mut exp_physical_idx = exponent_offset;
        for dim in 0..exponent_shape.len() {
            let broadcast_dim_offset = output_shape.len() - exponent_shape.len();
            let coord_idx = broadcast_dim_offset + dim;
            let index = if exponent_shape[dim] == 1 && output_shape[coord_idx] > 1 { 0 } else { current_coords[coord_idx] };
            exp_physical_idx += index * exponent_strides[dim];
        }

        if base_physical_idx >= base_data.len() || exp_physical_idx >= exponent_data.len() {
            return Err(NeuraRustError::InternalError(format!(
                 "Pow kernel index out of bounds. TargetCoords: {:?}, BaseIdx: {}, ExpIdx: {}, BaseLen: {}, ExpLen: {}",
                 current_coords,
                 base_physical_idx,
                 exp_physical_idx,
                 base_data.len(),
                 exponent_data.len()
            )));
        }

        let base_val = base_data[base_physical_idx];
        let exp_val = exponent_data[exp_physical_idx];
        output_data[i] = base_val.powf(exp_val);

        if i < output_numel - 1 {
            let mut dim_to_inc = output_shape.len();
            while dim_to_inc > 0 {
                dim_to_inc -= 1;
                current_coords[dim_to_inc] += 1;
                if current_coords[dim_to_inc] < output_shape[dim_to_inc] { break; }
                current_coords[dim_to_inc] = 0;
            }
        }
    }
    Ok(output_data)
}

// --- pow_op Implementation (Public API + Autograd Setup) ---

/// Computes element-wise power of `base` raised to `exponent`.
/// Supports broadcasting.
pub fn pow_op(base: &Tensor, exponent: &Tensor) -> Result<Tensor, NeuraRustError> {
    let base_guard = base.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock base data".to_string()))?;
    let exponent_guard = exponent.data.read().map_err(|_| NeuraRustError::InternalError("Failed to lock exponent data".to_string()))?;

    if base_guard.device != exponent_guard.device {
        return Err(NeuraRustError::DeviceMismatch {
            operation: "pow_op".to_string(),
            expected: base_guard.device,
            actual: exponent_guard.device,
        });
    }
    let device = base_guard.device;
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            "pow_op currently only supports CPU tensors.".to_string(),
        ));
    }

    if base_guard.dtype != DType::F32 || exponent_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "pow_op currently only supports F32 tensors.".to_string(),
        ));
    }
    let _output_dtype = DType::F32;

    let output_shape = broadcast_shapes(&base_guard.shape, &exponent_guard.shape)?;

    let base_shape = base_guard.shape.clone();
    let exponent_shape = exponent_guard.shape.clone();
    let base_strides = base_guard.strides.clone();
    let exponent_strides = exponent_guard.strides.clone();
    let base_offset = base_guard.offset;
    let exponent_offset = exponent_guard.offset;
    let base_requires_grad = base_guard.requires_grad;
    let exponent_requires_grad = exponent_guard.requires_grad;
    let autograd_needed = base_requires_grad || exponent_requires_grad;

    let base_buffer_arc = base_guard.buffer().try_get_cpu_f32()?.clone(); 
    let exponent_buffer_arc = exponent_guard.buffer().try_get_cpu_f32()?.clone();
    
    let base_node_arc = if autograd_needed { Some(base.data.clone()) } else { None };
    let exponent_node_arc = if autograd_needed { Some(exponent.data.clone()) } else { None };
    let base_clone = if autograd_needed { Some(base.clone()) } else { None };
    let exponent_clone = if autograd_needed { Some(exponent.clone()) } else { None };
    
    drop(base_guard);
    drop(exponent_guard);

    let result_data_vec = pow_kernel(
        &output_shape,
        base_buffer_arc.as_slice(),
        &base_shape,
        &base_strides,
        base_offset,
        exponent_buffer_arc.as_slice(),
        &exponent_shape,
        &exponent_strides,
        exponent_offset,
    )?;
    let result_buffer_arc = Arc::new(result_data_vec);

    let output_td = TensorData::new(
        result_buffer_arc.as_ref().clone(),
        output_shape,
    )?;
    let result_tensor = Tensor { data: Arc::new(RwLock::new(output_td)) };

    if autograd_needed {
        let a_arc = base_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing base_node_arc".to_string()))?;
        let e_arc = exponent_node_arc.ok_or_else(|| NeuraRustError::InternalError("Missing exponent_node_arc".to_string()))?;
        let b_clone = base_clone.ok_or_else(|| NeuraRustError::InternalError("Missing base_clone".to_string()))?;
        let exp_clone = exponent_clone.ok_or_else(|| NeuraRustError::InternalError("Missing exponent_clone".to_string()))?;
        let output_clone = result_tensor.clone();

        let mut output_guard = result_tensor.data.write().map_err(|_| NeuraRustError::InternalError("Failed to lock output tensor data for writing".to_string()))?;
        output_guard.requires_grad = true;
        let backward_context = PowBackward {
            base_node: a_arc,
            exponent_node: e_arc,
            base_clone: b_clone,
            exponent_clone: exp_clone,
            output_clone,
            base_requires_grad,
            exponent_requires_grad,
        };
        output_guard.grad_fn = Some(Arc::new(backward_context));
        println!("PowBackward grad_fn set for pow result.");
    }

    Ok(result_tensor)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::grad_check::check_grad;
    
    use crate::utils::testing::check_tensor_near;
    
    #[test]
    fn test_pow_forward() {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let exp = Tensor::new(vec![2.0, 3.0, 0.5], vec![3]).unwrap();
        let result = pow_op(&base, &exp).unwrap();
        let result_data = result.get_f32_data().expect("Failed to get result data");
        approx::assert_relative_eq!(result_data.as_slice(), [1.0, 8.0, 3.0_f32.sqrt()].as_slice(), epsilon = 1e-6);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_pow_forward_broadcast() {
        let base = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let exp = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
        let result = pow_op(&base, &exp).unwrap();
        let result_data = result.get_f32_data().expect("Failed to get result data");
        approx::assert_relative_eq!(result_data.as_slice(), [4.0, 8.0, 9.0, 27.0].as_slice(), epsilon = 1e-6);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_pow_backward_simple() {
        let base_data = vec![2.0, 3.0];
        let exp_data = vec![3.0, 2.0];
        let shape = vec![2];
        let base = Tensor::new(base_data.clone(), shape.clone()).unwrap();
        let exp = Tensor::new(exp_data.clone(), shape.clone()).unwrap();
        base.set_requires_grad(true).unwrap();
        exp.set_requires_grad(true).unwrap();

        let output = pow_op(&base, &exp).unwrap();
        
        let grad_output_data = vec![0.1, 0.2];
        let grad_output = Tensor::new(grad_output_data.clone(), shape.clone()).unwrap();

        output.backward(Some(grad_output)).expect("Backward pass failed");

        let grad_base_contig = base.grad().unwrap().contiguous().unwrap();
        let expected_grad_base = vec![1.2, 1.2];
        check_tensor_near(&grad_base_contig, &shape, &expected_grad_base, 1e-6);

        let grad_exp_contig = exp.grad().unwrap().contiguous().unwrap();
        let ln_base: Vec<f32> = base_data.iter().map(|&x| x.ln()).collect();
        let output_data = vec![8.0, 9.0];
        let expected_grad_exp: Vec<f32> = grad_output_data.iter().zip(output_data.iter()).zip(ln_base.iter())
            .map(|((&g, &z), &ln_b)| g * z * ln_b)
            .collect();
        check_tensor_near(&grad_exp_contig, &shape, &expected_grad_exp, 1e-6);
    }

    #[test]
    fn test_pow_backward_broadcast_exponent() {
        fn pow_func(inputs: &[Tensor]) -> Result<Tensor, NeuraRustError> {
            pow_op(&inputs[0], &inputs[1])
        }
        let base = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap(); 
        base.set_requires_grad(true).unwrap();
        let exp = Tensor::new(vec![3.0], vec![1]).unwrap(); 
        exp.set_requires_grad(true).unwrap();
        let output = pow_op(&base, &exp).unwrap();
        let output_grad_dummy = crate::tensor::ones_like(&output).expect("Failed to create dummy grad for broadcast_exponent test");
        let epsilon = 1e-3;
        let abs_tol = 1e-6;
        let rel_tol = 1e-4;
        check_grad(pow_func, &[base.clone(), exp.clone()], &output_grad_dummy, epsilon, abs_tol, rel_tol).unwrap();
    }

    #[test]
    fn test_pow_backward_broadcast_base() {
        let base_data = vec![2.0];
        let exp_data = vec![3.0, 4.0];
        let base_shape = vec![1];
        let exp_shape = vec![2];
        let base = Tensor::new(base_data.clone(), base_shape.clone()).unwrap();
        let exp = Tensor::new(exp_data.clone(), exp_shape.clone()).unwrap();
        base.set_requires_grad(true).unwrap();
        exp.set_requires_grad(true).unwrap();

        let output = pow_op(&base, &exp).unwrap();
        let output_shape = output.shape();
        let output_data = output.get_f32_data().unwrap();
        
        let grad_output_data = vec![0.1, 0.2];
        let grad_output = Tensor::new(grad_output_data.clone(), output_shape.clone()).unwrap();

        output.backward(Some(grad_output)).expect("Backward pass failed");

        let grad_base_contig = base.grad().unwrap().contiguous().unwrap();
        let expected_grad_base = vec![7.6];
        check_tensor_near(&grad_base_contig, &base_shape, &expected_grad_base, 1e-6);

        let grad_exp_contig = exp.grad().unwrap().contiguous().unwrap();
        let ln_base_val = base_data[0].ln();
        let expected_grad_exp: Vec<f32> = grad_output_data.iter().zip(output_data.iter())
            .map(|(&g, &z)| g * z * ln_base_val)
            .collect();
        check_tensor_near(&grad_exp_contig, &exp_shape, &expected_grad_exp, 1e-6);
    }

    #[test]
    fn test_pow_backward_only_base_grad() {
        fn pow_func(inputs: &[Tensor]) -> Result<Tensor, NeuraRustError> {
            pow_op(&inputs[0], &inputs[1])
        }
        let base = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
        base.set_requires_grad(true).unwrap();
        let exp = Tensor::new(vec![3.0, 2.0], vec![2]).unwrap();
        exp.set_requires_grad(false).unwrap();
        let output = pow_op(&base, &exp).unwrap();
        let output_grad_dummy = crate::tensor::ones_like(&output).expect("Failed to create dummy grad for only_base_grad test");
        let epsilon = 1e-3;
        let abs_tol = 1e-6;
        let rel_tol = 1e-4;
        check_grad(pow_func, &[base.clone(), exp.clone()], &output_grad_dummy, epsilon, abs_tol, rel_tol).unwrap();
    }

    #[test]
    fn test_pow_backward_only_exponent_grad() {
        fn pow_func(inputs: &[Tensor]) -> Result<Tensor, NeuraRustError> {
            pow_op(&inputs[0], &inputs[1])
        }
        let base = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap(); 
        base.set_requires_grad(false).unwrap();
        let exp = Tensor::new(vec![3.0, 2.0], vec![2]).unwrap(); 
        exp.set_requires_grad(true).unwrap();
        let output = pow_op(&base, &exp).unwrap();
        let output_grad_dummy = crate::tensor::ones_like(&output).expect("Failed to create dummy grad for only_exponent_grad test");
        let epsilon = 1e-3;
        let abs_tol = 1e-6;
        let rel_tol = 1e-4;
        check_grad(pow_func, &[base.clone(), exp.clone()], &output_grad_dummy, epsilon, abs_tol, rel_tol).unwrap(); 
    }
} 