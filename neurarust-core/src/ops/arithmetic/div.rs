use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::device::StorageDevice;
use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use crate::tensor::Tensor;
use num_traits::{One, Zero};
use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::Debug;
use std::iter::Sum;
// Add Neg, Sub traits if needed by internal ops
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::{Arc, RwLockReadGuard};

// --- Backward Operation Structure ---

/// Backward operation context for division.
#[derive(Debug)]
struct DivBackward<T: 'static + Debug + Copy + Send + Sync> {
    a: Tensor<T>,
    b: Tensor<T>,
    result: Tensor<T>, // Need the result z = a/b for grad_b calculation (a * (-1/b^2))
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

// --- Backward Operation Implementation ---

impl<T> BackwardOp<T> for DivBackward<T>
where
    T: Debug
        + Copy
        + Send
        + Sync
        + 'static
        + Clone
        + Default
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + AddAssign
        + Sum
        + PartialEq
        + PartialOrd
        + std::iter::Product,
{
    fn inputs(&self) -> Vec<NodeId<T>> {
        // Note: result is not an input in the graph sense
        vec![self.a.get_node_id(), self.b.get_node_id()]
    }

    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // grad_a = grad_output / b
        let grad_a_unreduced = div_op(grad_output, &self.b)?;
        let grad_a = reduce_gradient_to_shape(&grad_a_unreduced, &self.a_shape)?;

        // grad_b = grad_output * (- result / b)
        let neg_result_div_b = div_op(&crate::ops::arithmetic::neg_op(&self.result)?, &self.b)?;
        let grad_b_unreduced = crate::ops::arithmetic::mul_op(grad_output, &neg_result_div_b)?;
        let grad_b = reduce_gradient_to_shape(&grad_b_unreduced, &self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}

// --- Kernel de Calcul --- (Nouveau)

/// Noyau de calcul privé pour la division élément par élément avec broadcasting.
fn div_kernel<T>(
    a_guard: &RwLockReadGuard<'_, TensorData<T>>,
    b_guard: &RwLockReadGuard<'_, TensorData<T>>,
    a_data_slice: &[T],
    b_data_slice: &[T],
    output_shape: &[usize],
) -> Result<Vec<T>, NeuraRustError>
where
    // Bounds pour get_offset (Debug), calcul (Div, Copy), et check zero (Zero, PartialEq)
    T: Div<Output = T> + Copy + Debug + Zero + PartialEq,
{
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    let result_strides = calculate_strides(output_shape);

    let rank_diff_a = output_shape.len().saturating_sub(a_guard.shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_guard.shape.len());

    let mut input_a_coords = vec![0; a_guard.shape.len()];
    let mut input_b_coords = vec![0; b_guard.shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, output_shape);

        // Indice pour A
        for dim_idx in 0..a_guard.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_a = a_guard.get_offset(&input_a_coords);
        let val_a = a_data_slice[offset_a];

        // Indice pour B
        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 {
                0
            } else {
                output_coords[output_coord_idx]
            };
        }
        let offset_b = b_guard.get_offset(&input_b_coords);
        let val_b = b_data_slice[offset_b];

        // Vérification division par zéro
        if val_b == T::zero() {
            return Err(NeuraRustError::DivisionByZero);
        }

        result_data_vec.push(val_a / val_b); // Opération de division
    }

    Ok(result_data_vec)
}

// --- Forward Operation ---

/// Performs element-wise division for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
pub fn div_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    // Inclure tous les bounds requis par DivBackward et reduce_gradient
    T: Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + Add<Output = T>
        + AddAssign
        + Copy
        + Clone
        + Debug
        + Default
        + Zero
        + One
        + Sum
        + 'static
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + Sub<Output = T>
        + std::iter::Product,
{
    // --- Autograd Setup --- (Adapté pour cloner a, b, ET garder résultat)
    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut a_maybe_clone: Option<Tensor<T>> = None;
    let mut b_maybe_clone: Option<Tensor<T>> = None;
    let mut a_shape_maybe: Option<Vec<usize>> = None;
    let mut b_shape_maybe: Option<Vec<usize>> = None;
    if requires_grad {
        a_maybe_clone = Some(a.clone());
        b_maybe_clone = Some(b.clone());
        a_shape_maybe = Some(a.shape());
        b_shape_maybe = Some(b.shape());
    }

    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check ---
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Cannot divide tensors on different devices: {:?} and {:?}",
            a_guard.device, b_guard.device
        )));
    }
    let device = a_guard.device;
    if device != StorageDevice::CPU {
        return Err(NeuraRustError::UnsupportedOperation(format!(
            "Division is currently only supported on CPU, not {:?}",
            device
        )));
    }

    let a_data_arc = a_guard.data.cpu_data()?.clone();
    let b_data_arc = b_guard.data.cpu_data()?.clone();
    let a_data_slice = a_data_arc.as_slice();
    let b_data_slice = b_data_arc.as_slice();

    // --- Shape and Broadcasting ---
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    let output_shape = broadcast_shapes(a_shape, b_shape).map_err(|_e| {
        NeuraRustError::BroadcastError {
            shape1: a_shape.clone(),
            shape2: b_shape.clone(),
        }
    })?;

    // --- Calculation (Appel au Kernel) ---
    let result_data_vec = div_kernel(
        &a_guard,
        &b_guard,
        a_data_slice,
        b_data_slice,
        &output_shape,
    )?;

    // Drop read locks
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor ---
    let result_tensor = Tensor::new(result_data_vec, output_shape.clone())?;

    // --- Autograd Linkage --- (Adapté)
    if requires_grad {
        let backward_context = DivBackward {
            a: a_maybe_clone.unwrap(),
            b: b_maybe_clone.unwrap(),
            // Cloner le résultat est nécessaire pour le backward de la division
            result: result_tensor.clone(),
            a_shape: a_shape_maybe.unwrap(),
            b_shape: b_shape_maybe.unwrap(),
        };
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }

    Ok(result_tensor)
}

// Copier reduce_gradient_to_shape ici
/// Helper function to reduce a gradient tensor to a target shape.
fn reduce_gradient_to_shape<T>(
    gradient: &Tensor<T>,
    target_shape: &[usize],
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Copy
        + Debug
        + Zero
        + One
        + Add<Output = T>
        + AddAssign
        + Sum
        + Default
        + PartialEq
        + PartialOrd
        + Send
        + Sync
        + 'static
        + std::iter::Product,
{
    if gradient.shape() == target_shape {
        Ok(gradient.clone())
    } else {
        if target_shape.is_empty() {
            crate::ops::reduction::sum_axes(gradient, &[], false)
        } else {
            let current_shape = gradient.shape();
            let rank_diff = current_shape.len().saturating_sub(target_shape.len());
            let mut axes_to_reduce: Vec<usize> = (0..rank_diff).collect();

            for i in 0..target_shape.len() {
                if target_shape[i] == 1 && current_shape[rank_diff + i] > 1 {
                    axes_to_reduce.push(rank_diff + i);
                } else if target_shape[i] != current_shape[rank_diff + i] && target_shape[i] != 1 {
                    return Err(NeuraRustError::InternalError(format!(
                        "Cannot reduce gradient shape {:?} to {:?}: Incompatible dimensions found.",
                        current_shape, target_shape
                    )));
                }
            }

            if axes_to_reduce.is_empty() {
                if current_shape == target_shape {
                    Ok(gradient.clone())
                } else {
                    return Err(NeuraRustError::InternalError(format!(
                         "Cannot reduce gradient shape {:?} to {:?}: No reduction axes found but shapes differ.",
                         current_shape, target_shape
                      )));
                }
            } else {
                let reduced_grad =
                    crate::ops::reduction::sum_axes(gradient, &axes_to_reduce, false)?;
                let final_shape: Vec<usize> = target_shape.to_vec();
                let reduced_numel: usize = reduced_grad.shape().iter().product();
                let target_numel: usize = target_shape.iter().product();
                if reduced_numel != target_numel {
                    return Err(NeuraRustError::InternalError(format!(
                         "Gradient reduction produced incompatible shape {:?} (numel {}) for target {:?} (numel {}). Reduction axes: {:?}.",
                         reduced_grad.shape(), reduced_numel, target_shape, target_numel, axes_to_reduce
                     )));
                }
                crate::ops::view::reshape_op(&reduced_grad, final_shape)
            }
        }
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::utils::testing::{create_test_tensor, create_test_tensor_with_grad};
    use approx::assert_relative_eq;

    #[test]
    fn test_div_tensors_ok() {
        let t1 = create_test_tensor(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]);
        let t2 = create_test_tensor(vec![2.0f32, 5.0, 2.0, 8.0], vec![2, 2]);
        let expected_data = vec![5.0f32, 4.0, 15.0, 5.0];
        let expected_shape = vec![2, 2];

        let result = div_op(&t1, &t2).unwrap();
        assert_eq!(result.shape(), expected_shape);
        let res_buffer = result.borrow_data_buffer();
        let res_data = res_buffer.cpu_data().unwrap();
        assert_eq!(res_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_div_by_zero() {
        let t1 = create_test_tensor(vec![1.0f32], vec![1]);
        let t2 = create_test_tensor(vec![0.0f32], vec![1]);
        let result = div_op(&t1, &t2);
        assert!(matches!(result, Err(NeuraRustError::DivisionByZero)));
    }

    #[test]
    fn test_div_broadcasting() {
        let t1 = create_test_tensor(vec![10.0f32, 20.0], vec![2, 1]);
        let t2 = create_test_tensor(vec![2.0f32, 5.0], vec![1, 2]);
        let expected_data = vec![5.0f32, 2.0, 10.0, 4.0]; // 10/2, 10/5, 20/2, 20/5
        let expected_shape = vec![2, 2];

        let result = div_op(&t1, &t2).unwrap();
        assert_eq!(result.shape(), expected_shape);
        let res_buffer = result.borrow_data_buffer();
        let res_data = res_buffer.cpu_data().unwrap();
        assert_eq!(res_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_div_backward_simple() {
        let a = create_test_tensor_with_grad::<f64>(vec![10.0, 21.0], vec![2]);
        let b = create_test_tensor_with_grad::<f64>(vec![2.0, 3.0], vec![2]);
        let output_grad = Tensor::<f64>::ones(vec![2]).unwrap();

        let c = div_op(&a, &b).unwrap(); // c = [5.0, 7.0]
        c.backward(Some(output_grad)).unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        // grad_a = grad_output / b = [1, 1] / [2, 3] = [0.5, 1/3]
        let expected_grad_a = vec![0.5, 1.0 / 3.0];
        // grad_b = grad_output * (-result / b)
        //        = [1, 1] * [-2.5, -7/3] = [-2.5, -7/3]
        let expected_grad_b = vec![-2.5, -7.0 / 3.0];

        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();
        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        assert_eq!(grad_a.shape(), vec![2]);
        assert_eq!(grad_b.shape(), vec![2]);

        for (i, &val) in grad_a_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_a[i], epsilon = 1e-9);
        }
        for (i, &val) in grad_b_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_b[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_div_backward_broadcast() {
        let a = create_test_tensor_with_grad::<f64>(vec![12.0], vec![1]); // Scalar a
        let b = create_test_tensor_with_grad::<f64>(vec![2.0, 3.0, 4.0], vec![3]); // Vector b
        let output_grad = Tensor::<f64>::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        // c = a / b = [12, 12, 12] / [2, 3, 4] = [6, 4, 3]
        let c = div_op(&a, &b).unwrap();
        c.backward(Some(output_grad)).unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        // grad_a_unreduced = grad_output / b = [1, 2, 3] / [2, 3, 4] = [0.5, 2/3, 0.75]
        // grad_a = sum(grad_a_unreduced) = 0.5 + 2/3 + 0.75 = 1.91666...
        let expected_grad_a = vec![0.5 + 2.0/3.0 + 0.75];

        // grad_b_unreduced = grad_output * (-result / b)
        //                  = [1, 2, 3] * ( -[6, 4, 3] / [2, 3, 4] )
        //                  = [1, 2, 3] * [ -3, -4/3, -0.75 ]
        //                  = [ -3, -8/3, -2.25 ]
        // grad_b = grad_b_unreduced (no reduction needed for b's shape)
        let expected_grad_b = vec![-3.0, -8.0/3.0, -2.25];

        let grad_a_buffer = grad_a.borrow_data_buffer();
        let grad_a_data = grad_a_buffer.cpu_data().unwrap();
        let grad_b_buffer = grad_b.borrow_data_buffer();
        let grad_b_data = grad_b_buffer.cpu_data().unwrap();

        assert_eq!(grad_a.shape(), vec![1]);
        assert_eq!(grad_b.shape(), vec![3]);

        for (i, &val) in grad_a_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_a[i], epsilon = 1e-9);
        }
        for (i, &val) in grad_b_data.iter().enumerate() {
            assert_relative_eq!(val, expected_grad_b[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_div_backward_with_zero_divisor() {
        let a = create_test_tensor_with_grad::<f64>(vec![1.0], vec![1]);
        let b = create_test_tensor_with_grad::<f64>(vec![0.0], vec![1]); // Divisor is zero
        let output_grad = Tensor::<f64>::ones(vec![1]).unwrap();

        // Forward pass should fail
        let result_forward = div_op(&a, &b);
        assert!(matches!(result_forward, Err(NeuraRustError::DivisionByZero)));

        // Although forward fails, let's construct a hypothetical backward pass
        // scenario where the gradient calculation might involve division by b.
        // We need to ensure the backward pass handles this gracefully if possible,
        // though typically the forward error prevents reaching backward.

        // Create a dummy 'c' that requires grad to simulate reaching backward
        let dummy_c = Tensor::<f64>::scalar(1.0);
        dummy_c.set_requires_grad(true).unwrap();

        // Create DivBackward manually (this wouldn't happen naturally if forward fails)
        let backward_context = DivBackward {
            a: a.clone(),
            b: b.clone(), // b contains zero
            result: dummy_c.clone(), // dummy result
            a_shape: a.shape(),
            b_shape: b.shape(),
        };

        // Calling backward should fail due to division by zero in gradient calculation
        let result_backward = backward_context.backward(&output_grad);
        assert!(matches!(result_backward, Err(NeuraRustError::DivisionByZero)), "Backward pass did not detect division by zero");
    }
}
