use crate::autograd::backward_op::BackwardOp;
use crate::autograd::graph::NodeId;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

use crate::device::StorageDevice;
use crate::types::DType;

// --- Relu Operation ---

/// Backward pass structure for ReLU.
#[derive(Debug)]
struct ReluBackward {
    input_node: Arc<RwLock<TensorData>>,
}

impl BackwardOp for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let input_guard = self.input_node.read().expect("Failed to lock input node for read");

        let grad_output_guard = grad_output.read_data();
        if input_guard.device != grad_output_guard.device {
            return Err(NeuraRustError::DeviceMismatch {
                expected: input_guard.device,
                actual: grad_output_guard.device,
                operation: "ReLU backward".to_string(),
            });
        }
        if input_guard.device != StorageDevice::CPU || input_guard.dtype != DType::F32 {
            return Err(NeuraRustError::UnsupportedOperation(
                "ReLU backward currently only supports F32 tensors on CPU.".to_string(),
            ));
        }

        let input_buffer_arc = input_guard.buffer().try_get_cpu_f32()?.clone();
        let grad_output_buffer_arc = grad_output_guard.buffer().try_get_cpu_f32()?.clone();
        let input_slice = input_buffer_arc.as_slice();
        let grad_output_slice = grad_output_buffer_arc.as_slice();

        let mut grad_input_data = vec![0.0f32; input_guard.numel()];

        let input_offset = input_guard.offset;
        let grad_output_offset = grad_output_guard.offset;
        let len = grad_input_data.len();

        for i in 0..len {
            if input_slice[input_offset + i] > 0.0 {
                grad_input_data[i] = grad_output_slice[grad_output_offset + i];
            }
        }

        let grad_input_tensor = Tensor::new(grad_input_data, input_guard.shape.clone())?;

        Ok(vec![grad_input_tensor])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![Arc::as_ptr(&self.input_node)]
    }
}

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
///
/// ReLU(x) = max(0, x)
///
/// Currently only supports F32 tensors on the CPU.
pub fn relu_op(input: &Tensor) -> Result<Tensor, NeuraRustError> {
    let input_guard = input.read_data();

    if input_guard.device != StorageDevice::CPU || input_guard.dtype != DType::F32 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Relu op currently only supports F32 tensors on CPU.".to_string(),
        ));
    }
    let input_buffer_arc = input_guard.buffer().try_get_cpu_f32()?.clone();
    let input_slice = input_buffer_arc.as_slice();

    let mut output_data = vec![0.0f32; input_guard.numel()];
    let offset = input_guard.offset;
    for i in 0..output_data.len() {
        let val = input_slice[offset + i];
        if val > 0.0 {
            output_data[i] = val;
        }
    }

    let output_tensor = Tensor::new(output_data, input_guard.shape.clone())?;

    if input.requires_grad() {
        let backward_op = ReluBackward { input_node: Arc::clone(&input.data) };
        let mut output_write_guard = output_tensor.write_data();
        output_write_guard.grad_fn = Some(Arc::new(backward_op));
        output_write_guard.requires_grad = true;
    }

    Ok(output_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_f32_data(tensor: &Tensor) -> Vec<f32> {
        let guard = tensor.read_data();
        assert_eq!(guard.dtype, DType::F32);
        assert_eq!(guard.device, StorageDevice::CPU);
        let buffer_arc = guard.buffer().try_get_cpu_f32().unwrap().clone();
        buffer_arc.to_vec()
    }

    fn assert_tensor_eq(actual: &Tensor, expected_data: &[f32], expected_shape: &[usize]) {
        assert_eq!(actual.shape(), expected_shape, "Shape mismatch");
        let actual_data = get_f32_data(actual);
        assert_eq!(actual_data.as_slice(), expected_data, "Data mismatch");
    }

    #[test]
    fn test_relu_forward() -> Result<(), NeuraRustError> {
        let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2])?;
        let output = relu_op(&input)?;
        let expected_data = vec![0.0, 0.0, 1.0, 2.0];
        assert_tensor_eq(&output, &expected_data, &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_relu_backward() -> Result<(), NeuraRustError> {
        let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2])?;
        let _ = input.set_requires_grad(true);

        let output = relu_op(&input)?;
        assert!(output.requires_grad());
        assert!(output.grad_fn().is_some());

        let grad_output = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2])?;

        let grad_fn = output.grad_fn().unwrap();
        let grad_inputs = grad_fn.backward(&grad_output)?;

        assert_eq!(grad_inputs.len(), 1);
        let grad_input = &grad_inputs[0];
        let expected_grad_input = vec![0.0, 0.0, 0.3, 0.4];
        assert_tensor_eq(grad_input, &expected_grad_input, &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_relu_forward_zeros() -> Result<(), NeuraRustError> {
        let input = Tensor::new(vec![0.0, 0.0, 0.0], vec![3])?;
        let output = relu_op(&input)?;
        assert_tensor_eq(&output, &vec![0.0, 0.0, 0.0], &[3]);
        Ok(())
    }

    #[test]
    fn test_relu_forward_positive() -> Result<(), NeuraRustError> {
        let input = Tensor::new(vec![1.0, 10.0, 0.1], vec![3])?;
        let output = relu_op(&input)?;
        assert_tensor_eq(&output, &vec![1.0, 10.0, 0.1], &[3]);
        Ok(())
    }

    #[test]
    fn test_relu_backward_mixed() -> Result<(), NeuraRustError> {
        let input = Tensor::new(vec![-2.0, 3.0, 0.0, -5.0, 6.0], vec![5])?;
        let _ = input.set_requires_grad(true);
        let output = relu_op(&input)?;
        let grad_output = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0], vec![5])?;
        let grad_fn = output.grad_fn().unwrap();
        let grad_inputs = grad_fn.backward(&grad_output)?;
        assert_eq!(grad_inputs.len(), 1);
        assert_tensor_eq(&grad_inputs[0], &vec![0.0, 1.0, 0.0, 0.0, 1.0], &[5]);
        Ok(())
    }
} 