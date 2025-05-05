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

// --- Tests --- 
// Link the external test file
#[cfg(test)]
#[path = "relu_test.rs"]
mod tests;

/* --- REMOVED internal tests module --- 
#[cfg(test)]
mod tests {
    // ... (contenu de l'ancien module de tests) ...
}
*/ 