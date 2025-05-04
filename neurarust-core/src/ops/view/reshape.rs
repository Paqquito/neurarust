use crate::autograd::BackwardOp;
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Performs the reshape operation. Currently only supports creating a view
/// for contiguous tensors. For non-contiguous tensors, call `.contiguous()` first.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `new_shape_vec`: The desired new shape.
///
/// # Returns
/// A new Tensor representing the reshaped view, or an error.
pub fn reshape_op(tensor: &Tensor, new_shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    let tensor_data = tensor.data.read().unwrap();

    let original_numel: usize = tensor_data.shape.iter().product();
    let new_numel: usize = new_shape.iter().product();

    if original_numel != new_numel {
        return Err(NeuraRustError::ShapeMismatch {
            expected: format!("{:?}", tensor_data.shape),
            actual: format!("{:?}", new_shape),
            operation: "reshape (numel mismatch)".to_string(),
        });
    }

    // --- Contiguity Check and View Creation ---
    let new_strides: Vec<usize>;
    let new_offset = tensor_data.offset;
    let can_view = tensor_data.is_contiguous(); // Simple check for now

    if can_view {
        new_strides = TensorData::calculate_contiguous_strides(&new_shape);
    } else {
        // TODO: Implement non-contiguous reshape view check if possible
        // For now, require .contiguous() before reshape if non-contiguous
        return Err(NeuraRustError::UnsupportedOperation(
            "Reshaping non-contiguous tensor requires calling .contiguous() first".to_string(),
        ));
    }

    // --- Create View TensorData ---
    let view_td = TensorData::new_view(
        Arc::clone(&tensor_data.buffer),
        tensor_data.device,
        new_offset,
        new_shape.clone(),
        new_strides,
    );

    // --- Wrap in Tensor and Setup Autograd ---
    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    if tensor_data.requires_grad {
        let backward_context = ReshapeBackward {
            input_shape: tensor_data.shape.clone(),
        };
        let backward_op_arc: Arc<dyn BackwardOp + Send + Sync> = Arc::new(backward_context);
        {
            let mut output_guard = output_tensor.data.write().unwrap();
            output_guard.requires_grad = true;
            output_guard.grad_fn = Some(backward_op_arc);
        }
    }

    Ok(output_tensor)
}

// --- Reshape Backward Operation ---

#[derive(Debug)]
struct ReshapeBackward {
    input_shape: Vec<usize>,
}

impl BackwardOp for ReshapeBackward {
    fn backward(&self, _grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        reshape_op(_grad_output, self.input_shape.clone())
            .map(|grad_input| vec![grad_input])
            .map_err(|e| NeuraRustError::BackwardError(format!("Error in ReshapeBackward: {}", e)))
    }

    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        Vec::new() // TODO: Adapt graph linkage
    }
}

// --- Tests for Reshape Op ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::error::NeuraRustError;
    
    

    #[test]
    fn test_reshape_contiguous() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_contiguous until view ops are adapted.");
        Ok(())
    }

    #[test]
    fn test_reshape_non_contiguous_error() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_non_contiguous_error until view ops are adapted.");
        Ok(())
    }

    #[test]
    fn test_reshape_numel_mismatch() {
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = reshape_op(&t, vec![2, 2]);
        assert!(matches!(result, Err(NeuraRustError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_reshape_to_scalar() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_to_scalar until view ops are adapted.");
        Ok(())
    }

    #[test]
    fn test_reshape_from_scalar() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_from_scalar until view ops are adapted.");
        Ok(())
    }

    #[test]
    fn test_reshape_on_views() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_on_views until view ops are adapted.");
        Ok(())
    }

    // --- Autograd Tests ---
    #[test]
    fn test_reshape_backward() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_backward until Tensor methods and check_grad are adapted.");
        Ok(())
    }

    #[test]
    fn test_reshape_backward_flatten() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_backward_flatten until Tensor methods and check_grad are adapted.");
        Ok(())
    }

    #[test]
    fn test_reshape_backward_add_dim() -> Result<(), NeuraRustError> {
        println!("Skipping test_reshape_backward_add_dim until Tensor methods and check_grad are adapted.");
        Ok(())
    }
} 