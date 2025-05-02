use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

/// Performs the permute operation, creating a view with reordered dimensions.
///
/// # Arguments
/// * `tensor`: The input tensor.
/// * `dims`: A slice representing the desired permutation of dimensions.
///           Must contain each dimension index from 0 to rank-1 exactly once.
///
/// # Returns
/// A new Tensor representing the permuted view, or an error.
pub(crate) fn permute_op<T>(
    tensor: &Tensor<T>,
    dims: &[usize],
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Default
        + Send
        + Sync
        + 'static
        + Debug
        + Copy
        + Zero
        + AddAssign
        + PartialEq
        + PartialOrd
        + Sum
        + One,
{
    // --- Autograd Setup ---
    let requires_grad = tensor.requires_grad();
    let mut input_id_maybe: Option<NodeId<T>> = None;
    let mut inverse_dims_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        input_id_maybe = Some(tensor.get_node_id());
        let rank = tensor.shape().len();
        if rank > 0 {
            let mut inverse_dims = vec![0; rank];
            for (i, &dim) in dims.iter().enumerate() {
                if dim < rank {
                    inverse_dims[dim] = i;
                } else {
                    return Err(NeuraRustError::InvalidPermutation { dims: dims.to_vec(), rank });
                }
            }
            inverse_dims_maybe = Some(inverse_dims);
        } else if !dims.is_empty() {
            // If rank is 0 but dims is not empty, it's an error handled below,
            // so no need to calculate inverse_dims.
        } else {
            inverse_dims_maybe = Some(vec![]);
        }
    }
    // --- End Autograd Setup ---

    // 1. Acquire read lock
    let tensor_data_arc = Arc::clone(&tensor.data);
    let guard = tensor_data_arc.read().map_err(|_| {
        NeuraRustError::InternalError(
            "Failed to acquire read lock on TensorData for permute".to_string(),
        )
    })?;

    let rank = guard.shape.len();

    // Add check for scalar tensor (rank 0)
    if rank == 0 {
        if !dims.is_empty() {
            return Err(NeuraRustError::DimensionMismatch {
                expected: 0,
                actual: dims.len(),
            });
        } else {
             // Permute d'un scalaire avec [] est une opération valide,
             // elle retourne une copie du scalaire.
             // Libérer le verrou avant de cloner.
             drop(guard);
             // Créer un nouveau TensorData clonant les propriétés mais avec une nouvelle RwLock
             // Note: Tensor::clone() gère cela.
             let result = tensor.clone(); // Cloner le tenseur scalaire

             // Si le tenseur original nécessite des gradients, le clone aussi,
             // mais sans grad_fn car c'est une opération d'identité pour le backward.
             // NOTE: Si on voulait propager le graphe ici, il faudrait un IdentityBackward.
             // Pour l'instant, on suppose que clone() gère correctement requires_grad,
             // mais il ne mettra pas de grad_fn.
             // Le backward de permute([]) sur un scalaire serait trivial (retourner le grad_output).

             // On ne lie pas à l'autograd ici car c'est une copie.
             // Le clone devrait copier le flag requires_grad.
             // Si l'original requires_grad, le résultat le fera aussi, mais sans grad_fn.

             return Ok(result);
        }
    }

    // 2. Validate permutation dimensions (original check, now only for rank > 0)
    if dims.len() != rank {
        return Err(NeuraRustError::DimensionMismatch {
            expected: rank,
            actual: dims.len(),
        });
    }
    let mut seen = vec![false; rank];
    for &dim in dims {
        if dim >= rank || seen[dim] {
            return Err(NeuraRustError::InvalidPermutation {
                dims: dims.to_vec(),
                rank,
            });
        }
        seen[dim] = true;
    }

    // 3. Calculate new shape and strides
    let mut new_shape = Vec::with_capacity(rank);
    let mut new_strides = Vec::with_capacity(rank);
    for &new_dim_index in dims {
        new_shape.push(guard.shape[new_dim_index]);
        new_strides.push(guard.strides[new_dim_index]);
    }

    // 4. Get other necessary info
    let buffer_arc = Arc::clone(&guard.data);
    let device = guard.device;
    let offset = guard.offset;

    drop(guard);

    // 5. Create new TensorData using new_view
    let new_td = TensorData::new_view(buffer_arc, device, offset, new_shape, new_strides);

    // 6. Wrap in Tensor
    let new_tensor = Tensor {
        data: Arc::new(RwLock::new(new_td)),
    };

    // --- Autograd Linkage ---
    if requires_grad {
        let inverse_dims = inverse_dims_maybe.ok_or_else(|| NeuraRustError::InternalError("Missing inverse permutation for permute backward pass".to_string()))?;

        let backward_context = PermuteBackward {
            input_id: input_id_maybe.unwrap(),
            inverse_dims,
            _phantom: std::marker::PhantomData,
        };

        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        new_tensor.set_requires_grad(true)?;
        new_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage ---

    Ok(new_tensor)
}


// --- Permute Backward Operation ---

#[derive(Debug)]
struct PermuteBackward<T: 'static + Debug + Copy + Send + Sync> {
    input_id: NodeId<T>,
    inverse_dims: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for PermuteBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for PermuteBackward<T> {}

impl<T> BackwardOp<T> for PermuteBackward<T>
where
    T: Default
        + Send
        + Sync
        + 'static
        + Debug
        + Copy
        + Zero
        + AddAssign
        + PartialEq
        + PartialOrd
        + Sum
        + One,
{
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.input_id]
    }

    /// Computes the gradient for the input tensor of the permute operation.
    /// This involves applying the *inverse* permutation to the incoming gradient.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        let input_grad = permute_op(grad_output, &self.inverse_dims)?;
        Ok(vec![input_grad])
    }
}

// --- Tests for Permute Op ---
#[cfg(test)]
mod tests {
    use super::*; // Importe permute_op, etc.
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::utils::testing::create_test_tensor_with_grad;
    
    

    #[test]
    fn test_permute_backward() {
        let input_data = create_test_tensor_with_grad(
            (1..=24).map(|x| x as f64).collect(),
            vec![2, 3, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![4, 2, 3]).unwrap();

        // Calculate analytical gradient
        let dims = vec![2, 0, 1];
        let output = input_data.permute(&dims).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is permute(output_grad_val) with inverse dims
        let mut inverse_dims = vec![0; dims.len()];
        for (i, &dim) in dims.iter().enumerate() {
            inverse_dims[dim] = i;
        }
        let expected_grad = permute_op(&output_grad_val, &inverse_dims).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

    #[test]
    fn test_permute_backward_identity() {
        let permute_fn = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            inputs[0].permute(&[0, 1])
        };
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![2, 2]).unwrap();
        let result = check_grad(permute_fn, &[input_data], &output_grad_val, 1e-5, 1e-7);
        assert!(result.is_ok(), "Gradient check failed for identity permute: {:?}", result.err());
    }

} 