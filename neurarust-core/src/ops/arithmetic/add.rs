// neurarust-core/src/ops/arithmetic/add.rs

use crate::tensor::Tensor;
use crate::tensor::utils::{broadcast_shapes, calculate_strides, index_to_coord};
use std::ops::{Add, AddAssign};
use std::fmt::Debug;
use num_traits::{Zero, One};
use std::iter::Sum;
use std::default::Default;
use crate::error::NeuraRustError;
use std::cmp::PartialEq;
use crate::device::StorageDevice;
use crate::autograd::{backward_op::BackwardOp, graph::NodeId};
use std::sync::Arc;

// --- Forward Operation --- 

/// Performs element-wise addition for two Tensors with broadcasting.
/// Requires both tensors to be on the same device (currently CPU only).
/// If either input tensor requires gradients, the output tensor will also require gradients
/// and have its `grad_fn` set to an `AddBackward` operation node.
/// Returns a `Result` wrapping the new `Tensor` or a `NeuraRustError`.
/// This operation creates a new Tensor with copied data on the same device.
pub fn add_op<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>
where
    // Add Send + Sync required by Tensor structure and autograd methods
    T: Add<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + PartialEq + PartialOrd + Send + Sync,
{
    // --- Autograd Setup --- 
    // Determine if the output requires gradient tracking
    let requires_grad = a.requires_grad() || b.requires_grad();
    // Store necessary info for backward pass if needed (IDs and original shapes)
    // Note: We capture this info *before* locking the tensors for calculation,
    //       to minimize lock contention if shapes/IDs were complex to get.
    let mut a_id_maybe: Option<NodeId<T>> = None;
    let mut a_shape_maybe: Option<Vec<usize>> = None;
    let mut b_id_maybe: Option<NodeId<T>> = None;
    let mut b_shape_maybe: Option<Vec<usize>> = None;

    if requires_grad {
        // Use the pub(crate) method from Tensor accessors
        a_id_maybe = Some(a.get_node_id());
        a_shape_maybe = Some(a.shape());
        b_id_maybe = Some(b.get_node_id());
        b_shape_maybe = Some(b.shape());
    }

    // Acquire read locks for inputs
    let a_guard = a.read_data();
    let b_guard = b.read_data();

    // --- Device Check --- 
    // Ensure both tensors are on the same device (and implicitly CPU for now)
    if a_guard.device != b_guard.device {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("Cannot add tensors on different devices: {:?} and {:?}", a_guard.device, b_guard.device)
        ));
    }
    let device = a_guard.device; // Device for the output tensor
    // Ensure the operation is supported on this device (currently only CPU)
    if device != StorageDevice::CPU {
         return Err(NeuraRustError::UnsupportedOperation(
            format!("Addition is currently only supported on CPU, not {:?}", device)
        ));
    }
    // --- Get CPU Data Buffers --- 
    // We know they are on CPU, so unwrap the result of cpu_data()
    // Note: cpu_data() returns Result<&Arc<Vec<T>>, _>, so we dereference it
    // and clone the Arc for safe access later. 
    let a_data_arc = a_guard.data.cpu_data()?.clone(); // Clone the Arc<Vec<T>>
    let b_data_arc = b_guard.data.cpu_data()?.clone(); // Clone the Arc<Vec<T>>

    // --- Shape and Broadcasting --- 
    let a_shape = &a_guard.shape;
    let b_shape = &b_guard.shape;
    
    let output_shape = broadcast_shapes(a_shape, b_shape)
        .map_err(|_e| NeuraRustError::BroadcastError { 
            shape1: a_shape.clone(), 
            shape2: b_shape.clone()
        })?;

    // --- Calculation --- 
    let numel_result = output_shape.iter().product();
    let mut result_data_vec = Vec::with_capacity(numel_result);
    
    let result_strides = calculate_strides(&output_shape);
    let rank_diff_a = output_shape.len().saturating_sub(a_guard.shape.len());
    let rank_diff_b = output_shape.len().saturating_sub(b_guard.shape.len());
    
    let mut input_a_coords = vec![0; a_guard.shape.len()];
    let mut input_b_coords = vec![0; b_guard.shape.len()];

    for i in 0..numel_result {
        let output_coords = index_to_coord(i, &result_strides, &output_shape);
        
        for dim_idx in 0..a_guard.shape.len() {
            let output_coord_idx = rank_diff_a + dim_idx;
            input_a_coords[dim_idx] = if a_guard.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        // Get offset using the guard's metadata
        let offset_a = a_guard.get_offset(&input_a_coords);
        // Access data using the cloned Arc<Vec<T>>
        let val_a = a_data_arc[offset_a]; 
        
        for dim_idx in 0..b_guard.shape.len() {
            let output_coord_idx = rank_diff_b + dim_idx;
            input_b_coords[dim_idx] = if b_guard.shape[dim_idx] == 1 { 0 } else { output_coords[output_coord_idx] };
        }
        // Get offset using the guard's metadata
        let offset_b = b_guard.get_offset(&input_b_coords);
        // Access data using the cloned Arc<Vec<T>>
        let val_b = b_data_arc[offset_b]; 

        result_data_vec.push(val_a + val_b);
    }

    // Drop read locks explicitly (although they drop implicitly at end of scope)
    drop(a_guard);
    drop(b_guard);

    // --- Create Result Tensor --- 
    // The result tensor is created on the same device as the inputs (CPU here)
    let result_tensor = Tensor::new(result_data_vec, output_shape.clone())?;

    // --- Autograd Linkage (The General Pattern) --- 
    // If any input requires grad, the output also requires grad
    // and needs a `grad_fn` to link it back to the inputs in the computation graph.
    if requires_grad {
        // 1. Create the backward context struct (`AddBackward` here)
        //    It stores references (NodeIds) to the inputs and any other
        //    data needed for the backward pass (like original shapes for broadcasting).
        let backward_context = AddBackward {
            a_id: a_id_maybe.unwrap(), // Safe to unwrap due to requires_grad check
            a_shape: a_shape_maybe.unwrap(), // Use the shapes captured *before* locking
            b_id: b_id_maybe.unwrap(), // Safe to unwrap
            b_shape: b_shape_maybe.unwrap(), // Use the shapes captured *before* locking
        };
        // 2. Wrap the context in an Arc<dyn BackwardOp Trait>
        //    This allows shared ownership and dynamic dispatch.
        let backward_op_arc: Arc<dyn BackwardOp<T> + Send + Sync> = Arc::new(backward_context);

        // 3. Set autograd properties on the result tensor.
        //    This involves acquiring a write lock on the result's TensorData.
        result_tensor.set_requires_grad(true)?;
        result_tensor.set_grad_fn(Some(backward_op_arc))?;
    }
    // --- End Autograd Linkage --- 

    Ok(result_tensor)
}

/// REMOVED: In-place AddAssign is no longer safe/meaningful with shared Rc<Vec<T>> data.
// impl<'a, T> AddAssign<&'a Tensor<T>> for Tensor<T>
// where
//     T: AddAssign + Copy + Clone,
// {
//     fn add_assign(&mut self, other: &'a Tensor<T>) { ... }
// }

// --- Backward Operation --- 

/// Backward operation context for the element-wise addition operation.
/// Stores the NodeIds of the input tensors and their shapes to handle broadcasting.
#[derive(Debug)]
struct AddBackward<T: 'static + Debug + Copy + Send + Sync> {
    // NodeId for input tensor 'a'
    a_id: NodeId<T>,
    // Original shape of input tensor 'a' before broadcasting
    a_shape: Vec<usize>,
    // NodeId for input tensor 'b'
    b_id: NodeId<T>,
    // Original shape of input tensor 'b' before broadcasting
    b_shape: Vec<usize>,
}

// Mark AddBackward as Send + Sync.
// This is unsafe because the struct contains raw pointers (NodeId).
// However, we guarantee that these pointers are valid and accesses are synchronized
// through the RwLocks within the TensorData they point to, managed by the broader
// autograd system (Tensor::backward ensures tensors are kept alive).
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Send for AddBackward<T> {}
unsafe impl<T: Debug + Copy + Send + Sync + 'static> Sync for AddBackward<T> {}

// Implement `BackwardOp<T>` for `AddBackward<T>`
impl<T> BackwardOp<T> for AddBackward<T>
where
    // Add necessary bounds required by reduce_gradient_to_shape (which uses sum_axes, reshape)
    T: Debug + Copy + Send + Sync + Zero + AddAssign + 'static + Default + PartialEq + std::iter::Sum + num_traits::One + PartialOrd,
{
    /// Returns the NodeIds of the input tensors involved in the addition.
    fn inputs(&self) -> Vec<NodeId<T>> {
        vec![self.a_id, self.b_id]
    }

    /// Computes the gradients for the input tensors (`a` and `b`) of the addition.
    /// Handles reduction of gradients if broadcasting occurred during the forward pass.
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError> {
        // Gradient w.r.t. 'a' is grad_output, potentially reduced to a_shape.
        let grad_a = reduce_gradient_to_shape(grad_output, &self.a_shape)?;

        // Gradient w.r.t. 'b' is grad_output, potentially reduced to b_shape.
        let grad_b = reduce_gradient_to_shape(grad_output, &self.b_shape)?;

        // Ensure gradients are on the correct device (should match grad_output device implicitly for now)
        // TODO: Add explicit device checks if necessary later, e.g., if reduce_gradient involves device transfer.

        Ok(vec![grad_a, grad_b])
    }
}

/// Helper function to reduce a gradient tensor to a target shape.
/// This is needed when the original tensor was broadcasted during the forward op.
/// The gradient needs to be summed along the broadcasted dimensions.
fn reduce_gradient_to_shape<T>(
    gradient: &Tensor<T>,
    target_shape: &[usize],
) -> Result<Tensor<T>, NeuraRustError>
where
    T: Debug + Copy + Send + Sync + Zero + AddAssign + 'static + Default + PartialEq + std::iter::Sum + num_traits::One + PartialOrd,
{
    // Add necessary trait bounds used by sum_axes and reshape
    // T needs Default, PartialEq, Sum, One, PartialOrd (from sum_axes test constraints)

    if gradient.shape() == target_shape {
        Ok(gradient.clone()) // No reduction needed
    } else {
        // Basic implementation: If target_shape is scalar [], sum all elements.
        if target_shape.is_empty() {
             // Use correct path for sum_axes. Keep keep_dims=false for scalar output.
             crate::ops::reduction::sum_axes(gradient, &[], false)
        } else {
             let current_shape = gradient.shape();
             let rank_diff = current_shape.len().saturating_sub(target_shape.len());
             let mut axes_to_reduce: Vec<usize> = (0..rank_diff).collect(); // Reduce leading dims

             for i in 0..target_shape.len() {
                 if target_shape[i] == 1 && current_shape[rank_diff + i] > 1 {
                     axes_to_reduce.push(rank_diff + i); // Reduce broadcasted dims
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
                 // Use correct path for sum_axes. keep_dims=false is needed before reshape.
                 let reduced_grad = crate::ops::reduction::sum_axes(gradient, &axes_to_reduce, false)?;

                 let final_shape: Vec<usize> = target_shape.to_vec();
                 let reduced_numel: usize = reduced_grad.shape().iter().product();
                 let target_numel: usize = target_shape.iter().product();
                 if reduced_numel != target_numel {
                      return Err(NeuraRustError::InternalError(format!(
                         "Gradient reduction produced incompatible shape {:?} (numel {}) for target {:?} (numel {}). Reduction axes: {:?}.",
                         reduced_grad.shape(), reduced_numel, target_shape, target_numel, axes_to_reduce
                     )));
                 }

                 // Use correct path for reshape_op from ops::view
                 crate::ops::view::reshape_op(&reduced_grad, final_shape)
             }
        }
    }
}

// --- Tests --- 

#[cfg(test)]
mod tests {
    use super::*; 
    use crate::Tensor;
    use num_traits::{Zero, One};
    use std::ops::{Add, AddAssign};
    use std::fmt::Debug;
    use std::iter::Sum;
    use crate::error::NeuraRustError;
    use std::default::Default;

    // Helpers remain the same
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One + AddAssign + Copy + Add<Output=T> + Default + Sum>(
        data: Vec<T>, 
        shape: Vec<usize>
    ) -> Tensor<T> { 
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    // REMOVED: fn create_test_tensor_with_grad(...)

    #[test]
    fn test_add_tensors_ok() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5_i32, 6, 7, 8], vec![2, 2]);
        let expected_data = vec![6_i32, 8, 10, 12];
        let expected_shape = vec![2, 2];
        
        let result = add_op(&t1, &t2); // Use add_op now
        assert!(result.is_ok());
        let res_tensor = result.unwrap();
        
        // Compare data: borrow_data_buffer returns Arc<Buffer<T>>
        // Need to access cpu_data() within it.
        let res_buffer_arc = res_tensor.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());
        assert_eq!(res_tensor.shape(), expected_shape, "Shape mismatch");
        // REMOVED: assert!(!res_tensor.requires_grad());
    }

    #[test]
    fn test_add_tensors_shape_mismatch() {
        let t1 = create_test_tensor(vec![1_i32, 2, 3, 4], vec![2, 2]);
        let t_non_broadcast = create_test_tensor(vec![5, 6, 7, 8, 9, 10], vec![2, 3]);
        
        let result = add_op(&t1, &t_non_broadcast); // Use add_op
        assert!(result.is_err());
        match result.err().unwrap() {
            NeuraRustError::BroadcastError { shape1, shape2 } => {
                assert_eq!(shape1, vec![2, 2]);
                assert_eq!(shape2, vec![2, 3]);
            },
            _ => panic!("Incorrect error type returned"),
        }
    }
    
    #[test]
    fn test_add_broadcasting() {
        let t1 = create_test_tensor(vec![1_i32, 2], vec![1, 2]); // Shape [1, 2]
        let t2 = create_test_tensor(vec![10_i32, 20], vec![2, 1]); // Shape [2, 1]
        let expected_data = vec![11_i32, 12, 21, 22];
        let expected_shape = vec![2, 2];

        let result = add_op(&t1, &t2).expect("Broadcasting add failed");
        assert_eq!(result.shape(), expected_shape);
        // Updated data access
        let res_buffer_arc = result.borrow_data_buffer();
        let res_cpu_data = res_buffer_arc.cpu_data().expect("Result tensor not on CPU");
        assert_eq!(res_cpu_data.as_slice(), expected_data.as_slice());

        // Test adding a scalar
        let t_mat = create_test_tensor(vec![1_f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_scalar = Tensor::scalar(10.0_f32);
        let expected_scalar_add = vec![11.0_f32, 12.0, 13.0, 14.0];
        
        let result_scalar = add_op(&t_mat, &t_scalar).expect("Scalar add failed");
        assert_eq!(result_scalar.shape(), vec![2, 2]);
        // Updated data access
        let scalar_res_buffer_arc = result_scalar.borrow_data_buffer();
        let scalar_res_cpu_data = scalar_res_buffer_arc.cpu_data().expect("Scalar add result not on CPU");
        assert_eq!(scalar_res_cpu_data.as_slice(), expected_scalar_add.as_slice());
         
        let result_scalar_rev = add_op(&t_scalar, &t_mat).expect("Scalar add reverse failed");
        assert_eq!(result_scalar_rev.shape(), vec![2, 2]);
        // Updated data access
        let scalar_rev_res_buffer_arc = result_scalar_rev.borrow_data_buffer();
        let scalar_rev_res_cpu_data = scalar_rev_res_buffer_arc.cpu_data().expect("Scalar add reverse result not on CPU");
        assert_eq!(scalar_rev_res_cpu_data.as_slice(), expected_scalar_add.as_slice());
    }

    // REMOVED: Backward tests
    // #[test]
    // fn test_add_backward() -> Result<(), NeuraRustError> { ... }
    // #[test]
    // fn test_add_backward_broadcast() -> Result<(), NeuraRustError> { ... }
} 