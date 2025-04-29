use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::cell::RefCell;
use std::rc::Weak;
use std::collections::HashMap;
use std::rc::Rc;
use std::ops::AddAssign;
use std::fmt::Debug;
use num_traits::Zero;

pub mod graph;

/// Trait for operations that support backward pass (gradient calculation).
pub trait BackwardOp<T>: std::fmt::Debug {
    /// Performs the backward pass, calculating gradients for the inputs.
    /// Takes the upstream gradient and a mutable reference to the gradient map.
    fn backward(&self, upstream_grad: &Tensor<T>, gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>);

    /// Returns weak references to the input tensors.
    fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>>;
}

// Concrete BackwardOp implementations will live alongside their corresponding
// forward operations (e.g., in neurarust-core/src/ops/arithmetic.rs).
// Remove the placeholder AddBackward struct from here. 

// --- Autograd Helper Functions ---

/// Accumulates the local gradient into the gradient map for the given input tensor.
/// If a gradient already exists for the tensor, it adds the local gradient to it.
/// Otherwise, it inserts the local gradient into the map.
/// 
/// # Arguments
/// * `gradients` - The HashMap storing accumulated gradients, keyed by tensor data pointers.
/// * `input_weak_ref` - A weak reference to the input tensor data whose gradient is being computed.
/// * `local_gradient` - The gradient computed by the current backward operation for this input.
pub(crate) fn accumulate_gradient<T>(
    gradients: &mut HashMap<*const RefCell<TensorData<T>>, Tensor<T>>,
    input_weak_ref: &Weak<RefCell<TensorData<T>>>,
    local_gradient: Tensor<T>,
)
where
    T: AddAssign + Clone + Debug + Zero + Copy + 'static, 
{
    if let Some(input_rc) = input_weak_ref.upgrade() {
        let input_ptr = Rc::as_ptr(&input_rc);
        
        // Clone local_gradient data *before* accessing the HashMap entry
        // to avoid potential borrow conflicts if local_gradient somehow depends
        // on what's being modified.
        // Note: accumulate_gradient takes ownership of local_gradient now.
        let local_gradient_clone = local_gradient.clone(); 

        gradients.entry(input_ptr)
            .and_modify(|existing_grad| {
                assert_eq!(existing_grad.shape(), local_gradient_clone.shape(), 
                           "Gradient shape mismatch during accumulation: existing {:?} vs new {:?}",
                           existing_grad.shape(), local_gradient_clone.shape());
                
                // Manual element-wise addition using the clone
                let mut existing_data = existing_grad.borrow_tensor_data_mut();
                // Borrow data from the clone immutably
                let local_gradient_data = local_gradient_clone.borrow_tensor_data(); 
                
                existing_data.data.iter_mut()
                    .zip(local_gradient_data.data.iter())
                    .for_each(|(e, &l)| *e += l); 
            })
            .or_insert(local_gradient); // Insert the original local_gradient (moves ownership)
    }
} 