use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::cell::RefCell;
use std::rc::Weak;
use std::collections::HashMap;
use std::rc::Rc;
use std::ops::{AddAssign, Add};
use std::fmt::Debug;
use num_traits::Zero;
use crate::ops::arithmetic::add;
use std::iter::Sum;
use num_traits::One;

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
    // Use the bounds required by `ops::arithmetic::add` and the logic inside
    T: Add<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static,
{
    if let Some(input_rc) = input_weak_ref.upgrade() {
        let input_ptr = Rc::as_ptr(&input_rc);
        gradients.entry(input_ptr)
            .and_modify(|existing_grad| { 
                // Use the fallible add operation
                let sum_result = add(existing_grad, &local_gradient);
                match sum_result {
                    Ok(summed_grad) => {
                         *existing_grad = summed_grad; 
                    },
                    Err(e) => {
                         panic!("Error adding gradients during accumulation: {:?}. Existing shape {:?}, New shape {:?}", 
                               e, existing_grad.shape(), local_gradient.shape());
                    }
                }
            })
            .or_insert(local_gradient);
    }
} 