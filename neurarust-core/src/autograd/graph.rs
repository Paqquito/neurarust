use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use std::collections::HashSet;
use std::rc::Rc;
use std::cell::RefCell;

/// Recursively builds a topological sort of the computation graph.
/// Used by `backward()` to process nodes in the correct order.
/// Uses `HashSet` based on `Tensor`'s `Hash` impl (pointer address).
/// Made `pub(crate)` as it's an internal detail of the autograd system.
pub(crate) fn build_topo<T: Clone + 'static>(
    node: &Tensor<T>,
    visited: &mut HashSet<*const RefCell<TensorData<T>>>,
    sorted_list: &mut Vec<Tensor<T>>
) {
    let node_ptr = Rc::as_ptr(&node.0);
    if !visited.contains(&node_ptr) {
        visited.insert(node_ptr);

        // Clone the Rc for grad_fn to avoid holding borrow across recursive call
        let grad_fn_clone = node.0.borrow().grad_fn.clone();

        if let Some(grad_fn) = grad_fn_clone {
            // Get weak references to inputs from the BackwardOp
            let inputs_weak = grad_fn.inputs();
            for input_weak in inputs_weak {
                // Attempt to upgrade the weak reference to a strong one (Rc)
                if let Some(input_rc) = input_weak.upgrade() {
                    // Create a temporary Tensor wrapper around the upgraded Rc
                    let input_tensor = Tensor(input_rc);
                    // Recurse
                    build_topo(&input_tensor, visited, sorted_list);
                }
                // If upgrade fails, the input tensor was dropped, which is expected in some cases
            }
        }
        // Add node to the sorted list *after* visiting all its children
        sorted_list.push(node.clone());
    }
} 