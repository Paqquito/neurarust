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
    // Accéder directement au champ `data` (Rc) pour obtenir le pointeur
    let node_ptr = Rc::as_ptr(&node.data);
    if !visited.contains(&node_ptr) {
        visited.insert(node_ptr);

        // Clone the Rc for grad_fn to avoid holding borrow across recursive call
        let grad_fn_clone = node.borrow_tensor_data().grad_fn.clone();

        if let Some(grad_fn) = grad_fn_clone {
            // Get weak references to inputs from the BackwardOp
            let inputs_weak = grad_fn.inputs();
            for input_weak in inputs_weak {
                // Attempt to upgrade the weak reference to a strong one (Rc)
                if let Some(input_rc) = input_weak.upgrade() {
                    // Créer un Tensor en utilisant le Rc<RefCell<TensorData<T>>>
                    let input_tensor = Tensor { data: input_rc };
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