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
    let node_ptr = Rc::as_ptr(&node.data);
    println!("[build_topo] Visiting node: {:?}", node_ptr);
    if !visited.contains(&node_ptr) {
        println!("[build_topo]  Node {:?} is new.", node_ptr);
        visited.insert(node_ptr);

        let grad_fn_clone = node.borrow_tensor_data().grad_fn.clone();

        if let Some(grad_fn) = grad_fn_clone {
            println!("[build_topo]  Node {:?} has grad_fn. Processing inputs...", node_ptr);
            let inputs_weak = grad_fn.inputs();
            for (i, input_weak) in inputs_weak.iter().enumerate() {
                println!("[build_topo]   Input {}", i);
                if let Some(input_rc) = input_weak.upgrade() {
                    let input_ptr = Rc::as_ptr(&input_rc);
                    println!("[build_topo]    Input ptr: {:?}", input_ptr);
                    let input_tensor = Tensor { data: input_rc };
                    build_topo(&input_tensor, visited, sorted_list);
                } else {
                    println!("[build_topo]    Input weak ref failed to upgrade.");
                }
            }
        } else {
             println!("[build_topo]  Node {:?} is leaf.", node_ptr);
        }
        println!("[build_topo] Adding node {:?} to sorted_list", node_ptr);
        sorted_list.push(node.clone());
    } else {
        println!("[build_topo]  Node {:?} already visited.", node_ptr);
    }
} 