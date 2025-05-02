use crate::error::NeuraRustError;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use crate::autograd::BackwardOp; // Needed for op.inputs()

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Type alias for the unique identifier of a node in the computation graph.
/// It's the raw pointer to the RwLock containing the TensorData.
pub type NodeIdType<T> = *const RwLock<TensorData<T>>;

/// Performs a topological sort of the computation graph starting from a given tensor.
/// Returns the nodes in reverse topological order (dependencies first), suitable for backpropagation.
/// Detects cycles in the graph.
///
/// # Arguments
/// * `start_tensor`: The tensor from which to start the backward traversal.
///
/// # Returns
/// A `Result` containing a `Vec` of `NodeIdType<T>` in reverse topological order,
/// or a `NeuraRustError::CycleDetected` if a cycle is found.
pub fn topological_sort<T: 'static + Debug + Copy>(
    start_tensor: &Tensor<T>,
) -> Result<Vec<NodeIdType<T>>, NeuraRustError> {
    let mut visited: HashSet<NodeIdType<T>> = HashSet::new();
    let mut visiting: HashSet<NodeIdType<T>> = HashSet::new();
    let mut sorted_nodes: Vec<NodeIdType<T>> = Vec::new();

    let start_node_id = start_tensor.id_ptr();

    dfs(start_node_id, &mut visited, &mut visiting, &mut sorted_nodes)?;

    Ok(sorted_nodes)
}

/// Recursive Depth First Search helper for topological sort.
/// Uses unsafe block to dereference NodeId pointers.
fn dfs<T: 'static + Debug + Copy>(
    node_id: NodeIdType<T>,
    visited: &mut HashSet<NodeIdType<T>>,
    visiting: &mut HashSet<NodeIdType<T>>,
    sorted_nodes: &mut Vec<NodeIdType<T>>,
) -> Result<(), NeuraRustError> {
    // Mark node as currently being visited (for cycle detection)
    visiting.insert(node_id);

    let inputs: Option<Vec<NodeIdType<T>>> = {
        // SAFETY: We are dereferencing a raw pointer obtained from Arc::as_ptr.
        // This pointer is assumed to be valid for the duration of the sort,
        // as the Tensors involved in the graph are expected to be kept alive
        // (e.g., by the main backward call context).
        // Incorrect management of Tensor lifetimes could lead to dangling pointers here.
        let tensor_data_lock = unsafe { &*node_id };
        let tensor_data_guard = tensor_data_lock.read().map_err(|_| {
            NeuraRustError::InternalError("RwLock poisoned during topological sort".to_string())
        })?;

        // Clone the grad_fn Arc if it exists
        let grad_fn_arc = tensor_data_guard.grad_fn.clone();

        // Get input IDs if grad_fn exists
        grad_fn_arc.map(|op| op.inputs())
        // Guard is dropped here implicitly when grad_fn_arc goes out of scope or below
    };

    if let Some(input_ids) = inputs {
        // Recursively visit dependencies (inputs to the forward op)
        for input_id in input_ids {
            if visiting.contains(&input_id) {
                // Cycle detected!
                return Err(NeuraRustError::CycleDetected);
            }
            if !visited.contains(&input_id) {
                // Visit unvisited dependencies
                dfs(input_id, visited, visiting, sorted_nodes)?;
            }
        }
    }

    // Finished visiting this node and its dependencies
    visiting.remove(&node_id); // Remove from current path
    visited.insert(node_id); // Mark as fully visited
    sorted_nodes.push(node_id); // Add to the sorted list (reverse topological order)

    Ok(())
}

// --- Tests --- (Placeholder - Need to create mock BackwardOps and Tensors)
#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::tensor::Tensor;
    // use crate::autograd::BackwardOp;
    // use crate::error::NeuraRustError;
    // use std::sync::{Arc, RwLock};
    // use std::fmt::Debug;

    // Define Mock Tensor Creation and Mock BackwardOp here for testing
    // ...

    // #[test]
    // fn test_linear_graph() {
    //     // Create Tensors t1, t2, t3
    //     // Create Ops op1 (t1 -> t2), op2 (t2 -> t3)
    //     // Set grad_fn for t2 and t3
    //     // let sorted = topological_sort(&t3).unwrap();
    //     // assert_eq!(sorted, vec![t1.id_ptr(), t2.id_ptr(), t3.id_ptr()]); // Example assertion
    // }

    // #[test]
    // fn test_branching_graph() {
    //     // t1, t2 -> t3 (via op1)
    //     // t3 -> t4 (via op2)
    //     // let sorted = topological_sort(&t4).unwrap();
    //     // Order of t1, t2 might vary, but t3 must come after t1/t2, and t4 after t3.
    // }

    // #[test]
    // fn test_shared_node_graph() {
    //     // t1 -> t2 (op1)
    //     // t1 -> t3 (op2)
    //     // t2, t3 -> t4 (op3)
    //     // let sorted = topological_sort(&t4).unwrap();
    //     // t1 must come before t2 and t3. t2, t3 must come before t4.
    // }

    // #[test]
    // fn test_cycle_detection() {
    //     // t1 -> t2 (op1)
    //     // t2 -> t1 (op2) // Create cycle
    //     // let result = topological_sort(&t2);
    //     // assert!(matches!(result, Err(NeuraRustError::CycleDetected)));
    // }
} 