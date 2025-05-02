use crate::error::NeuraRustError;
use crate::tensor_data::TensorData;
// Remove top-level unused import (used inside build_graph_dfs)
// use crate::autograd::backward_op::BackwardOp;

use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::RwLock;

/// Type alias for the unique identifier of a node in the computation graph.
/// Uses the raw pointer to the RwLock<TensorData> as a stable ID.
pub type NodeId<T> = *const RwLock<TensorData<T>>;

/// Performs a topological sort of the computation graph starting from `start_node`.
/// Returns the nodes in reverse topological order (suitable for backward pass).
/// Uses Depth First Search (DFS) and detects cycles.
pub fn topological_sort<T>(
    start_node: NodeId<T>,
) -> Result<Vec<NodeId<T>>, NeuraRustError>
where
    T: 'static + Debug + Copy + Send + Sync,
{
    let mut sorted_nodes = Vec::new();
    let mut visited = HashSet::new(); // Nodes for which DFS has *completed*
    let mut recursion_stack = HashSet::new(); // Nodes currently in the DFS recursion stack

    // Define the recursive DFS function
    fn dfs<T>(
        node_id: NodeId<T>,
        visited: &mut HashSet<NodeId<T>>,
        recursion_stack: &mut HashSet<NodeId<T>>,
        sorted_nodes: &mut Vec<NodeId<T>>,
    ) -> Result<(), NeuraRustError>
    where
        T: 'static + Debug + Copy + Send + Sync,
    {
        // 1. Check if already visited (and DFS completed for this node)
        if visited.contains(&node_id) {
            return Ok(());
        }
        // 2. Check for cycle (if node is already in the current recursion path)
        if recursion_stack.contains(&node_id) {
            return Err(NeuraRustError::CycleDetected);
        }

        // 3. Mark node as currently being visited (add to recursion stack)
        recursion_stack.insert(node_id);

        // 4. Recursively visit parents (dependencies)
        let grad_fn_arc = unsafe { (*node_id).read().expect("RwLock poisoned").grad_fn.clone() };
        if let Some(grad_fn) = grad_fn_arc {
            // Remove unused import - method call works directly on the trait object
            // use crate::autograd::backward_op::BackwardOp;
            let parents = grad_fn.inputs();
            for parent_id in parents {
                dfs(parent_id, visited, recursion_stack, sorted_nodes)?;
            }
        }

        // 5. Finished visiting this node and all its dependencies
        recursion_stack.remove(&node_id); // Remove from current recursion path
        visited.insert(node_id);          // Mark as fully visited
        sorted_nodes.push(node_id);       // Add to the sorted list (Post-order)

        Ok(())
    }

    // Start the DFS from the root node of the backward pass
    dfs(start_node, &mut visited, &mut recursion_stack, &mut sorted_nodes)?;

    // The result `sorted_nodes` is already in reverse topological order due to post-order DFS
    Ok(sorted_nodes)
}

// --- Tests --- (Placeholder - Need to create mock BackwardOps and Tensors)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward_op::BackwardOp;
    use crate::tensor::Tensor;
    use std::fmt::Debug;
    use std::sync::Arc;

    // Mock BackwardOp for testing
    #[derive(Debug)]
    struct MockOp { inputs: Vec<NodeId<f32>> }
    // UNSAFE: Only for tests! Mark MockOp as Send + Sync despite raw pointers.
    // This is acceptable because tests run sequentially and don't share across threads.
    unsafe impl Send for MockOp {}
    unsafe impl Sync for MockOp {}

    impl BackwardOp<f32> for MockOp {
        fn backward(&self, _grad_output: &Tensor<f32>) -> Result<Vec<Tensor<f32>>, NeuraRustError> {
            Ok(vec![]) // Not needed for graph test
        }
        fn inputs(&self) -> Vec<NodeId<f32>> {
            self.inputs.clone()
        }
    }

    // Helper to create a mock tensor with a grad_fn
    fn mock_tensor_with_grad_fn(op: MockOp) -> Tensor<f32> {
        let t = Tensor::scalar(0.0); // Content doesn't matter
        // The cast to Arc<dyn ... Send + Sync> works now because MockOp is marked Send+Sync
        t.set_grad_fn(Some(Arc::new(op))).unwrap();
        t
    }

    fn get_node_id<T: 'static + Debug + Copy + Send + Sync>(tensor: &Tensor<T>) -> NodeId<T> {
        Arc::as_ptr(&tensor.data)
    }

    #[test]
    fn test_topological_sort_linear() {
        let t1 = Tensor::scalar(1.0);
        let t2 = Tensor::scalar(2.0);
        let t3 = mock_tensor_with_grad_fn(MockOp { inputs: vec![get_node_id(&t1), get_node_id(&t2)] });
        let t4 = mock_tensor_with_grad_fn(MockOp { inputs: vec![get_node_id(&t3)] });

        let order = topological_sort(get_node_id(&t4)).unwrap();

        // Expected reverse topological order: t1, t2, t3, t4 (order of t1/t2 might swap)
        assert_eq!(order.len(), 4);
        assert!( (order[0] == get_node_id(&t1) && order[1] == get_node_id(&t2)) ||
                   (order[0] == get_node_id(&t2) && order[1] == get_node_id(&t1)) );
        assert_eq!(order[2], get_node_id(&t3));
        assert_eq!(order[3], get_node_id(&t4));
    }

    #[test]
    fn test_topological_sort_branch() {
        let t1 = Tensor::scalar(1.0);
        let t2 = Tensor::scalar(2.0);
        let t3 = mock_tensor_with_grad_fn(MockOp { inputs: vec![get_node_id(&t1)] });
        let t4 = mock_tensor_with_grad_fn(MockOp { inputs: vec![get_node_id(&t1), get_node_id(&t2)] });
        let t5 = mock_tensor_with_grad_fn(MockOp { inputs: vec![get_node_id(&t3), get_node_id(&t4)] });

        let order = topological_sort(get_node_id(&t5)).unwrap();

        // Expected order: t1, t2, t3, t4, t5 (or other valid reverse topological orders)
        // We check dependencies: t1 must come before t3, t4. t2 must come before t4.
        // t3 must come before t5. t4 must come before t5.
        assert_eq!(order.len(), 5);
        let pos = |id| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(get_node_id(&t1)) < pos(get_node_id(&t3)));
        assert!(pos(get_node_id(&t1)) < pos(get_node_id(&t4)));
        assert!(pos(get_node_id(&t2)) < pos(get_node_id(&t4)));
        assert!(pos(get_node_id(&t3)) < pos(get_node_id(&t5)));
        assert!(pos(get_node_id(&t4)) < pos(get_node_id(&t5)));
        assert_eq!(order[4], get_node_id(&t5)); // t5 must be last
    }

     #[test]
     fn test_topological_sort_cycle() {
        let _t1 = Tensor::scalar(1.0);
        let t2 = mock_tensor_with_grad_fn(MockOp { inputs: vec![] }); // Placeholder grad_fn
        let t3 = mock_tensor_with_grad_fn(MockOp { inputs: vec![get_node_id(&t2)] }); 

        // Manually create a cycle: t2's grad_fn depends on t3
        let op_for_t2 = MockOp { inputs: vec![get_node_id(&t3)] }; 
        t2.set_grad_fn(Some(Arc::new(op_for_t2))).unwrap();

        let result = topological_sort(get_node_id(&t3));
        assert!(matches!(result, Err(NeuraRustError::CycleDetected)));
    }
    
    #[test]
    fn test_topological_sort_single_node() {
        let t1 = Tensor::scalar(1.0); // No grad_fn
        let order = topological_sort(get_node_id(&t1)).unwrap();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], get_node_id(&t1));
    }
} 