// neurarust-core/src/tensor.rs
use std::cell::{Ref, RefCell, RefMut}; // Import RefCell related types
use std::fmt;
use std::rc::{Rc, Weak}; // Import Rc and Weak
use crate::autograd::BackwardOp; // Import the new trait
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use num_traits::One; // Import the One trait
use std::fmt::{Debug, Formatter, Result as FmtResult};

// --- Internal Data Structure ---

/// Holds the actual data and metadata for a tensor.
/// Uses Rc<RefCell<...>> for shared ownership and interior mutability.
// Cannot derive Debug/PartialEq because of dyn BackwardOp field
// #[derive(Debug, PartialEq)] // Ensure this is commented out or removed
pub(crate) struct TensorData<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Tensor<T>>,
    pub(crate) grad_fn: Option<Rc<dyn BackwardOp<T>>>,
    pub(crate) _ctx: Option<Weak<dyn BackwardOp<T>>>,
}

// Manual implementation of Debug
impl<T: Debug> Debug for TensorData<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("TensorData")
         .field("data", &self.data)
         .field("shape", &self.shape)
         .field("requires_grad", &self.requires_grad)
         .field("grad_defined", &self.grad.is_some())
         .field("grad_fn_defined", &self.grad_fn.is_some())
         .field("_ctx_defined", &self._ctx.is_some())
         .finish()
    }
}

// Manual implementation of PartialEq
impl<T: PartialEq> PartialEq for TensorData<T> {
    fn eq(&self, other: &Self) -> bool {
        // Compare only data, shape, and requires_grad for equality.
        // Ignore grad, grad_fn, _ctx etc.
        self.data == other.data && self.shape == other.shape && self.requires_grad == other.requires_grad
    }
}

// Eq requires that a == a always holds, which is true for our PartialEq implementation IF T: Eq.
impl<T: Eq> Eq for TensorData<T> {}

impl<T> TensorData<T> {
    // Helper to get number of elements, used internally
    pub(crate) fn numel(&self) -> usize {
        self.data.len()
    }
}

// --- Public Tensor Wrapper ---

/// The public, user-facing Tensor type.
///
/// Wraps the internal `TensorData` in an `Rc<RefCell<>>` to allow
/// shared ownership and interior mutability needed for autograd.
pub struct Tensor<T>(pub(crate) Rc<RefCell<TensorData<T>>>);

// --- Implementations for the Public Tensor Wrapper ---

impl<T> Tensor<T> {
    /// Creates a new `Tensor` from raw data and shape.
    /// Gradient tracking is disabled by default.
    /// Data is moved into the new Tensor.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length ({}) must match the product of shape dimensions ({})",
            data.len(),
            expected_len
        );
        let tensor_data = TensorData {
            data,
            shape,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            _ctx: None,
        };
        Tensor(Rc::new(RefCell::new(tensor_data)))
    }

    /// Creates a new `Tensor` with gradient tracking enabled.
    pub fn new_with_grad(data: Vec<T>, shape: Vec<usize>) -> Self {
        let tensor = Self::new(data, shape);
        tensor.0.borrow_mut().requires_grad = true;
        tensor
    }

    // --- Accessors ---

    /// Returns the shape of the tensor as a `Vec<usize>` (cloned).
    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.0.borrow().numel() // Use helper on TensorData
    }

    /// Returns a clone of the underlying data buffer as a `Vec<T>`.
    /// Requires `T: Clone`.
    pub fn data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.0.borrow().data.clone()
    }

    /// Provides temporary immutable access to the internal `TensorData` via `Ref`.
    /// The `Ref` acts like a read lock; ensure it's dropped promptly.
    /// Made `pub(crate)` because it exposes the internal `TensorData` type.
    pub(crate) fn borrow_tensor_data(&self) -> Ref<TensorData<T>> {
        self.0.borrow()
    }

    /// Provides temporary mutable access to the internal `TensorData` via `RefMut`.
    /// The `RefMut` acts like a write lock; ensure it's dropped promptly.
    /// Made `pub(crate)` because it exposes the internal `TensorData` type.
    pub(crate) fn borrow_tensor_data_mut(&self) -> RefMut<TensorData<T>> {
        self.0.borrow_mut()
    }

    /// Provides temporary immutable access to the shape via `Ref`.
    /// The `Ref` acts like a read lock; ensure it's dropped promptly.
    pub fn borrow_shape(&self) -> Ref<[usize]> {
        Ref::map(self.0.borrow(), |td| td.shape.as_slice())
    }

    /// Checks if the tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    /// Provides temporary access to the gradient tensor (`Ref`).
    /// The `Ref` acts like a read lock; ensure it's dropped promptly.
    pub fn borrow_grad(&self) -> Ref<Option<Tensor<T>>> {
        Ref::map(self.0.borrow(), |td| &td.grad)
    }

    /// Provides temporary mutable access to the gradient tensor (`RefMut`).
    /// The `RefMut` acts like a write lock; ensure it's dropped promptly.
    pub fn borrow_grad_mut(&self) -> RefMut<Option<Tensor<T>>> {
        RefMut::map(self.0.borrow_mut(), |td| &mut td.grad)
    }

    // --- Autograd methods ---

    /// Enables or disables gradient tracking for this tensor.
    /// If set to `false`, clears any existing gradient and grad_fn.
    /// Takes `&self` due to interior mutability via `RefCell`.
    pub fn set_requires_grad(&self, requires_grad: bool) {
        let mut tensor_data = self.0.borrow_mut();
        tensor_data.requires_grad = requires_grad;
        if !requires_grad {
            tensor_data.grad = None;
            tensor_data.grad_fn = None;
        }
    }

    /// Initiates the backward pass to compute gradients for the entire graph.
    ///
    /// Should be called on a scalar tensor representing the loss.
    /// Requires the element type `T` to support `One` and `Clone` for gradient initialization,
    /// and `Eq`, `Hash` for graph traversal via HashSet.
    pub fn backward(&self)
    where
        T: One + Clone + 'static, // Removed Eq + Hash. Only need One+Clone for grad init, 'static for storage.
    {
        // Check requires_grad and scalar property
        {
            let tensor_data = self.0.borrow();
            if !tensor_data.requires_grad {
                eprintln!("Warning: Called backward() on a tensor that does not require gradients.");
                return;
            }
            if tensor_data.numel() != 1 {
                // TODO: Allow backward on non-scalar tensors by providing the initial gradient
                panic!("backward() can only be called on scalar tensors (for now).");
            }
        } // Drop borrow

        // --- Topological Sort --- 
        let mut visited = HashSet::<*const RefCell<TensorData<T>>>::new();
        let mut nodes_sorted = Vec::new();

        // build_topo recursively adds nodes to nodes_sorted
        build_topo(self, &mut visited, &mut nodes_sorted);

        // --- Initialize Gradient for the Final Node --- 
        // Gradient of the loss (self) w.r.t itself is 1.
        { // New scope for mutable borrow
            let mut self_data_mut = self.0.borrow_mut();
            // Create a tensor of ones with the same shape as self
            // Requires T: One + Clone
            let grad_data = vec![T::one(); self_data_mut.numel()];
            let grad_shape = self_data_mut.shape.clone();
            // Initialize or overwrite the gradient
            self_data_mut.grad = Some(Tensor::new(grad_data, grad_shape));
        }

        // --- Backward Pass through Sorted Nodes --- 
        println!("Backward: Processing {} nodes in topological order.", nodes_sorted.len());
        for node in nodes_sorted.iter().rev() { // Process in reverse topological order
            let node_data = node.0.borrow(); // Immutable borrow for reading grad_fn and grad
            
            // Option 1: Clone grad and pass it (safer borrow-wise)
            let current_grad = node_data.grad.clone(); // Clone Option<Tensor<T>>
            let grad_fn = node_data.grad_fn.clone(); // Clone Rc<dyn BackwardOp<T>>
            
            // Drop the borrow on node_data before calling backward potentially on the same node?
            // Unlikely needed if we pass cloned grad, but good practice.
            drop(node_data);

            if let Some(grad_fn) = grad_fn {
                if let Some(grad) = current_grad { // Use the cloned grad
                     println!("  -> Calling backward on grad_fn for node..."); // Avoid borrow in format
                    grad_fn.backward(&grad); // Pass the computed gradient of *this* node
                } else {
                     // This should ideally not happen for nodes in the graph after init
                     eprintln!("  -> Skipping node - grad is None unexpectedly.");
                }
            } // else: Leaf node or node without grad_fn, do nothing
        }
        println!("Backward pass complete.");
    }

    // Internal helper for graph building using weak refs to avoid cycles.
    pub(crate) fn get_weak_ref(&self) -> Weak<RefCell<TensorData<T>>> {
        Rc::downgrade(&self.0)
    }

     // Helper for indexing (temporary replacement for std::ops::Index)
     // Needed by naive matmul. Returns owned value.
     pub(crate) fn get_val(&self, index: [usize; 2]) -> T
     where T: Copy
     {
         let td = self.0.borrow();
         assert_eq!(td.shape.len(), 2, "get_val with [row, col] requires a 2D tensor.");
         let rows = td.shape[0];
         let cols = td.shape[1];
         let row_idx = index[0];
         let col_idx = index[1];
         assert!(row_idx < rows, "Row index {} out of bounds for shape {:?}. Needed by get_val.", row_idx, td.shape);
         assert!(col_idx < cols, "Column index {} out of bounds for shape {:?}. Needed by get_val.", col_idx, td.shape);
         let flat_index = row_idx * cols + col_idx;
         td.data[flat_index] // Returns a copy
     }
}

// --- Trait Implementations for the Tensor Wrapper ---

impl<T> Clone for Tensor<T> {
    /// Clones the `Tensor` wrapper (bumps the `Rc` count).
    fn clone(&self) -> Self {
        Tensor(Rc::clone(&self.0))
    }
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    /// Formats the `Tensor` for display.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Access fields via borrow()
        let td = self.0.borrow();
        f.debug_struct("Tensor")
            .field("data", &td.data)
            .field("shape", &td.shape)
            .field("requires_grad", &td.requires_grad)
            .field("grad_defined", &td.grad.is_some())
            .field("grad_fn_defined", &td.grad_fn.is_some())
            .finish()
    }
}

/// PartialEq for Tensor is based on Rc pointer equality, consistent with Hash.
/// Two Tensors are considered equal only if they point to the exact same data allocation.
impl<T> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

// Eq for Tensor follows from PartialEq (pointer equality is reflexive, symmetric, transitive).
// No dependency on T needed here.
impl<T> Eq for Tensor<T> {}

impl<T> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

// --- Tests for the Public Tensor Wrapper ---
#[cfg(test)]
mod tests {
    use super::*; // Import the public Tensor wrapper
    use num_traits::Zero; // Import necessary traits for tests
    

    // Helper to create a simple tensor for testing
    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }

    // Helper to create a tensor that requires grad
    // Remove Eq bound, not needed for floats, Tensor<T> Eq is ptr based.
    fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // Borrow the whole TensorData struct
        let td = tensor.borrow_tensor_data(); 
        assert_eq!(td.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(td.shape, vec![2, 2]);
        assert!(!td.requires_grad);
        assert!(td.grad.is_none());
        assert!(td.grad_fn.is_none());
        assert!(td._ctx.is_none());
        // drop(td); // Ref is dropped automatically here
    }

    #[test]
    fn test_tensor_equality() {
        let t1 = create_test_tensor(vec![1, 2], vec![2]);
        let t2 = create_test_tensor(vec![1, 2], vec![2]);
        let t3 = create_test_tensor(vec![3, 4], vec![2]);
        let t4 = create_test_tensor(vec![1, 2], vec![1, 2]);

        // PartialEq now compares Rc pointers, so t1 != t2
        assert_ne!(t1, t2);
        assert_ne!(t1, t3);
        assert_ne!(t1, t4);

        // Check content equality explicitly where needed
        assert_eq!(t1.data(), t2.data());
        assert_eq!(t1.shape(), t2.shape());

        // Check pointer equality for clones
        let t1_clone = t1.clone();
        assert_eq!(t1, t1_clone); // Pointer equality holds for clones
    }

    #[test]
    fn test_tensor_hash_eq_for_set() {
        // Specify the type T = i32 explicitly
        let t1: Tensor<i32> = create_test_tensor(vec![1, 2], vec![2]);
        let t2: Tensor<i32> = create_test_tensor(vec![1, 2], vec![2]); // Same content, different Rc
        let t3: Tensor<i32> = t1.clone(); // Same Rc

        // Specify the HashSet type argument
        let mut set: HashSet<Tensor<i32>> = HashSet::new();
        
        // i32 implements Eq, so Tensor<i32> implements Eq + Hash
        assert!(set.insert(t1.clone())); // insert should now work and return true

        assert!(set.contains(&t1));
        assert!(set.contains(&t3)); // Eq based on Rc pointer -> true
        assert!(!set.contains(&t2)); // Eq based on Rc pointer -> false
        assert_eq!(set.len(), 1);

        // Insert the second tensor (different Rc)
        assert!(set.insert(t2.clone())); // Should return true as it's a new element (different hash)
        assert!(set.contains(&t2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_backward_basic() {
        // Should work with floats now (no Eq constraint on helper)
        // Prefix x and y with _ as they are unused in this basic test
        let _x = create_test_tensor_with_grad(vec![2.0], vec![1]);
        let _y = create_test_tensor_with_grad(vec![3.0], vec![1]);

        // Mock a simple operation: z = x * y
        // Need a proper grad_fn for a real test, create dummy tensor for now.
        let z = Tensor::new_with_grad(vec![6.0], vec![1]); // Dummy result
        
        // If Mul op existed and set grad_fn:
        // let z = &_x * &_y;
        // assert!(z.0.borrow().grad_fn.is_some());

        z.backward(); // Call the backward function

        // Assertions require actual gradient computation via grad_fn
        // assert_eq!(x.borrow_grad().unwrap().borrow_data(), &[3.0]);
        // assert_eq!(y.borrow_grad().unwrap().borrow_data(), &[2.0]);
    }

    #[test]
    fn test_add() {
        let t1 = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5, 6, 7, 8], vec![2, 2]);
        // Use references for the Add impl
        let result = &t1 + &t2;
        let expected_data = vec![6, 8, 10, 12];
        // Access data via borrow_tensor_data
        assert_eq!(result.borrow_tensor_data().data, expected_data);
        assert_eq!(result.borrow_tensor_data().shape, vec![2, 2]);
    }

    #[test]
    fn test_add_assign() {
        let mut t1 = create_test_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let t2 = create_test_tensor(vec![5, 6, 7, 8], vec![2, 2]);
        // Use reference for the rhs of AddAssign
        t1 += &t2;
        let expected_data = vec![6, 8, 10, 12];
        // Access data via borrow_tensor_data
        assert_eq!(t1.borrow_tensor_data().data, expected_data);
        assert_eq!(t1.borrow_tensor_data().shape, vec![2, 2]);
    }
}

// --- Standalone Helper Function for Topological Sort ---
// Moved outside impl block, takes HashSet by value now? No, needs mut ref.
// Only need Clone for node.clone(), 'static for storage.
fn build_topo<T: Clone + 'static>(
    node: &Tensor<T>, 
    visited: &mut HashSet<*const RefCell<TensorData<T>>>,
    sorted_list: &mut Vec<Tensor<T>>
) {
    let node_ptr = Rc::as_ptr(&node.0);
    if visited.insert(node_ptr) { 
        // Clone grad_fn Rc to avoid holding borrow across recursive calls
        let grad_fn = node.0.borrow().grad_fn.clone(); 
        if let Some(grad_fn_rc) = grad_fn {
            // Get inputs using the trait method
            for input_weak in grad_fn_rc.inputs() {
                if let Some(input_rc) = input_weak.upgrade() {
                    // Reconstruct Tensor wrapper to pass to build_topo
                    let input_tensor = Tensor(input_rc);
                    build_topo(&input_tensor, visited, sorted_list);
                }
                // Optional: Warn if upgrade fails?
            }
        }
        // Add node to sorted list *after* visiting its children (inputs)
        sorted_list.push(node.clone());
    }
}