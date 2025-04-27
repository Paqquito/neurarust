// neurarust-core/src/tensor.rs
use std::cell::{Ref, RefCell, RefMut}; // Import RefCell related types
use std::fmt;
use std::rc::{Rc, Weak}; // Import Rc and Weak
use crate::autograd::BackwardOp; // Import the new trait
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use num_traits::{One, Zero}; // Import the One and Zero traits
use crate::tensor_data::TensorData; // Use the new module

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

    /// Creates a new tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero + Clone, // Requires Zero trait and Clone for vec! macro
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![T::zero(); numel];
        Tensor::new(data, shape)
    }

    /// Creates a new tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Clone, // Requires One trait and Clone
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![T::one(); numel];
        Tensor::new(data, shape)
    }

    /// Creates a tensor of ones with the same shape as the given tensor.
    pub fn ones_like(other: &Tensor<T>) -> Self
    where
        T: One + Clone,
    {
        let shape = other.shape();
        let numel = other.numel();
        let data = vec![T::one(); numel];
        Tensor::new(data, shape)
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

    /// Returns a clone of the gradient tensor, if it exists.
    /// Returns `None` if no gradient has been computed or stored.
    pub fn grad(&self) -> Option<Tensor<T>>
    where
        T: Clone, // Needed to clone the TensorData inside Option<Tensor<T>>
    {
        self.0.borrow().grad.clone()
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

     /// Provides temporary access to the grad_fn (`Ref`).
     pub fn grad_fn(&self) -> Ref<Option<Rc<dyn BackwardOp<T>>>> {
        Ref::map(self.0.borrow(), |td| &td.grad_fn)
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
        T: One + Clone + 'static,
    {
        {
            let tensor_data = self.0.borrow();
            if !tensor_data.requires_grad {
                eprintln!("Warning: Called backward() on a tensor that does not require gradients.");
                return;
            }
            if tensor_data.numel() != 1 {
                panic!("backward() can only be called on scalar tensors (for now).");
            }
        }

        let mut visited = HashSet::<*const RefCell<TensorData<T>>>::new();
        let mut nodes_sorted = Vec::new();

        // Call build_topo from the autograd::graph module
        crate::autograd::graph::build_topo(self, &mut visited, &mut nodes_sorted);

        { 
            let mut self_data_mut = self.0.borrow_mut();
            let grad_data = vec![T::one(); self_data_mut.numel()];
            let grad_shape = self_data_mut.shape.clone();
            self_data_mut.grad = Some(Tensor::new(grad_data, grad_shape));
        }

        println!("Backward: Processing {} nodes in topological order.", nodes_sorted.len());
        for node in nodes_sorted.iter().rev() {
            let node_data = node.0.borrow(); 
            let current_grad = node_data.grad.clone();
            let grad_fn = node_data.grad_fn.clone();
            drop(node_data);

            if let Some(grad_fn) = grad_fn {
                if let Some(grad) = current_grad { 
                     println!("  -> Calling backward on grad_fn for node...");
                    grad_fn.backward(&grad);
                } else {
                     eprintln!("  -> Skipping node - grad is None unexpectedly.");
                }
            }
        }
        println!("Backward pass complete.");
    }

    // Internal helper for graph building using weak refs to avoid cycles.
    pub(crate) fn get_weak_ref(&self) -> Weak<RefCell<TensorData<T>>> {
        Rc::downgrade(&self.0)
    }
}

// --- Trait Implementations for Tensor ---

impl<T> Clone for Tensor<T> {
    /// Cloning a `Tensor` creates a new `Tensor` that shares the same underlying `TensorData`.
    /// This is a shallow clone due to `Rc`.
    fn clone(&self) -> Self {
        Tensor(Rc::clone(&self.0))
    }
}

// Debug implementation relies on TensorData's Debug
impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    /// Formats the Tensor for display, showing its data and shape.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Borrow immutably to access data and shape
        let tensor_data = self.0.borrow();
        write!(f, "Tensor(data={:?}, shape={:?}, requires_grad={})",
               tensor_data.data, tensor_data.shape, tensor_data.requires_grad)
        // Optionally, add grad info if present:
        // .field("grad", &tensor_data.grad)
        // .field("grad_fn", &tensor_data.grad_fn.is_some()) // Only show if grad_fn exists
    }
}

// PartialEq implementation relies on TensorData's PartialEq
impl<T: PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        // Compare the internal TensorData directly
        *self.0.borrow() == *other.0.borrow()
        // Alternatively, compare Rc pointers if identity matters (e.g., in specific graph contexts)
        // Rc::ptr_eq(&self.0, &other.0)
    }
}

// Eq relies on TensorData's Eq
impl<T: Eq> Eq for Tensor<T> {}

// Hash implementation based on the memory address of the Rc payload (TensorData)
impl<T> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use the pointer address of the Rc's allocation for hashing.
        // This ensures that two Tensors pointing to the *same* data structure
        // hash to the same value, which is crucial for using Tensors in HashSets
        // during the topological sort in `backward`.
        Rc::as_ptr(&self.0).hash(state);
    }
}

// --- Arithmetic Operators are now in neurarust-core/src/ops/arithmetic/ ---

// --- Tests ---

#[cfg(test)]
pub(crate) mod tests {
    use super::*; // Import parent module content
    use num_traits::Zero;
    use std::collections::HashSet;
     

    // Helper to create a basic tensor for tests
    pub(crate) fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    // Helper to create a tensor that requires grad for tests
    pub(crate) fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let t = Tensor::new(data, shape);
        t.set_requires_grad(true);
        t
    }

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let t = create_test_tensor(data.clone(), shape.clone());

        assert_eq!(t.data(), data);
        assert_eq!(t.shape(), shape);
        assert_eq!(t.numel(), 4);
        assert!(!t.requires_grad());
        assert!(t.borrow_grad().is_none());
    }

    #[test]
    fn test_tensor_equality() {
        let t1 = create_test_tensor(vec![1_i32, 2], vec![2]);
        let t2 = create_test_tensor(vec![1_i32, 2], vec![2]);
        let t3 = create_test_tensor(vec![3_i32, 4], vec![2]);
        let t4 = create_test_tensor(vec![1_i32, 2], vec![1, 2]); // Different shape
        let t5 = t1.clone(); // Shallow clone

        assert_eq!(t1, t2); // Same data and shape
        assert_ne!(t1, t3); // Different data
        assert_ne!(t1, t4); // Different shape
        assert_eq!(t1, t5); // Clone points to same data

        // Test PartialEq with requires_grad
        let t1_grad = create_test_tensor_with_grad::<i32>(vec![1, 2], vec![2]);
        assert_ne!(t1, t1_grad); // requires_grad differs
    }

    #[test]
    fn test_tensor_hash_eq_for_set() {
        let t1 = create_test_tensor(vec![1_i32, 2], vec![2]);
        let t2 = create_test_tensor(vec![1_i32, 2], vec![2]); // Logically equal, different allocation
        let t3 = t1.clone(); // Points to same allocation as t1

        let mut set = HashSet::new();
        assert!(set.insert(t1.clone())); // Insert t1
        assert!(!set.insert(t3));      // t3 has same address, should not insert
        assert!(set.insert(t2.clone())); // t2 has different address, should insert - CLONE t2 here

        assert_eq!(set.len(), 2);
        assert!(set.contains(&t1));
        assert!(set.contains(&t2)); // Check for t2 after inserting its clone

        // Demonstrate pointer equality check (used by Hash)
        assert!(Rc::ptr_eq(&t1.0, &t1.clone().0));
        assert!(!Rc::ptr_eq(&t1.0, &t2.0));
    }


    #[test]
    fn test_backward_basic() {
        // Requires dummy operations or manual graph setup if ops are separate
        // For now, just test backward on a single node requires grad
        let t1 = create_test_tensor_with_grad(vec![5.0_f32], vec![1]);

        assert!(t1.borrow_grad().is_none());
        t1.backward(); // Should init grad to 1.0 for a scalar

        let grad = t1.grad();
        assert!(grad.is_some());
        let grad_tensor = grad.unwrap();
        assert_eq!(grad_tensor.data(), vec![1.0_f32]);
        assert_eq!(grad_tensor.shape(), vec![1]);
    }

    // --- Tests for Add, Sub, Mul, Div, Neg have been moved to their respective files ---
    // e.g., neurarust-core/src/ops/arithmetic/add.rs

} // end mod tests