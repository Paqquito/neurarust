// src/tensor/mod.rs
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::{AddAssign, Mul, Add};
use std::rc::{Rc, Weak};
use std::iter::Sum;

use num_traits::{Float, Zero, One};

use crate::tensor_data::TensorData;
use crate::autograd::{BackwardOp, graph::build_topo};
use std::collections::{HashSet, HashMap};
use crate::ops::indexing::TensorSlice;
// Supprimer HashSet (non utilisé directement dans ce fichier)
// use std::collections::HashSet;

pub mod utils; // Declare the utils submodule

/// Represents a multi-dimensional array (Tensor).
/// Uses Rc<RefCell<TensorData>> for interior mutability and shared ownership.
pub struct Tensor<T> {
    pub(crate) data: Rc<RefCell<TensorData<T>>>,
}

// --- Combine all inherent methods into one block ---
impl<T> Tensor<T> {
    // --- Constructors and Basic Properties ---
    /// Creates a new tensor with the given data and shape.
    /// Calculates contiguous strides.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self where T: Clone {
        // Utilise le nouveau constructeur TensorData::new qui calcule les strides
        let tensor_data = TensorData::new(data, shape);
        Tensor { data: Rc::new(RefCell::new(tensor_data)) }
    }

    /// Creates a new tensor that requires gradient tracking.
    pub fn new_with_grad(data: Vec<T>, shape: Vec<usize>) -> Self where T: Clone + Debug {
        let tensor = Tensor::new(data, shape); // Calls the updated new()
        tensor.set_requires_grad(true);
        tensor
    }

    /// Creates a tensor of zeros with the same shape as the given tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Self where T: Zero + Clone {
        let shape = other.shape();
        let numel = shape.iter().product::<usize>();
        let data = vec![T::zero(); numel];
        Tensor::new(data, shape) // Calls the updated new()
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.data.borrow().shape.clone()
    }

    /// Returns the number of dimensions (rank) of the tensor.
    pub fn ndim(&self) -> usize {
        self.data.borrow().shape.len()
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.data.borrow().shape.iter().product()
    }

    /// Returns a slice view of the tensor's data.
    pub fn data(&self) -> Ref<[T]> {
        Ref::map(self.data.borrow(), |td| td.data.as_slice())
    }

    /// Returns a mutable slice view of the tensor's data.
    /// Allows in-place modification of tensor values.
    /// Panics if the data is already borrowed mutably elsewhere.
    pub fn data_mut(&self) -> RefMut<[T]> {
        RefMut::map(self.data.borrow_mut(), |td| td.data.as_mut_slice())
    }

    /// Provides mutable access to the underlying TensorData.
    pub(crate) fn borrow_tensor_data_mut(&self) -> RefMut<TensorData<T>> {
        self.data.borrow_mut()
    }

    /// Provides immutable access to the underlying TensorData.
    pub(crate) fn borrow_tensor_data(&self) -> Ref<TensorData<T>> {
        self.data.borrow()
    }

    /// Reshapes the tensor to the new shape, without copying data.
    /// Panics if the total number of elements differs.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor<T> where T: Clone {
        let numel_new: usize = new_shape.iter().product();
        let numel_old = self.numel();
        assert_eq!(numel_old, numel_new,
                   "Cannot reshape tensor with {} elements to shape {:?} (requires {} elements)",
                   numel_old, new_shape, numel_new);
        
        // Create a new TensorData with the same data but new shape
        // Note: This creates a new Rc/RefCell, breaking autograd tracking if not handled carefully.
        // A proper reshape should ideally modify the shape in-place or create a view.
        // For now, let's assume this reshape is mainly for non-gradient tensors or specific internal uses.
        let data_clone = self.data().to_vec(); // Clone data for the new tensor
        
        // Calculate contiguous strides for the new shape
        let new_strides = TensorData::<T>::calculate_contiguous_strides(&new_shape);
        
        let new_tensor_data = TensorData {
            data: data_clone,
            shape: new_shape,
            strides: new_strides, // Add calculated strides
            requires_grad: false, // Reshaped tensor does not track grad by default
            grad: None,
            grad_fn: None, // No grad_fn for basic reshape
            _ctx: None,
        };
        Tensor { data: Rc::new(RefCell::new(new_tensor_data)) }
    }

    /// Performs slicing/indexing on the tensor.
    /// 
    /// Returns a new tensor containing the selected elements.
    /// This operation supports autograd.
    ///
    /// # Arguments
    /// * `slices` - A slice of `TensorSlice` enum variants specifying the selection 
    ///   for each dimension. The length of `slices` must match the number of 
    ///   dimensions of the tensor.
    ///
    /// # Panics
    /// Panics if the number of slices does not match the tensor dimensions, 
    /// or if indices/ranges are out of bounds.
    ///
    /// # Example
    /// ```ignore
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// // Get the first row
    /// let row0 = tensor.slice(&[TensorSlice::Index(0), TensorSlice::Full]); 
    /// // Get the element at [1, 1]
    /// let elem11 = tensor.slice(&[TensorSlice::Index(1), TensorSlice::Index(1)]);
    /// // Get the sub-matrix with rows 0..1 and columns 1..3
    /// let sub = tensor.slice(&[TensorSlice::Range(0..1), TensorSlice::Range(1..3)]);
    /// ```
    pub fn slice(&self, slices: &[TensorSlice]) -> Tensor<T>
    where
        T: Clone + Debug + Default + Zero + AddAssign + 'static,
    {
        // Implementation will delegate to a function in ops::indexing
        crate::ops::indexing::slice_op(self, slices)
    }

    // --- Autograd related methods ---
    /// Sets whether this tensor requires gradient tracking.
    pub fn set_requires_grad(&self, requires_grad: bool) where T: Debug {
        let mut td = self.data.borrow_mut();
        if requires_grad && td.grad_fn.is_some() {
            eprintln!("PANIC set_requires_grad: Attempting on non-leaf tensor:");
            eprintln!("  Shape: {:?}", td.shape);
            eprintln!("  Requires Grad: {}", td.requires_grad);
            eprintln!("  Has Grad Fn: {}", td.grad_fn.is_some());
            panic!("Cannot set requires_grad on a non-leaf tensor.");
        }
        td.requires_grad = requires_grad;
        // No need to initialize grad here, handled in _backward_recursive or zero_grad
    }

    /// Checks if this tensor requires gradient tracking.
    pub fn requires_grad(&self) -> bool {
        self.data.borrow().requires_grad
    }

    /// Returns an immutable reference to the gradient tensor, if it exists.
    pub fn grad(&self) -> Option<Ref<Tensor<T>>> {
        Ref::filter_map(self.data.borrow(), |td| td.grad.as_ref()).ok()
    }

    /// Returns a mutable reference to the gradient tensor, if it exists.
    pub(crate) fn grad_mut(&self) -> Option<RefMut<Tensor<T>>> {
        RefMut::filter_map(self.data.borrow_mut(), |td| td.grad.as_mut()).ok()
    }

    /// Returns an immutable reference to the gradient tensor. Alias for grad().
    pub fn borrow_grad(&self) -> Option<Ref<Tensor<T>>> {
        self.grad()
    }

    /// Zeroes out the gradient tensor associated with this tensor.
    pub fn zero_grad(&self) where T: Zero {
        if let Some(grad_tensor) = self.grad_mut() {
            let mut grad_data = grad_tensor.borrow_tensor_data_mut();
            grad_data.data.iter_mut().for_each(|x| *x = T::zero());
        }
    }

    /// Initiates the backward pass (autograd) starting from this tensor.
    /// Uses topological sort and external gradient storage.
    /// 
    /// This method computes the gradient of this tensor with respect to 
    /// its ancestors in the computation graph that require gradients.
    /// 
    /// The backward pass proceeds as follows:
    /// 1. Build a topological sort of the computation graph starting from this tensor.
    /// 2. Initialize a HashMap to store gradients, keyed by the raw pointer 
    ///    of the `Rc<RefCell<TensorData>>` for each tensor in the graph.
    /// 3. Seed the gradient for this tensor (the starting point of the backward pass).
    ///    If `upstream_grad` is provided, it's used; otherwise, if the tensor is 
    ///    a scalar, a gradient of `1.0` is used. Panics if called on a non-scalar 
    ///    tensor without an explicit `upstream_grad`.
    /// 4. Iterate through the topologically sorted graph in reverse order.
    /// 5. For each node, retrieve its accumulated gradient from the HashMap.
    /// 6. If the node has a `grad_fn` (meaning it was produced by an operation),
    ///    call the `backward` method of its `grad_fn`. The `grad_fn` will compute
    ///    the local gradients with respect to its inputs and use the provided 
    ///    HashMap to accumulate these gradients into the entries for its input tensors.
    /// 7. After iterating through the graph, the HashMap contains the final gradients
    ///    for all tensors that required gradients.
    /// 8. Finally, copy the computed gradients from the HashMap into the `.grad` 
    ///    field of the corresponding `TensorData` for compatibility with the `grad()` method.
    ///
    /// # Arguments
    /// * `upstream_grad` - An optional tensor representing the gradient flowing into 
    ///   this tensor from subsequent operations. Must have the same shape as this tensor.
    ///   If `None`, assumes this is the final tensor in the computation and initiates
    ///   the backward pass with a gradient of `1.0` (only valid for scalar tensors).
    pub fn backward(&self, upstream_grad: Option<&Tensor<T>>)
        where 
        T: Clone + Zero + One + Copy + Debug + 'static + AddAssign + Add<Output=T> + Mul<Output=T> + PartialEq + Default + Sum,
    {
        // --- 1. Build Topological Sort --- 
        // `build_topo` performs a Depth First Search (DFS) starting from the current tensor (`self`)
        // and populates `sorted_graph` with tensors in a reverse topological order
        // (dependents appear before dependencies). `visited` prevents cycles and redundant visits.
        let mut visited = HashSet::new();
        let mut sorted_graph = Vec::new();
        build_topo(self, &mut visited, &mut sorted_graph);

        // --- 2. Initialize external gradient storage --- 
        // We use a HashMap to store gradients externally during the backward pass.
        // This avoids potential borrowing issues if we tried to store gradients directly 
        // within the TensorData while iterating.
        // The key is the raw pointer to the `Rc<RefCell<TensorData>>`, providing a stable 
        // identifier for each tensor node in the graph during this pass.
        // Using `Rc::as_ptr` is generally safe here because the `Rc`s involved are kept 
        // alive by the `sorted_graph` list and potentially other references during the pass.
        let mut gradients: HashMap<*const RefCell<TensorData<T>>, Tensor<T>> = HashMap::new();

        // --- 3. Set initial gradient for the final node --- 
        // The backward pass starts with an initial gradient for the tensor `self`.
        let final_grad_val = match upstream_grad {
            Some(grad) => {
                // If an upstream gradient is provided, use it.
                assert_eq!(self.shape(), grad.shape(),
                           "Upstream gradient shape {:?} must match tensor shape {:?} for backward call",
                           grad.shape(), self.shape());
                grad.clone()
            },
            None => {
                // If no upstream gradient, assume it's the final loss (must be scalar).
                assert_eq!(self.numel(), 1, "backward() without arguments can only be called on a scalar tensor.");
                // Default gradient for a scalar loss is 1.0.
                Tensor::new(vec![T::one()], vec![]) // Use empty vec for scalar shape
            }
        };
        let self_ptr = Rc::as_ptr(&self.data);
        gradients.insert(self_ptr, final_grad_val);

        // --- 4. Propagate gradients backward through the sorted graph --- 
        println!("[Tensor::backward] Starting propagation loop. Initial grad set for {:?}", self_ptr);
        for node in sorted_graph.iter().rev() { 
            let node_ptr = Rc::as_ptr(&node.data); 
            println!("[Tensor::backward] Processing node: {:?}", node_ptr);

            if gradients.contains_key(&node_ptr) {
                 let grad_to_propagate = gradients.get(&node_ptr).expect("Checked contains key").clone(); 
                 println!("[Tensor::backward]  Node {:?} has gradient.", node_ptr);
                 if let Some(grad_fn_op) = node.grad_fn() { 
                     println!("[Tensor::backward]  Node {:?} has grad_fn. Calling its backward...", node_ptr);
                     grad_fn_op.backward(&grad_to_propagate, &mut gradients);
                     println!("[Tensor::backward]  ...backward call finished for node {:?}", node_ptr);
                 } else {
                     println!("[Tensor::backward]  Node {:?} is a leaf or has no grad_fn.", node_ptr);
                 }
            } else {
                 println!("[Tensor::backward]  Node {:?} has no gradient yet.", node_ptr);
            }
        }
        println!("[Tensor::backward] Propagation loop finished.");

        // --- 5. Copy final gradients to TensorData --- 
        println!("[Tensor::backward] Final gradients map keys: {:?}", gradients.keys().collect::<Vec<_>>());
        for node_ptr in visited { 
            if gradients.contains_key(&node_ptr) { // Check if grad exists before removing
                 println!("[Tensor::backward] Checking assignment for node {:?}", node_ptr);
                 let requires_grad_flag = unsafe { (*node_ptr).borrow().requires_grad };
                 if requires_grad_flag { 
                      let final_grad = gradients.remove(&node_ptr).unwrap(); // Now remove safely
                      println!("[Tensor::backward]   Node requires grad. Assigning gradient.");
                      unsafe {
                          let mut td = (*node_ptr).borrow_mut();
                          td.grad = Some(final_grad);
                      }
                 } else {
                     println!("[Tensor::backward]   Node does NOT require grad. Skipping assignment.");
                 }
            } else {
                println!("[Tensor::backward] No final gradient found for visited node {:?}", node_ptr);
            }
        }
        println!("[Tensor::backward] Finished.");
    }

    /// Returns a clone of the gradient function Rc, if it exists.
    pub fn grad_fn(&self) -> Option<Rc<dyn BackwardOp<T>>> where T: 'static { 
        self.data.borrow().grad_fn.clone()
    }

    /// Sets the gradient function for this tensor.
    /// Used internally by operations during the forward pass to build the graph.
    pub fn set_grad_fn(&self, grad_fn: Option<Rc<dyn BackwardOp<T>>>) where T: 'static {
        // Should only be set on non-leaf tensors
        // We might add assertions here later if needed.
        self.data.borrow_mut().grad_fn = grad_fn;
    }

    /// Returns a weak reference to the tensor data.
    /// Useful for autograd graph construction without creating cycles.
    pub fn get_weak_ref(&self) -> Weak<RefCell<TensorData<T>>> {
        Rc::downgrade(&self.data)
    }

    /// Returns a stable identifier (pointer address) for the underlying data allocation.
    /// Useful as a key in HashMaps when the Tensor itself might be passed via different Rcs.
    pub fn id(&self) -> *const () {
        Rc::as_ptr(&self.data) as *const ()
    }

    /// Creates a scalar tensor (0-dimensional) containing a single value.
    pub fn scalar(value: T) -> Self where T: Clone {
        Tensor::new(vec![value], vec![])
    }

    /// Computes the element-wise square root of the tensor.
    /// This operation supports autograd.
    /// 
    /// # Returns
    /// A new tensor containing the square root of each element.
    /// 
    /// # Panics
    /// Typically panics if the tensor contains negative values when using standard floats,
    /// or might produce NaN depending on the float type's behavior.
    pub fn sqrt(&self) -> Tensor<T>
    where
        T: Float + Debug + 'static + Clone + Zero + One + AddAssign + Default + Copy + Mul<Output = T>,
    {
        crate::ops::math_elem::sqrt::SqrtOp::forward(self)
    }

    /// Stacks a sequence of tensors along a new dimension.
    ///
    /// All input tensors must have the same shape.
    /// The new dimension is inserted at the position `dim`.
    /// Supports autograd.
    ///
    /// # Arguments
    /// * `tensors` - A slice of tensors to stack.
    /// * `dim` - The dimension along which to stack (0 <= dim <= rank).
    ///
    /// # Returns
    /// A `Result` containing the stacked tensor or an error string.
    ///
    /// # Example
    /// ```ignore
    /// let t1 = Tensor::new(vec![1, 2], vec![2]);
    /// let t2 = Tensor::new(vec![3, 4], vec![2]);
    /// // Stack along new dimension 0
    /// let stacked0 = Tensor::stack(&[t1.clone(), t2.clone()], 0).unwrap(); 
    /// // stacked0 shape: [2, 2], data: [1, 2, 3, 4]
    /// // Stack along new dimension 1
    /// let stacked1 = Tensor::stack(&[t1.clone(), t2.clone()], 1).unwrap();
    /// // stacked1 shape: [2, 2], data: [1, 3, 2, 4] (depends on copy logic)
    /// ```
    pub fn stack(tensors: &[Tensor<T>], dim: usize) -> Result<Tensor<T>, String>
    where
        T: Clone + Debug + Default + Zero + One + AddAssign + 'static,
    {
        crate::ops::stack::stack_op(tensors, dim)
    }
} // End of the large impl<T> Tensor<T> block

// --- Trait Implementations ---
impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor { data: Rc::clone(&self.data) }
    }
}

impl<T: Debug> Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let td = self.data.borrow();
        f.debug_struct("Tensor")
         .field("data", &td.data)
         .field("shape", &td.shape)
         .field("requires_grad", &td.requires_grad)
         .field("grad_fn", &td.grad_fn.is_some())
         .field("grad", &td.grad)
         .finish()
    }
}

impl<T: PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        let self_td = self.data.borrow();
        let other_td = other.data.borrow();
        // Compare shape and data only
        self_td.shape == other_td.shape && self_td.data == other_td.data
    }
}
impl<T: Eq> Eq for Tensor<T> {} // T must be Eq for Tensor<T> to be Eq

impl<T: Hash> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let td = self.data.borrow();
        // Hash based on shape and data content
        td.shape.hash(state);
        td.data.hash(state);
    }
}

// --- Tests module ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::BackwardOp;
    use std::collections::HashSet;
    use num_traits::{Zero, One};
    use std::ops::AddAssign;

    // Helper for creating tensors
    fn create_test_tensor<T: Clone + Debug + PartialEq + Zero + One>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    // Add necessary bounds for backward tests
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + Copy + 'static + AddAssign>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let t = create_test_tensor::<f32>(data.clone(), shape.clone());
        assert_eq!(t.shape(), shape);
        assert_eq!(t.data().as_ref(), data.as_slice());
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 4);
        assert!(!t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    #[should_panic]
    fn test_tensor_creation_panic() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2];
        Tensor::new(data, shape); // Should panic
    }

    #[test]
    fn test_tensor_with_grad() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_equality() {
        let t1 = create_test_tensor::<i32>(vec![1, 2], vec![2]);
        let t2 = create_test_tensor::<i32>(vec![1, 2], vec![2]);
        let t3 = create_test_tensor::<i32>(vec![1, 3], vec![2]);
        let t4 = create_test_tensor::<i32>(vec![1, 2], vec![1, 2]);
        let t1_grad = create_test_tensor_with_grad::<i32>(vec![1, 2], vec![2]);

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
        assert_ne!(t1, t4);
        assert_eq!(t1, t1_grad); // Equality ignores autograd state
    }

    #[test]
    fn test_tensor_hash_eq_for_set() {
        // Use i32 because f32 is not Eq or Hash
        let t1 = create_test_tensor::<i32>(vec![1, 2], vec![2]);
        let t2 = create_test_tensor::<i32>(vec![1, 2], vec![2]); // Same content, different allocation
        let t3 = create_test_tensor::<i32>(vec![1, 3], vec![2]);
        let t1_grad = create_test_tensor_with_grad::<i32>(vec![1, 2], vec![2]); // Same content, diff grad state

        let mut set = HashSet::new();
        set.insert(t1.clone());

        // Hash/Eq are based on shape and data content
        assert!(set.contains(&t1));
        assert!(set.contains(&t2));
        assert!(!set.contains(&t3));
        assert!(set.contains(&t1_grad));
        assert_eq!(set.len(), 1); // Only one unique tensor value inserted
    }

    #[test]
    fn test_zero_grad() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0f32], vec![1]);
        // Manually set a gradient
        t.data.borrow_mut().grad = Some(Tensor::new(vec![5.0], vec![1]));
        assert_eq!(&*t.grad().unwrap().data(), &[5.0]); // Compare slices

        t.zero_grad();
        assert!(t.grad().is_some());
        assert_eq!(&*t.grad().unwrap().data(), &[0.0]); // Compare slices

        // Test on tensor without existing grad
        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        t2.zero_grad();
        assert!(t2.grad().is_none());
    }

    #[test]
    fn test_backward_basic() {
        // Garder AddAssign pour TestType
        type TestType = f32;

        let a = create_test_tensor_with_grad::<TestType>(vec![2.0], vec![1]);
        let b = create_test_tensor_with_grad::<TestType>(vec![3.0], vec![1]);

        // Mock Add operation and backward
        let c_data = vec![a.data()[0] + b.data()[0]];
        let c = Tensor::new(c_data, vec![1]);
        c.set_requires_grad(true);
        #[derive(Debug)]
        struct MockAddBackward {
            a_ref: Weak<RefCell<TensorData<TestType>>>,
            b_ref: Weak<RefCell<TensorData<TestType>>>,
        }
        impl BackwardOp<TestType> for MockAddBackward where TestType: AddAssign + Copy + Zero + One + Clone + Debug + 'static {
            fn backward(&self, upstream_grad: &Tensor<TestType>, gradients: &mut HashMap<*const RefCell<TensorData<TestType>>, Tensor<TestType>>) { 
                if let Some(a_rc) = self.a_ref.upgrade() {
                    let _a_td = a_rc.borrow_mut(); // Prefix with underscore
                    let grad_a = upstream_grad.clone(); // Mock: Add just passes upstream grad
                    gradients.entry(Rc::as_ptr(&a_rc))
                        .and_modify(|g| *g += &grad_a)
                        .or_insert(grad_a);
                }
                if let Some(b_rc) = self.b_ref.upgrade() {
                    let _b_td = b_rc.borrow_mut(); // Prefix with underscore
                    let grad_b = upstream_grad.clone(); // Mock: Add just passes upstream grad
                    gradients.entry(Rc::as_ptr(&b_rc))
                        .and_modify(|g| *g += &grad_b)
                        .or_insert(grad_b);
                }
            }
            fn inputs(&self) -> Vec<Weak<RefCell<TensorData<TestType>>>> { vec![self.a_ref.clone(), self.b_ref.clone()] }
        }
        let grad_fn = Rc::new(MockAddBackward { a_ref: a.get_weak_ref(), b_ref: b.get_weak_ref() });
        c.data.borrow_mut().grad_fn = Some(grad_fn);

        // --- Test Accumulation ---
        // We need to adapt the test because the current simple backward *overwrites* grads.
        // A true accumulation test requires the `_backward_recursive` TODO to be resolved.

        // Test backward with default gradient (1.0)
        a.zero_grad(); // Ensure grads are None initially
        b.zero_grad();
        c.backward(None); // Should set grad = 1.0 (overwriting)
        assert_eq!(&*a.borrow_grad().expect("Grad A missing").data(), &[1.0]);
        assert_eq!(&*b.borrow_grad().expect("Grad B missing").data(), &[1.0]);

        // Test backward with specified gradient (5.0)
        c.backward(Some(&Tensor::new(vec![5.0], vec![1]))); // Should set grad = 5.0 (overwriting)
        assert_eq!(&*a.borrow_grad().expect("Grad A missing after second backward").data(), &[5.0]);
        assert_eq!(&*b.borrow_grad().expect("Grad B missing after second backward").data(), &[5.0]);

        // TODO: Add a test for real accumulation once _backward_recursive is fixed.
    }

    #[test]
    #[should_panic(expected = "backward() without arguments can only be called on a scalar tensor.")]
    fn test_backward_none_on_non_scalar() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        t.backward(None); // Should panic
    }

    #[test]
    #[should_panic(expected = "Upstream gradient shape [1] must match tensor shape [2] for backward call")]
    fn test_backward_some_shape_mismatch() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        let upstream = Tensor::new(vec![1.0], vec![1]);
        t.backward(Some(&upstream)); // Should panic
    }

    // zeros_like is now part of the main impl block
}
