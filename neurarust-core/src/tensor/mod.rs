// src/tensor/mod.rs
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::{AddAssign, Mul, Add};
use std::rc::{Rc, Weak};
use std::iter::Sum;

use num_traits::{Float, Zero, One};
use num_traits::FromPrimitive;

use crate::tensor_data::TensorData;
use crate::autograd::{BackwardOp, graph::build_topo};
use std::collections::{HashSet, HashMap};
use crate::ops::indexing::TensorSlice;
use crate::error::NeuraRustError;
use crate::ops::math_elem;
use crate::ops::arithmetic::add;

pub mod utils; // Declare the utils submodule

/// Represents a multi-dimensional array (Tensor).
/// Uses Rc<RefCell<TensorData>> for interior mutability and shared ownership.
pub struct Tensor<T> {
    pub(crate) data: Rc<RefCell<TensorData<T>>>,
}

// --- Combine all inherent methods into one block ---
impl<T> Tensor<T> {
    // --- Constructors and Basic Properties ---
    /// Creates a new tensor from a vector of data and a shape.
    ///
    /// # Arguments
    /// * `data` - A vector containing the tensor data in row-major order.
    /// * `shape` - A vector defining the dimensions of the tensor.
    ///
    /// # Returns
    /// A `Result` containing the new `Tensor` on success, or a `NeuraRustError`
    /// if the data length does not match the product of the shape dimensions.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: Clone {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(NeuraRustError::TensorCreationError {
                data_len: data.len(),
                shape: shape.clone(),
            });
        }

        let strides = TensorData::<T>::calculate_contiguous_strides(&shape);

        let tensor_data = TensorData {
            data,
            shape,
            strides,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            _ctx: None,
        };

        Ok(Tensor {
            data: Rc::new(RefCell::new(tensor_data)),
        })
    }

    /// Creates a tensor with the same properties as this tensor, but requires gradient tracking.
    /// Intended for creating leaf nodes in the computation graph that need gradients.
    pub fn new_with_grad(data: Vec<T>, shape: Vec<usize>) -> Result<Self, NeuraRustError> where T: Clone + Debug {
        let tensor = Tensor::new(data, shape)?;
        tensor.set_requires_grad(true); // Note: set_requires_grad might panic if called on non-leaf, need to check its logic later.
        Ok(tensor)
    }

    /// Creates a tensor of zeros with the same shape as the given tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Result<Self, NeuraRustError> where T: Zero + Clone {
        let shape = other.shape();
        let numel = shape.iter().product::<usize>();
        let data = vec![T::zero(); numel];
        Tensor::new(data, shape) // Use the fallible new()
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
    pub fn borrow_tensor_data_mut(&self) -> RefMut<TensorData<T>> {
        self.data.borrow_mut()
    }

    /// Provides immutable access to the underlying TensorData.
    pub fn borrow_tensor_data(&self) -> Ref<TensorData<T>> {
        self.data.borrow()
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
    /// Panics if slicing fails (e.g., wrong number of slices, index out of bounds).
    /// Use `neurarust::ops::indexing::slice_op` for fallible slicing.
    pub fn slice(&self, slices: &[TensorSlice]) -> Tensor<T>
    where
        T: Clone + Debug + Default + Zero + AddAssign + Copy + 'static,
    {
        // Call the fallible op and expect success (panic on error)
        crate::ops::indexing::slice_op(self, slices)
            .unwrap_or_else(|e| panic!("Tensor slice failed: {:?}", e))
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
        let mut visited = HashSet::new();
        let mut sorted_graph: Vec<Tensor<T>> = Vec::new();
        build_topo(self, &mut visited, &mut sorted_graph);

        let mut gradients: HashMap<*const RefCell<TensorData<T>>, Tensor<T>> = HashMap::new();

        let initial_grad = match upstream_grad {
            Some(grad) => {
                if grad.shape() != self.shape() {
                    panic!("Upstream gradient shape {:?} must match tensor shape {:?}", grad.shape(), self.shape());
                }
                grad.clone()
            }
            None => {
                if self.numel() != 1 {
                    panic!("backward() called on non-scalar tensor without explicit upstream_grad");
                }
                Tensor::scalar(T::one())
            }
        };
        gradients.insert(Rc::as_ptr(&self.data), initial_grad);

        for node_tensor in sorted_graph.iter().rev() { 
            let tensor_ptr = Rc::as_ptr(&node_tensor.data); 
            
            if let Some(grad) = gradients.get(&tensor_ptr) { 
                let tensor_data = node_tensor.data.borrow(); 
                if let Some(grad_fn) = tensor_data.grad_fn.clone() {
                    let current_grad = grad.clone(); 
                    grad_fn.backward(&current_grad, &mut gradients);
                }
            }
        }

        for (tensor_ptr, final_grad) in gradients.into_iter() {
            unsafe {
                // Borrow mutably only once
                let mut td_borrow = (*tensor_ptr).borrow_mut();
                
                if td_borrow.requires_grad { // Only accumulate if grad is required
                    match td_borrow.grad.as_mut() {
                        Some(existing_grad) => {
                            // Use the fallible add operation
                            match add(existing_grad, &final_grad) {
                                Ok(summed_grad) => {
                                    // If add succeeds, replace the existing gradient
                                    *existing_grad = summed_grad; 
                                },
                                Err(e) => {
                                    // Panic if shapes mismatch during final accumulation
                                    panic!("Gradient shape mismatch during final accumulation: {:?}. Existing {:?}, Final {:?}", 
                                           e, existing_grad.shape(), final_grad.shape());
                                }
                            }
                        }
                        None => {
                            // If no grad exists yet, just set it
                            td_borrow.grad = Some(final_grad);
                        }
                    }
                }
            } // Mutable borrow drops here
        }
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
            .expect("Internal error: Failed to create scalar tensor")
    }

    /// Computes the element-wise square root of the tensor.
    ///
    /// # Panics
    /// Panics if the operation fails (e.g., negative input for integer types).
    /// Use `neurarust::ops::math_elem::sqrt_op` for fallible sqrt.
    pub fn sqrt(&self) -> Tensor<T>
    where
        T: Float + Debug + Clone + AddAssign + Default + Zero + One + Sum + 'static + FromPrimitive,
    {
        // Call the fallible op and expect success
        math_elem::sqrt_op(self)
            .unwrap_or_else(|e| panic!("Tensor sqrt failed: {:?}", e))
    }

    /// Sums the tensor elements along the specified axes.
    /// See `crate::ops::reduction::sum::sum_axes` for details.
    pub fn reduce_sum(&self, axes: &[usize], keep_dim: bool) -> Result<Tensor<T>, NeuraRustError>
    where
        T: Clone + Zero + AddAssign + Debug + Copy + Send + Sync + 'static + Default + PartialEq + Sum + PartialOrd + One,
    {
        crate::ops::reduction::sum::sum_axes(self, axes, keep_dim)
    }

    /// Stacks a sequence of tensors along a new dimension.
    /// See `crate::ops::stack::stack_op` for details.
    pub fn stack(tensors: &[Tensor<T>], dim: usize) -> Tensor<T>
    where
        // Add Copy constraint required by stack_op
        T: Clone + Debug + Default + Zero + One + AddAssign + 'static + Copy,
    {
        // Call the fallible stack_op and panic on error
        crate::ops::stack::stack_op(tensors, dim)
            .unwrap_or_else(|e| panic!("Tensor stack failed: {:?}", e))
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
        Tensor::new(data, shape).expect("Test tensor creation failed")
    }
    // Add necessary bounds for backward tests
    fn create_test_tensor_with_grad<T: Clone + Debug + PartialEq + Zero + One + Copy + 'static + AddAssign>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let tensor = Tensor::new(data, shape).expect("Test tensor_with_grad creation failed (new)");
        tensor.set_requires_grad(true); // Now tensor is a Tensor<T>
        tensor // Return the tensor
    }

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data.clone(), shape.clone());
        assert!(tensor.is_ok());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.data().as_ref(), data.as_slice());
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 4);
        assert!(!t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_creation_error() {
        let data = vec![1.0, 2.0, 3.0]; // Incorrect data length
        let shape = vec![2, 2];
        let tensor_result = Tensor::<f32>::new(data, shape.clone());
        assert!(tensor_result.is_err());
        assert!(matches!(tensor_result.err().unwrap(), NeuraRustError::TensorCreationError { data_len: 3, shape: s } if s == shape));
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
        t.data.borrow_mut().grad = Some(Tensor::new(vec![5.0], vec![])
            .expect("Test setup failed: creating manual grad"));
        assert_eq!(&*t.grad().unwrap().data(), &[5.0]);

        t.zero_grad();
        assert!(t.grad().is_some());
        assert_eq!(&*t.grad().unwrap().data(), &[0.0]);

        let t2 = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        t2.zero_grad();
        assert!(t2.grad().is_none());
    }

    #[test]
    fn test_backward_basic() {
        type TestType = f32;
        let a = create_test_tensor_with_grad::<TestType>(vec![2.0], vec![1]);
        let b = create_test_tensor_with_grad::<TestType>(vec![3.0], vec![1]);

        #[derive(Debug)]
        struct MockAddBackward {
            a_ref: Weak<RefCell<TensorData<TestType>>>,
            b_ref: Weak<RefCell<TensorData<TestType>>>,
        }
        impl BackwardOp<TestType> for MockAddBackward where TestType: AddAssign + Copy + Zero + One + Clone + Debug + 'static + PartialEq + Default {
            fn backward(&self, upstream_grad: &Tensor<TestType>, gradients: &mut HashMap<*const RefCell<TensorData<TestType>>, Tensor<TestType>>) { 
                let grad_a = upstream_grad.clone();
                let grad_b = upstream_grad.clone();

                if let Some(a_rc) = self.a_ref.upgrade() {
                    let a_ptr = Rc::as_ptr(&a_rc);
                    // Use Tensor::zeros_like
                    let current_grad = gradients.entry(a_ptr)
                        .or_insert_with(|| Tensor::zeros_like(&grad_a).expect("Failed to create zero grad like grad_a"));
                    *current_grad += &grad_a; 
                }
                if let Some(b_rc) = self.b_ref.upgrade() {
                    let b_ptr = Rc::as_ptr(&b_rc);
                    // Use Tensor::zeros_like
                    let current_grad = gradients.entry(b_ptr)
                        .or_insert_with(|| Tensor::zeros_like(&grad_b).expect("Failed to create zero grad like grad_b"));
                    *current_grad += &grad_b;
                }
            }
            fn inputs(&self) -> Vec<Weak<RefCell<TensorData<TestType>>>> { vec![self.a_ref.clone(), self.b_ref.clone()] }
        }

        let c_data = vec![a.data()[0] + b.data()[0]];
        let c = create_test_tensor::<TestType>(c_data, vec![1]);
        c.set_grad_fn(Some(Rc::new(MockAddBackward {
            a_ref: a.get_weak_ref(),
            b_ref: b.get_weak_ref(),
        })));

        c.backward(None); 

        let grad_a = a.grad().expect("Gradient for a not found");
        assert_eq!(grad_a.data().as_ref(), &[1.0]);
        let grad_b = b.grad().expect("Gradient for b not found");
        assert_eq!(grad_b.data().as_ref(), &[1.0]);
    }

    #[test]
    #[should_panic] // Keep should_panic until backward returns Result
    fn test_backward_none_on_non_scalar() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        t.backward(None); // Should panic
    }

    #[test]
    #[should_panic] // Keep should_panic until backward returns Result
    fn test_backward_some_shape_mismatch() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0], vec![1]);
        let grad = create_test_tensor::<f32>(vec![1.0, 2.0], vec![2]); // Mismatched shape
        t.backward(Some(&grad)); // Should panic
    }

    // zeros_like is now part of the main impl block
    
    #[test]
    fn test_zeros_creation() {
        let shape = vec![2, 3];
        let tensor = zeros::<f32>(shape.clone());
        assert!(tensor.is_ok());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.data().as_ref(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_ones_creation() {
        let shape = vec![1, 4];
        let tensor = ones::<i32>(shape.clone());
        assert!(tensor.is_ok());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.data().as_ref(), &[1, 1, 1, 1]);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_full_creation() {
        let shape = vec![2, 1, 2];
        let fill_val = 42.0f64;
        let tensor = full(shape.clone(), fill_val);
        assert!(tensor.is_ok());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.data().as_ref(), &[42.0, 42.0, 42.0, 42.0]);
        assert!(!t.requires_grad());
    }
}

// --- Standalone Creation Functions --- 

/// Creates a tensor filled with zeros.
///
/// # Arguments
/// * `shape` - The desired shape of the tensor.
///
/// # Returns
/// A new `Tensor<T>` filled with zeros.
pub fn zeros<T: Zero + Clone>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data = vec![T::zero(); numel];
    Tensor::new(data, shape)
}

/// Creates a tensor filled with ones.
///
/// # Arguments
/// * `shape` - The desired shape of the tensor.
///
/// # Returns
/// A new `Tensor<T>` filled with ones.
pub fn ones<T: One + Clone>(shape: Vec<usize>) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data = vec![T::one(); numel];
    Tensor::new(data, shape)
}

/// Creates a tensor filled with a specific value.
///
/// # Arguments
/// * `shape` - The desired shape of the tensor.
/// * `fill_value` - The value to fill the tensor with.
///
/// # Returns
/// A new `Tensor<T>` filled with `fill_value`.
pub fn full<T: Clone>(shape: Vec<usize>, fill_value: T) -> Result<Tensor<T>, NeuraRustError> {
    let numel = shape.iter().product::<usize>();
    let data = vec![fill_value; numel];
    Tensor::new(data, shape)
}
