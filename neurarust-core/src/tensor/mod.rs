// src/tensor/mod.rs
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::{AddAssign};
use std::rc::{Rc, Weak};

use num_traits::{Zero, One};

use crate::tensor_data::TensorData;
use crate::autograd::{BackwardOp, graph::build_topo};
use std::collections::{HashSet, HashMap};
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
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self where T: Clone {
        let numel = if shape.is_empty() {
            1 // Scalar tensor has 1 element
        } else {
            shape.iter().product()
        };
        assert_eq!(data.len(), numel, 
            "Data length ({}) does not match shape {:?} (expected {} elements)", 
            data.len(), shape, numel);
        
        // Initialize TensorData directly
        let tensor_data = TensorData {
            data,
            shape,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            _ctx: None,
        };
        Tensor { data: Rc::new(RefCell::new(tensor_data)) }
    }

    /// Creates a new tensor that requires gradient tracking.
    pub fn new_with_grad(data: Vec<T>, shape: Vec<usize>) -> Self where T: Clone + Debug {
        let tensor = Tensor::new(data, shape);
        tensor.set_requires_grad(true);
        tensor
    }

    /// Creates a tensor of zeros with the same shape as the given tensor.
    pub fn zeros_like(other: &Tensor<T>) -> Self where T: Zero + Clone {
        let shape = other.shape();
        let numel = shape.iter().product::<usize>();
        let data = vec![T::zero(); numel];
        Tensor::new(data, shape)
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
        let new_tensor_data = TensorData {
            data: data_clone,
            shape: new_shape,
            requires_grad: false, // Reshaped tensor does not track grad by default
            grad: None,
            grad_fn: None, // No grad_fn for basic reshape
            _ctx: None,
        };
        Tensor { data: Rc::new(RefCell::new(new_tensor_data)) }
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
    pub fn backward(&self, upstream_grad: Option<&Tensor<T>>)
        where 
        T: Clone + Zero + One + Copy + Debug + 'static + AddAssign 
    {
        // --- 1. Build Topological Sort --- 
        let mut visited = HashSet::new();
        let mut sorted_graph = Vec::new();
        build_topo(self, &mut visited, &mut sorted_graph);

        // --- 2. Initialize external gradient storage --- 
        // Utiliser un pointeur vers le RefCell comme clé
        let mut gradients: HashMap<*const RefCell<TensorData<T>>, Tensor<T>> = HashMap::new();

        // --- 3. Set initial gradient for the final node --- 
        let final_grad_val = match upstream_grad {
            Some(grad) => {
                assert_eq!(self.shape(), grad.shape(),
                           "Upstream gradient shape {:?} must match tensor shape {:?} for backward call",
                           grad.shape(), self.shape());
                grad.clone()
            },
            None => {
                assert_eq!(self.numel(), 1, "backward() without arguments can only be called on a scalar tensor.");
                Tensor::new(vec![T::one()], vec![1]) 
            }
        };
        // Stocker le gradient initial dans le HashMap
        gradients.insert(Rc::as_ptr(&self.data), final_grad_val);

        // --- 4. Propagate gradients backward through the sorted graph --- 
        for node in sorted_graph.iter().rev() { // Iterate in reverse topological order
            // Obtenir le pointeur vers le RefCell pour la clé du HashMap
            let node_ptr = Rc::as_ptr(&node.data);

            // Get the gradient for the current node from the HashMap
            // Cloner le gradient si trouvé pour le passer à backward
            let grad_to_propagate = match gradients.get(&node_ptr) {
                Some(grad) => grad.clone(),
                None => continue, // Pas de gradient pour ce nœud (e.g., ne requiert pas grad)
            };
            
            // Get the grad_fn associated with this node
            let grad_fn = node.grad_fn(); // Clones the Rc<dyn BackwardOp>
            
            // If a grad_fn exists, call its backward method
            if let Some(grad_fn_op) = grad_fn {
                 // Passer le gradient et une référence mutable au HashMap pour accumulation
                 grad_fn_op.backward(&grad_to_propagate, &mut gradients);
            }
        }

        // --- 5. (Optionnel) Copier les gradients finaux dans les tenseurs originaux --- 
        // Pour correspondre à l'API précédente où .grad() renvoie le résultat
        for node in sorted_graph {
            let node_ptr = Rc::as_ptr(&node.data);
            if let Some(final_grad) = gradients.remove(&node_ptr) {
                 // Emprunter mutablament pour assigner le grad final
                 let mut td = node.borrow_tensor_data_mut(); 
                 td.grad = Some(final_grad);
            }
        }
        // --- Old recursive call (to be removed) ---
        // self._backward_recursive(&final_grad);
    }

    /// Returns a clone of the gradient function Rc, if it exists.
    pub fn grad_fn(&self) -> Option<Rc<dyn BackwardOp<T>>> where T: 'static { 
        self.data.borrow().grad_fn.clone()
    }

    /// Returns a weak reference to the tensor data.
    pub(crate) fn get_weak_ref(&self) -> Weak<RefCell<TensorData<T>>> {
        Rc::downgrade(&self.data)
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
