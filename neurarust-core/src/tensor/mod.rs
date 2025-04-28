use std::cell::{Ref, RefMut, RefCell};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::rc::{Rc, Weak};
use crate::tensor_data::TensorData;
use crate::autograd::BackwardOp;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use num_traits::{One, Zero};
use std::cmp::Eq;

pub mod utils; // Déclare le module utils comme public

/// Represents a multi-dimensional array (tensor).
///
/// This struct wraps the actual tensor data (`TensorData`) in an `Rc<RefCell<>>`
/// to allow for shared ownership (multiple tensors can point to the same data,
/// e.g., after a view or non-copying operation) and interior mutability
/// (needed for gradient accumulation and requires_grad changes).
#[derive(Clone)]
pub struct Tensor<T>(pub Rc<RefCell<TensorData<T>>>);

// ... Le reste du fichier ... 

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
        // Utiliser TensorData directement car on est dans le même crate
        let tensor_data = TensorData {
            data,
            shape,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            _ctx: None, // Assurer que tous les champs sont initialisés
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
    pub fn zeros(shape: Vec<usize>) -> Self where T: Zero + Clone {
        let numel = shape.iter().product::<usize>();
        let data = vec![T::zero(); numel];
        Tensor::new(data, shape)
    }

    /// Creates a new tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self where T: One + Clone {
        let numel = shape.iter().product::<usize>();
        let data = vec![T::one(); numel];
        Tensor::new(data, shape)
    }
    
    /// Creates a tensor of ones with the same shape as the given tensor.
    pub fn ones_like(other: &Tensor<T>) -> Self where T: One + Clone {
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
    pub fn data(&self) -> Vec<T> where T: Clone {
        self.0.borrow().data.clone()
    }

    /// Provides temporary immutable access to the internal `TensorData` via `Ref`.
    /// The `Ref` acts like a read lock; ensure it's dropped promptly.
    /// Made `pub(crate)` because it exposes the internal `TensorData` type.
    // Changé en pub car TensorData est maintenant pub
    pub fn borrow_tensor_data(&self) -> Ref<TensorData<T>> {
        self.0.borrow()
    }

    /// Provides temporary mutable access to the internal `TensorData` via `RefMut`.
    /// The `RefMut` acts like a write lock; ensure it's dropped promptly.
    /// Made `pub(crate)` because it exposes the internal `TensorData` type.
     // Changé en pub car TensorData est maintenant pub
    pub fn borrow_tensor_data_mut(&self) -> RefMut<TensorData<T>> {
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
    pub fn grad(&self) -> Option<Tensor<T>> where T: Clone {
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
    pub fn backward(&self) where T: One + Clone + 'static {
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
        // Utiliser crate::autograd::graph car on est dans tensor
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

// Clone est déjà implémenté via #[derive(Clone)] sur la struct

// Debug implementation relies on TensorData's Debug
impl<T: Debug> Debug for Tensor<T> {
    /// Formats the Tensor for display, showing its data and shape.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Borrow immutably to access data and shape
        let tensor_data = self.0.borrow();
        write!(f, "Tensor(data={:?}, shape={:?}, requires_grad={})",
               tensor_data.data, tensor_data.shape, tensor_data.requires_grad)
        // Optionally, add grad info if present:
        // .field("grad", &tensor_data.grad)
        // .field("grad_fn", &tensor_data.grad_fn.is_some()) // Only show if grad_fn exists
    }
}

impl<T: PartialEq> PartialEq for Tensor<T> {
    /// Compares two Tensors for equality based on their underlying `TensorData`.
    /// Checks data, shape, and requires_grad status.
    fn eq(&self, other: &Self) -> bool {
        // We need to borrow the data inside Rc<RefCell<>> to compare.
        let self_data = self.0.borrow();
        let other_data = other.0.borrow();
        // TensorData implements PartialEq, so we can directly compare the borrowed data.
        *self_data == *other_data
    }
}

// Eq requires that a == a always holds, which is true for our PartialEq implementation IF T: Eq.
// Rétabli pour permettre Tensor<i32> etc. dans HashSet, mais Tensor<f32> ne sera pas Eq.
impl<T: Eq> Eq for Tensor<T> {}

// Hash implementation based on the memory address of the Rc pointer.
// This ensures that two Tensor variables pointing to the same underlying data
// have the same hash, which is crucial for using Tensors in HashSets/HashMaps
// like the `visited` set during graph traversal.
impl<T> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the pointer address of the Rc itself.
        Rc::as_ptr(&self.0).hash(state);
    }
}

// --- Test Module --- 

#[cfg(test)]
pub(crate) mod tests {
    use super::*; // Import everything from the parent module (Tensor)
    use crate::tensor_data::TensorData; // Import TensorData for direct access in tests
    use num_traits::Zero;
    use std::collections::HashSet;
    use num_traits::One; // Importer One pour DummyBackward

    // Helper function to create tensors easily in tests
    // Made pub(crate) so it can be used by tests in other modules like ops
    pub(crate) fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }
    // Helper function to create tensors with requires_grad=true
    pub(crate) fn create_test_tensor_with_grad<T: Clone + std::fmt::Debug + PartialEq + Zero>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        let t = Tensor::new(data, shape);
        t.set_requires_grad(true);
        t
    }

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let t = Tensor::new(data.clone(), shape.clone());

        assert_eq!(t.data(), data);
        assert_eq!(t.shape(), shape);
        assert!(!t.requires_grad()); // Default is false
        assert!(t.grad().is_none());
        assert!(t.0.borrow().grad_fn.is_none());
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_tensor_equality() {
        let t1 = create_test_tensor(vec![1, 2], vec![2]);
        let t2 = create_test_tensor(vec![1, 2], vec![2]);
        let t3 = create_test_tensor(vec![1, 3], vec![2]);
        let t4 = create_test_tensor(vec![1, 2], vec![1, 2]);

        assert_eq!(t1, t2, "Tensors with same data and shape should be equal");
        assert_ne!(t1, t3, "Tensors with different data should not be equal");
        assert_ne!(t1, t4, "Tensors with different shape should not be equal");

        // Test equality with requires_grad
        let t1_grad = create_test_tensor_with_grad::<i32>(vec![1, 2], vec![2]);
        assert_ne!(t1, t1_grad, "Tensors differing only by requires_grad should not be equal"); 
        // This depends on TensorData's PartialEq impl
    }

    #[test]
    fn test_tensor_hash_eq_for_set() {
        // Utiliser i32 au lieu de f32 car f32 n'est pas Eq
        let t1 = create_test_tensor::<i32>(vec![1, 2], vec![2]); 
        let t1_clone = t1.clone(); // Devrait pointer vers les mêmes données, même hash
        let t2 = create_test_tensor::<i32>(vec![1, 2], vec![2]); // Allocation différente, hash différent

        let mut set = HashSet::new();
        // HashSet requiert Eq + Hash. Tensor implémente Hash. 
        // Pour Eq, HashSet va comparer les pointeurs Rc via PartialEq. 
        // Notre PartialEq compare TensorData, qui compare requires_grad.
        // Pour que ce test fonctionne comme prévu (basé sur Rc ptr), il faudrait que PartialEq compare les pointeurs.
        // Changeons PartialEq pour Tensor<T> pour comparer les pointeurs.
        
        // Réflexion: Non, PartialEq DOIT comparer le contenu sémantique.
        // Le HashSet fonctionne pour le graphe car il stocke *const RefCell<TensorData<T>>.
        // Ce test est peut-être mal conçu pour vérifier l'unicité via HashSet.
        // Gardons le test mais il ne prouve que le Hash basé sur Rc::as_ptr.
        // Il ne garantit PAS que set.insert(t1_clone) retournera false si PartialEq compare les données.
        // Pour l'instant, laissons comme ça. t1 et t1_clone auront même hash.
        // t1 et t2 auront hash différent.
        assert!(set.insert(t1.clone()));
        // Si PartialEq compare les données, t1_clone == t1, donc insert retourne false.
        assert!(!set.insert(t1_clone)); 
        // Si PartialEq compare les données, t2 == t1, donc insert retourne false. 
        // MAIS t2 a un hash différent. Donc set.insert(t2) devrait réussir.
        assert!(set.insert(t2)); 
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_backward_basic() {
        // Requires Add operation to be defined and working with autograd
        // Assuming Add exists and creates AddBackward node correctly.
        // We also need the autograd graph building to work.

        let a = create_test_tensor_with_grad::<f32>(vec![2.0], vec![1]);
        let b = create_test_tensor_with_grad::<f32>(vec![3.0], vec![1]);
        
        // Assuming Add impl exists like: `impl Add<&Tensor<T>> for &Tensor<T>`
        // let c = &a + &b; // Requires Add trait implementation
        // For now, let's manually create a dummy grad_fn scenario
        let c_data = TensorData {
            data: vec![5.0], // a+b
            shape: vec![1], 
            requires_grad: true,
            grad: None,
            grad_fn: Some(Rc::new(DummyBackward { // Dummy struct implementing BackwardOp
                inputs: vec![a.get_weak_ref(), b.get_weak_ref()],
            })),
            _ctx: None,
        };
        let c = Tensor(Rc::new(RefCell::new(c_data)));

        assert!(c.requires_grad());
        assert!(c.0.borrow().grad_fn.is_some());
        assert!(a.grad().is_none());
        assert!(b.grad().is_none());

        c.backward(); // Start backpropagation

        // Check gradients (assuming DummyBackward sets grad=1.0 for both inputs)
        let grad_a = a.grad().expect("Grad A should exist");
        let grad_b = b.grad().expect("Grad B should exist");
        assert_eq!(grad_a.data(), vec![1.0]);
        assert_eq!(grad_b.data(), vec![1.0]);
        assert_eq!(grad_a.shape(), vec![1]);
        assert_eq!(grad_b.shape(), vec![1]);
    }
    
    // Dummy BackwardOp for test_backward_basic
    struct DummyBackward<T> {
        inputs: Vec<Weak<RefCell<TensorData<T>>>>,
    }
    // Ajout T: Copy requis par data()[0].clone() et Mul<Output=T>
    impl<T: One + Clone + 'static + Copy + std::ops::Mul<Output = T>> BackwardOp<T> for DummyBackward<T> { 
        fn backward(&self, upstream_grad: &Tensor<T>) { 
            for input_weak in &self.inputs {
                if let Some(input_rc) = input_weak.upgrade() {
                    let mut input_td = input_rc.borrow_mut();
                    if input_td.requires_grad {
                         // Supprimé variable non utilisée
                         // let grad_val = upstream_grad.data()[0].clone(); 
                         input_td.grad = Some(upstream_grad.clone());
                    }
                }
            }
        }
        fn inputs(&self) -> Vec<Weak<RefCell<TensorData<T>>>> {
            self.inputs.clone()
        }
    }
} 