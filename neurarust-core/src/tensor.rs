// neurarust-core/src/tensor.rs
use std::cell::{Ref, RefCell, RefMut}; // Import RefCell related types
use std::fmt;
use std::rc::{Rc, Weak}; // Import Rc and Weak

// --- Internal Data Structure ---

// Represents the actual tensor properties. Not public.
// No derive traits here; they are implemented on the public Tensor wrapper.
pub(crate) struct TensorData<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    // Autograd fields
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Tensor<T>>, // Stores the public wrapper type
    pub(crate) grad_fn: Option<Rc<dyn std::any::Any>>, // Placeholder type
                                             // TODO: Replace Any with actual BackwardOp trait later
                                             // pub(crate) grad_fn: Option<GradientContext<T>>,
}

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

    /// Provides temporary immutable access to the underlying data via `Ref`.
    /// The `Ref` acts like a read lock; ensure it's dropped promptly.
    pub fn borrow_data(&self) -> Ref<Vec<T>> {
        Ref::map(self.0.borrow(), |td| &td.data)
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

    /// Initiates the backward pass to compute gradients (stub).
    /// Requires the tensor to be scalar and `requires_grad` to be true.
    pub fn backward(&self) {
        let tensor_data = self.0.borrow(); // Borrow immutably first for checks
        if !tensor_data.requires_grad {
            eprintln!("Warning: Called backward() on a tensor that does not require gradients.");
            return;
        }
        if tensor_data.numel() != 1 {
            panic!("backward() can only be called on scalar tensors (for now).");
        }

        // --- Actual backward logic would go here ---
        // 1. Set self.grad to Tensor::ones(self.shape) if it's None. Requires T: One+Clone...
        //    Need to handle this initialization carefully. Maybe backward takes initial grad?
        //    For now, let's assume it's handled externally or in the first grad_fn call.

        // 2. Call the backward function stored in grad_fn, passing the gradient.
        println!(
            "Backward pass initiated from tensor (shape: {:?}). (grad_fn execution pending)",
            tensor_data.shape
        );

        // Example of how it might look (needs BackwardOp trait defined):
        // if let Some(grad_fn) = &tensor_data.grad_fn {
        //     let grad_to_pass = self.borrow_grad().clone(); // Need grad ready here
        //     if let Some(grad_tensor) = grad_to_pass {
        //          grad_fn.backward(grad_tensor);
        //      } else {
        //          // Handle case where initial grad isn't set (maybe set it to ones here?)
        //      }
        // }
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

impl<T: PartialEq> PartialEq for Tensor<T> {
    /// Compares two `Tensor`s based on shape and data.
    /// Ignores autograd state (`requires_grad`, `grad`, `grad_fn`).
    fn eq(&self, other: &Self) -> bool {
        let self_td = self.0.borrow();
        let other_td = other.0.borrow();
        self_td.shape == other_td.shape && self_td.data == other_td.data
    }
}

// --- Tests for the Public Tensor Wrapper ---
#[cfg(test)]
mod tests {
    use super::*; // Import the public Tensor wrapper

    // Helper function for tests
    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }

    fn create_test_tensor_with_grad<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new_with_grad(data, shape)
    }

    #[test]
    fn test_tensor_wrapper_new_ok() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let t = create_test_tensor(data.clone(), shape.clone());

        assert_eq!(t.shape(), shape);
        assert_eq!(t.data(), data);
        assert_eq!(t.numel(), 6);
        assert!(!t.requires_grad());
        assert!(t.borrow_grad().is_none());
    }

    #[test]
    #[should_panic]
    fn test_tensor_wrapper_new_panic() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![2, 3];
        let _t = create_test_tensor(data, shape);
    }

    #[test]
    fn test_tensor_wrapper_clone_partial_eq() {
        let data = vec![1.0_f32, 2.0];
        let shape = vec![1, 2];
        let t1 = create_test_tensor(data.clone(), shape.clone());
        let t2 = t1.clone(); // Clone the wrapper

        println!("t1: {:?}", t1);
        println!("t2: {:?}", t2);

        assert_eq!(t1, t2); // PartialEq compares data
        assert!(Rc::ptr_eq(&t1.0, &t2.0)); // Should point to same internal data

        // Mutating via one affects the other
        t1.set_requires_grad(true);
        assert!(t1.requires_grad());
        assert!(t2.requires_grad()); // t2 sees the change via shared RefCell
    }

    #[test]
    fn test_wrapper_new_with_grad() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        assert!(t.requires_grad());
        assert!(t.borrow_grad().is_none());
    }

    #[test]
    fn test_wrapper_set_requires_grad() {
        let t = create_test_tensor::<f32>(vec![1.0, 2.0], vec![2]);
        assert!(!t.requires_grad());
        t.set_requires_grad(true);
        assert!(t.requires_grad());

        // Add a dummy grad
        {
            let mut grad_opt = t.borrow_grad_mut();
            *grad_opt = Some(create_test_tensor(vec![0.0, 0.0], vec![2]));
        }
        assert!(t.borrow_grad().is_some());

        t.set_requires_grad(false);
        assert!(!t.requires_grad());
        assert!(t.borrow_grad().is_none()); // Grad should be cleared
    }

    #[test]
    #[should_panic = "backward() can only be called on scalar tensors (for now)."]
    fn test_wrapper_backward_on_non_scalar() {
        let t = create_test_tensor_with_grad::<f32>(vec![1.0, 2.0], vec![2]);
        t.backward();
    }

    #[test]
    fn test_wrapper_backward_on_scalar_no_panic_for_now() {
        let t = create_test_tensor_with_grad::<f32>(vec![5.0], vec![1]);
        t.backward(); // Just check it doesn't panic
    }

    #[test]
    fn test_wrapper_backward_on_non_tracking_tensor() {
        let t = create_test_tensor::<f32>(vec![5.0], vec![1]);
        t.backward(); // Should just print warning and return
        assert!(t.borrow_grad().is_none());
    }
}