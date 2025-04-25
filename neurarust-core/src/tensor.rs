use num_traits::{One, Zero}; // Keep imports needed by creation/ops if they remain here initially, or move them
use std::rc::Rc;
use std::cell::RefCell;

/// Represents a multi-dimensional array, the core data structure of NeuraRust.
///
/// Stores data in a flat `Vec<T>` on the CPU.
/// `T` represents the element type (e.g., `f32`, `i64`).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    /// Flattened data buffer stored contiguously in memory.
    /// Kept private to enforce invariants via methods.
    pub(crate) data: Vec<T>,
    /// The dimensions of the tensor (e.g., `[2, 3]` for a 2x3 matrix).
    /// Kept private to enforce invariants.
    pub(crate) shape: Vec<usize>,
    // Strides might be added later for optimized indexing

    // --- Autograd fields ---
    /// Flag indicating if gradients need to be computed for this tensor.
    pub requires_grad: bool,
    /// Stores the computed gradient for this tensor after `.backward()` is called.
    /// Uses Rc<RefCell<>> for shared mutability, allowing gradient accumulation.
    pub grad: Option<Rc<RefCell<Tensor<T>>>>,
    // /// Reference to the operation that created this tensor, used for backpropagation.
    // /// TODO: Define and implement the GradientContext/BackwardOp part.
    // grad_fn: Option<Rc<dyn BackwardOp<T>>>,
}

// Implementation block for core Tensor methods (creation from data, accessors)
impl<T> Tensor<T> {
    /// Creates a new `Tensor` from raw data and a shape, without gradient tracking by default.
    ///
    /// The provided data vector (`data`) must contain exactly as many elements
    /// as specified by the product of the dimensions in `shape`.
    ///
    /// # Panics
    /// Panics if the number of elements in `data` does not match the
    /// product of the dimensions in `shape`.
    ///
    /// # Examples
    /// ```
    /// // Assuming Tensor is in scope
    /// // let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// // assert_eq!(tensor.shape(), &[2, 2]);
    /// ```
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length ({}) must match the product of shape dimensions ({})",
            data.len(),
            expected_len
        );
        Tensor {
            data,
            shape,
            requires_grad: false, // Default to no gradient tracking
            grad: None,
            // grad_fn: None, // TODO
        }
    }

    /// Creates a new `Tensor` with gradient tracking enabled.
    pub fn new_with_grad(data: Vec<T>, shape: Vec<usize>) -> Self {
        let mut tensor = Self::new(data, shape);
        tensor.requires_grad = true;
        tensor
    }

    /// Returns a slice providing read-only access to the underlying data buffer.
    ///
    /// The data is stored in a flattened, contiguous layout (row-major order).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a slice representing the shape (dimensions) of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This is equivalent to the product of all dimensions in the shape.
    pub fn numel(&self) -> usize {
        // Shape is guaranteed to match data.len() by `new`, so either works.
        // Iterating shape might be slightly less direct if shape is complex later,
        // but conceptually links numel to shape. Let's use data.len() for now.
        self.data.len()
    }

    // --- Autograd methods ---

    /// Enables gradient tracking for this tensor.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if !requires_grad {
            self.grad = None; // Clear gradient if tracking is disabled
            // self.grad_fn = None; // TODO: Clear grad_fn too
        }
    }

    /// Initiates the backward pass to compute gradients.
    ///
    /// Starts the backpropagation process from this tensor.
    /// Assumes this tensor holds a scalar value (e.g., the loss).
    /// The gradient of the loss with respect to itself is 1.
    ///
    /// TODO: Implement the actual backpropagation logic.
    pub fn backward(&self) {
        // Check if gradient tracking was enabled
        if !self.requires_grad {
            // Maybe warn or ignore? For now, just return.
            eprintln!("Warning: Called backward() on a tensor that does not require gradients.");
            return;
        }

        // Ensure the tensor is scalar (or handle non-scalar backward later)
        if self.numel() != 1 {
             panic!("backward() can only be called on scalar tensors (for now).");
        }

        // Initialize the gradient for this tensor (dL/dL = 1)
        // We need a way to create a tensor of ones here.
        // Assume type T implements One + Clone for the gradient.
        // This part needs access to Tensor::ones, which is tricky from here.
        // Let's postpone the full implementation.
        println!("Backward pass initiated from tensor (shape: {:?}). (Implementation pending)", self.shape);

        // TODO: Traverse the computation graph using grad_fn and compute gradients.
        // self.grad = Some(Rc::new(RefCell::new(Tensor::ones(self.shape.clone())))); // Requires T: One + Clone
        // if let Some(grad_fn) = &self.grad_fn {
        //     grad_fn.backward(self.grad.clone().unwrap());
        // }

    }
}

// --- Tests for core Tensor functionality ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;

    #[test]
    fn test_tensor_new_ok() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // Specify type for clarity
        let shape = vec![2, 3];
        let t = Tensor::new(data.clone(), shape.clone());

        assert_eq!(t.data(), &data);
        assert_eq!(t.shape(), &shape);
        assert_eq!(t.numel(), 6);
        assert!(!t.requires_grad); // Default should be false
        assert!(t.grad.is_none());
    }

    #[test]
    #[should_panic]
    fn test_tensor_new_panic() {
        let data = vec![1.0_f32, 2.0, 3.0]; // Specify type
        let shape = vec![2, 3];
        let _t = Tensor::new(data, shape);
    }

    #[test]
    fn test_tensor_debug_clone_partial_eq() { // Renamed to reflect added PartialEq
        let data = vec![1.0_f32, 2.0]; // Specify type
        let shape = vec![1, 2];
        let t1 = Tensor::new(data, shape);
        let t2 = t1.clone(); // Test Clone trait
        println!("{:?}", t1); // Test Debug trait
        assert_eq!(t1, t2); // Test PartialEq trait
        assert!(!t1.requires_grad);
        assert!(!t2.requires_grad);
    }

    // --- New tests for autograd fields ---
    #[test]
    fn test_new_with_grad() {
        let t = Tensor::<f32>::new_with_grad(vec![1.0, 2.0], vec![2]);
        assert!(t.requires_grad);
        assert!(t.grad.is_none()); // Grad is initially None
    }

    #[test]
    fn test_set_requires_grad() {
        let mut t = Tensor::<f32>::new(vec![1.0, 2.0], vec![2]);
        assert!(!t.requires_grad);

        t.set_requires_grad(true);
        assert!(t.requires_grad);

        // Add a dummy grad to test clearing
        t.grad = Some(Rc::new(RefCell::new(Tensor::new(vec![0.0, 0.0], vec![2]))));
        assert!(t.grad.is_some());

        t.set_requires_grad(false);
        assert!(!t.requires_grad);
        assert!(t.grad.is_none()); // Should be cleared
    }

    #[test]
    #[should_panic = "backward() can only be called on scalar tensors (for now)."]
    fn test_backward_on_non_scalar() {
        let t = Tensor::<f32>::new_with_grad(vec![1.0, 2.0], vec![2]);
        t.backward(); // Should panic
    }

    #[test]
    fn test_backward_on_scalar_no_panic_for_now() {
        // This test just ensures backward doesn't panic on a scalar tensor,
        // even though the implementation is pending.
        let t = Tensor::<f32>::new_with_grad(vec![5.0], vec![1]);
        t.backward();
        // No panic expected, prints warning/message
    }

     #[test]
    fn test_backward_on_non_tracking_tensor() {
        // This test ensures backward does nothing harmful on non-tracking tensors.
        let t = Tensor::<f32>::new(vec![5.0], vec![1]);
        t.backward();
        // No panic expected, prints warning
        assert!(t.grad.is_none()); // Grad should remain None
    }
}

// --- Tests moved to appropriate modules --- 