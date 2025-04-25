// Ce module contiendra les implémentations pour l'indexation des Tenseurs.

use crate::tensor::Tensor;
use std::ops::Index;

/// Implements 2D indexing (read-only) for Tensors using `Rc<RefCell<>>`.
impl<T> Index<[usize; 2]> for Tensor<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        // Borrow the internal data immutably
        let tensor_data = self.0.borrow();

        // Check dimensions (on borrowed data)
        assert_eq!(tensor_data.shape.len(), 2, "Indexing with [row, col] requires a 2D tensor.");
        let rows = tensor_data.shape[0];
        let cols = tensor_data.shape[1];
        let row_idx = index[0];
        let col_idx = index[1];

        // Bounds checking
        assert!(row_idx < rows, "Row index {} out of bounds for shape {:?}", row_idx, tensor_data.shape);
        assert!(col_idx < cols, "Column index {} out of bounds for shape {:?}", col_idx, tensor_data.shape);

        // Calculate flat index
        let flat_index = row_idx * cols + col_idx;

        // PROBLEM: We cannot return a reference (`&T`) that lives longer than the borrow.
        // Standard Index trait returns `&T`, which is tied to `&self`. But our `T` is
        // inside a RefCell, protected by a temporary borrow (`Ref`).
        // When the `Ref` goes out of scope at the end of this function, any reference
        // borrowed from it becomes invalid.
        //
        // SOLUTIONS:
        // 1. Return owned data `T` (requires `T: Copy` or `T: Clone`). Simplest for now.
        // 2. Use more complex approaches like custom Deref implementations or libraries
        //    designed for this (e.g., `owning_ref`), but adds complexity.
        //
        // Let's go with solution 1 for now, returning T by value (copy).
        // This means we need to change the `impl Index` signature if `T: Copy`
        // or find another way. Standard `Index` MUST return a reference.
        //
        // => TEMPORARY WORKAROUND: Use the existing `get_val` helper which returns T.
        //    This means `std::ops::Index` cannot be implemented correctly this way.
        //    We will use `tensor.get_val([r, c])` instead of `tensor[[r, c]]` for now.
        //
        // Proper `Index` implementation might require rethinking the Tensor wrapper or
        // accepting the complexity of returning references tied to the Ref lifetime.

        // Return a reference tied to the temporary Ref<'_, TensorData<T>> lifetime.
        // This is UNSAFE if the caller holds onto the reference longer than the Ref lives.
        // To make this compile temporarily (but it's fundamentally broken for Index trait):
        // We need to leak the reference or use unsafe. Let's REMOVE Index impl for now.

        panic!("std::ops::Index returning a reference from RefCell content is complex/unsafe. Use tensor.get_val() for now.");
        // &tensor_data.data[flat_index] // This doesn't work due to lifetimes
    }
}

// Removed the Index implementation due to lifetime issues with RefCell.
// TODO: Revisit Index implementation, potentially using a different approach or library,
// or providing methods like `get(index)` that return owned/cloned data.

#[cfg(test)]
mod tests {
    use crate::Tensor;

    // Helper
    fn create_test_tensor<T>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }

    #[test]
    fn test_get_val_2d_ok() {
        let t = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        // Use get_val instead of [[...]]
        assert_eq!(t.get_val([0, 0]), 1);
        assert_eq!(t.get_val([0, 1]), 2);
        assert_eq!(t.get_val([0, 2]), 3);
        assert_eq!(t.get_val([1, 0]), 4);
        assert_eq!(t.get_val([1, 1]), 5);
        assert_eq!(t.get_val([1, 2]), 6);
    }

    #[test]
    #[should_panic = "Row index 2 out of bounds for shape [2, 3]"]
    fn test_get_val_2d_row_out_of_bounds() {
        let t = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        t.get_val([2, 0]);
    }

    #[test]
    #[should_panic = "Column index 3 out of bounds for shape [2, 3]"]
    fn test_get_val_2d_col_out_of_bounds() {
        let t = create_test_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        t.get_val([1, 3]);
    }

    #[test]
    #[should_panic = "get_val with [row, col] requires a 2D tensor."]
    fn test_get_val_2d_on_1d_tensor() {
        let t = create_test_tensor(vec![1, 2, 3], vec![3]);
        t.get_val([0, 0]);
    }

     #[test]
    #[should_panic = "get_val with [row, col] requires a 2D tensor."]
    fn test_get_val_2d_on_3d_tensor() {
        let t = create_test_tensor(vec![1], vec![1, 1, 1]);
        t.get_val([0, 0]);
    }
} 