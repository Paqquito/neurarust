// Ce module contiendra les implémentations pour l'indexation des Tenseurs.

use crate::tensor::Tensor;
use std::ops::{Index, IndexMut};

/// Allows indexing into a 2D tensor using `tensor[[row, col]]`.
/// Requires the element type `T` to be `Copy`.
impl<T> Index<[usize; 2]> for Tensor<T>
where
    T: Copy,
{
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let td = self.0.borrow();
        assert_eq!(td.shape.len(), 2, "2D indexing requires a 2D tensor.");
        let rows = td.shape[0];
        let cols = td.shape[1];
        let row_idx = index[0];
        let col_idx = index[1];
        assert!(row_idx < rows, "Row index out of bounds");
        assert!(col_idx < cols, "Column index out of bounds");
        // Prefix flat_index with _ as it's unused for direct return by reference
        let _flat_index = row_idx * cols + col_idx;
        // This is tricky. We need to return a reference, but the data is in a Vec
        // inside a RefCell. Returning a direct reference `&T` that lives longer
        // than the Ref borrow is unsafe or impossible directly.
        // For immutable indexing, perhaps returning a copy is the only safe way?
        // Let's reconsider the return type or the approach.
        // Panicking for now as returning a reference is non-trivial.
        panic!("Immutable indexing returning a reference is not safely implemented yet.");
        // &td.data[flat_index] // This would be unsafe if Ref drops
    }
}

/// Allows mutable indexing into a 2D tensor using `tensor[[row, col]] = value`.
/// Requires the element type `T` to be `Copy`.
impl<T> IndexMut<[usize; 2]> for Tensor<T>
where
    T: Copy,
{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let td = self.0.borrow_mut();
        assert_eq!(td.shape.len(), 2, "2D indexing requires a 2D tensor.");
        let rows = td.shape[0];
        let cols = td.shape[1];
        let row_idx = index[0];
        let col_idx = index[1];
        assert!(row_idx < rows, "Row index out of bounds");
        assert!(col_idx < cols, "Column index out of bounds");
        // Prefix flat_index with _ as it's unused for direct return by reference
        let _flat_index = row_idx * cols + col_idx;
        // This is also tricky for mutable access. Returning &mut T requires careful handling
        // of the RefMut borrow.
        panic!("Mutable indexing returning a reference is not safely implemented yet.");
        // &mut td.data[flat_index]
    }
}

// Removed the Index implementation due to lifetime issues with RefCell.
// TODO: Revisit Index implementation, potentially using a different approach or library,
// or providing methods like `get(index)` that return owned/cloned data.

#[cfg(test)]
mod tests {
    use crate::{Tensor, tensor::TensorData};
    use std::rc::Rc;
    use std::cell::RefCell;

    // Helper from tensor tests (if needed, or define locally)
    fn create_test_tensor<T: Clone + std::fmt::Debug + PartialEq>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        Tensor::new(data, shape)
    }

    #[test]
    fn test_get_val_2d_ok() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.get_val([0, 0]), 1.0);
        assert_eq!(t.get_val([0, 2]), 3.0);
        assert_eq!(t.get_val([1, 1]), 5.0);
        assert_eq!(t.get_val([1, 2]), 6.0);
    }

    #[test]
    #[should_panic(expected = "get_val with [row, col] requires a 2D tensor")]
    fn test_get_val_2d_on_1d_tensor() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let _ = t.get_val([0, 1]);
    }

    #[test]
    #[should_panic(expected = "get_val with [row, col] requires a 2D tensor")]
    fn test_get_val_2d_on_3d_tensor() {
        let t = create_test_tensor(vec![1.0; 8], vec![2, 2, 2]);
        let _ = t.get_val([0, 1]);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")] // Updated expected message
    fn test_get_val_2d_row_out_of_bounds() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let _ = t.get_val([2, 1]); // Row 2 is out of bounds
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")] // Updated expected message
    fn test_get_val_2d_col_out_of_bounds() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let _ = t.get_val([1, 3]); // Col 3 is out of bounds
    }
} 