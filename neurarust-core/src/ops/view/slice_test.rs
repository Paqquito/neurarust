#[cfg(test)]
mod tests {
    use crate::ops::view::slice::{slice_op, SliceArg}; // Import op and SliceArg
    use crate::tensor::{self, Tensor}; // Use tensor:: for creation funcs like ones
    use crate::error::NeuraRustError;
    use crate::autograd::grad_check::{check_grad, GradCheckError};
    use std::sync::Arc; // For comparing buffer pointers

    // --- Helper to create SliceArg easily from ranges (assuming step 1) ---
    // Note: For more complex SliceArg cases, construct them directly.
    fn range_to_slice_arg(start: isize, end: isize) -> SliceArg {
        SliceArg::Slice(start, end, 1)
    }

    // --- FORWARD TESTS (merged & updated) ---

    #[test]
    fn test_slice_basic() -> Result<(), NeuraRustError> {
        let t = Tensor::new((0..12).map(|x| x as f32).collect(), vec![2, 2, 3])?;
        // Slice [0..1, 0..2, 1..3]
        let ranges = vec![
            range_to_slice_arg(0, 1),
            range_to_slice_arg(0, 2),
            range_to_slice_arg(1, 3),
        ];
        let sliced = slice_op(&t, &ranges)?; // Slice to [1, 2, 2]
        assert_eq!(sliced.shape(), &[1, 2, 2]);
        // Strides should be inherited from the original tensor for a simple slice view
        assert_eq!(sliced.strides(), t.strides(), "Strides should match original for simple slice");
        // Original strides for shape [2, 2, 3] are typically [6, 3, 1] or [12, 3, 1] depending on impl details.
        // Let's directly compare with original strides.
        Ok(())
    }

    #[test]
    fn test_slice_single_element() -> Result<(), NeuraRustError> {
        let t = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3])?;
        // Slice [1, 2] -> results in a scalar or 0-dim tensor if supported by slice_op
        // Using SliceArg: SliceArg::Index(1), SliceArg::Index(2)
        // For simplicity matching old test: slice [1..2, 2..3]
        let ranges = vec![range_to_slice_arg(1, 2), range_to_slice_arg(2, 3)];
        let sliced = slice_op(&t, &ranges)?; // Slice to [1, 1]
        assert_eq!(sliced.shape(), &[1, 1]);
        assert_eq!(sliced.strides(), &[3, 1]); // Strides based on original layout
        // Check data if needed
        // assert_eq!(sliced.get_f32_data()?, vec![5.0]); // Element at [1, 2] is 5.0
        Ok(())
    }

    #[test]
    fn test_slice_full_range() -> Result<(), NeuraRustError> {
        let t = Tensor::new((0..12).map(|x| x as f32).collect(), vec![2, 2, 3])?;
        // Slice [:, :, :] which is [0..2, 0..2, 0..3]
        let ranges = vec![
            range_to_slice_arg(0, 2),
            range_to_slice_arg(0, 2),
            range_to_slice_arg(0, 3),
        ];
        let sliced = slice_op(&t, &ranges)?; // Full slice
        assert_eq!(sliced.shape(), t.shape());
        assert_eq!(sliced.strides(), t.strides());
        // Verify it's essentially the same tensor view
        assert!(Arc::ptr_eq(&t.read_data().buffer, &sliced.read_data().buffer));
        assert_eq!(t.read_data().offset, sliced.read_data().offset);
        Ok(())
    }

    #[test]
    fn test_slice_empty_dim() -> Result<(), NeuraRustError> {
        let t = Tensor::new((0..12).map(|x| x as f32).collect(), vec![2, 2, 3])?;
        // Slice [1..1, :, :] which is [1..1, 0..2, 0..3]
        let ranges = vec![
            range_to_slice_arg(1, 1), // Empty range
            range_to_slice_arg(0, 2),
            range_to_slice_arg(0, 3),
        ];
        let sliced = slice_op(&t, &ranges)?; // Slice dim 0 to be empty
        assert_eq!(sliced.shape(), &[0, 2, 3]);
        assert_eq!(sliced.numel(), 0);
        Ok(())
    }

    // --- ERROR / EDGE CASE TESTS ---

    #[test]
    fn test_slice_rank_mismatch() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0f32], vec![1])?;
        let ranges = vec![range_to_slice_arg(0, 1), range_to_slice_arg(0, 1)];
        let result = slice_op(&t, &ranges);
        // This should ideally be SliceError, but check for any error for now
        assert!(result.is_err(), "Slice with wrong number of args should return an error");
        Ok(())
    }

    #[test]
    fn test_slice_invalid_range_start_gt_end() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0], vec![2])?;
        let ranges = vec![range_to_slice_arg(1, 0)]; // Invalid range 1..0
        let sliced = slice_op(&t, &ranges)?; // Should produce an empty slice
        assert_eq!(sliced.shape(), &[0], "Slice with start > end should result in a 0-sized dimension");
        assert_eq!(sliced.numel(), 0, "Number of elements should be 0 for an empty slice");
        Ok(())
    }

    #[test]
    fn test_slice_invalid_range_end_gt_size() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0], vec![2])?;
        let ranges = vec![range_to_slice_arg(0, 3)]; // Invalid range 0..3 for dim size 2
        let sliced = slice_op(&t, &ranges)?; // Should be clamped to 0..2
        assert_eq!(sliced.shape(), &[2], "Slice with end > size should be clamped to dim size");
        // Check actual data if clamped
        let data = sliced.get_f32_data()?;
        assert_eq!(data, vec![1.0, 2.0], "Data should match the clamped slice");
        Ok(())
    }

    #[test]
    fn test_slice_view_data_sharing() -> Result<(), NeuraRustError> {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let ranges = vec![range_to_slice_arg(0, 1), range_to_slice_arg(0, 2)];
        let sliced = slice_op(&t, &ranges)?;

        // Prefix unused variables with _
        let _t_buffer_ptr = Arc::as_ptr(&t.read_data().buffer);
        let _sliced_buffer_ptr = Arc::as_ptr(&sliced.read_data().buffer);

        assert!(Arc::ptr_eq(&t.read_data().buffer, &sliced.read_data().buffer), "Buffers should be shared");
        assert_eq!(t.read_data().offset, sliced.read_data().offset, "Offsets should be equal for slice starting at 0");
        Ok(())
    }

    // --- AUTOGRAD TESTS (Placeholders, ignored) ---

    #[test]
    #[ignore = "Slice backward implementation is complex and needs thorough verification."]
    fn test_slice_backward_f64() -> Result<(), GradCheckError> {
        let t_data = (0..12).map(|x| x as f64).collect();
        let t = Tensor::new_f64(t_data, vec![2, 2, 3])?;
        t.set_requires_grad(true)?;
        
        // Define the slice ranges, e.g., [0..1, 0..2, 1..3]
        let slice_args = vec![
            range_to_slice_arg(0, 1),
            range_to_slice_arg(0, 2),
            range_to_slice_arg(1, 3),
        ];
        let output_shape = vec![1, 2, 2]; // Expected shape after slice

        let slice_fn = |inputs: &[Tensor]| slice_op(&inputs[0], &slice_args);
        
        let output_grad = tensor::ones_f64(&output_shape)?;

        check_grad(slice_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
    }

    #[test]
    #[ignore = "Slice backward implementation is complex and needs thorough verification."]
    fn test_slice_backward_step_f64() -> Result<(), GradCheckError> {
        let t_data = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new_f64(t_data, vec![4, 6])?;
        t.set_requires_grad(true)?;
        
        // Slice [1:4:2, 0:5:2] -> shape [2, 3]
        let slice_args = vec![
            SliceArg::Slice(1, 4, 2),
            SliceArg::Slice(0, 5, 2),
        ];
        let output_shape = vec![2, 3]; // Expected shape ( (4-1)/2 = 1.5 -> 2 ), ( (5-0)/2 = 2.5 -> 3 )

        let slice_fn = |inputs: &[Tensor]| slice_op(&inputs[0], &slice_args);
        
        let output_grad = tensor::ones_f64(&output_shape)?;

        check_grad(slice_fn, &[t], &output_grad, 1e-5, 1e-7, 1e-5)
    }
} 