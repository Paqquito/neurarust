#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::utils::testing::{create_test_tensor, create_test_tensor_with_grad};
    use std::sync::Arc;
    use crate::ops::view::transpose::transpose_op;

    #[test]
    fn test_transpose_basic() {
        let t = create_test_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let transposed = t.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), vec![3, 2]);

        let t_guard = t.read_data();
        let transposed_guard = transposed.read_data();
        assert_eq!(Arc::as_ptr(&t_guard.data), Arc::as_ptr(&transposed_guard.data), "Transpose should share buffer");
        assert_eq!(transposed_guard.offset, t_guard.offset);
        assert_eq!(transposed_guard.strides, vec![1, 3]);
        assert!(!transposed.is_contiguous());

        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[0, 0])], 1.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[0, 1])], 4.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[1, 0])], 2.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[1, 1])], 5.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[2, 0])], 3.0);
        assert_eq!(transposed_guard.data.cpu_data().unwrap()[transposed_guard.get_offset(&[2, 1])], 6.0);
    }

    #[test]
    fn test_transpose_backward() {
        let input_data = create_test_tensor_with_grad(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![3, 2]).unwrap();

        // Calculate analytical gradient
        let output = input_data.transpose(0, 1).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is transpose(output_grad_val)
        let expected_grad = transpose_op(&output_grad_val, 0, 1).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }

    #[test]
    fn test_transpose_backward_higher_dim() {
        let input_data = create_test_tensor_with_grad(
            (1..=24).map(|x| x as f64).collect(),
            vec![2, 3, 4],
        );
        let output_grad_val = Tensor::<f64>::ones(vec![2, 4, 3]).unwrap();

        // Calculate analytical gradient
        let output = input_data.transpose(1, 2).unwrap();
        output.backward(Some(output_grad_val.clone())).unwrap();

        let input_grad = input_data.grad().unwrap();

        // Expected grad is transpose(output_grad_val)
        let expected_grad = transpose_op(&output_grad_val, 1, 2).unwrap();

        assert_eq!(input_grad.shape(), expected_grad.shape(), "Shape mismatch");
        // Compare data (assuming CPU)
        let input_grad_data = input_grad.read_data().data.cpu_data().unwrap().clone();
        let expected_grad_data = expected_grad.read_data().data.cpu_data().unwrap().clone();
        assert_eq!(input_grad_data.as_slice(), expected_grad_data.as_slice(), "Data mismatch");
    }
} 