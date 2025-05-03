// REMOVED: use super::*;

#[cfg(test)]
mod tests {
    // REMOVED: use super::*;
    use crate::ops::reduction::sum::sum_axes; // Explicit import
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;
    use num_traits::{One, Zero};
    use std::default::Default;
    use std::fmt::Debug;
    use std::iter::Sum;
    use std::ops::{Add, AddAssign};
    
    fn create_test_tensor<T>(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Tensor<T>
    where
        T: Clone
            + Zero
            + AddAssign
            + Debug
            + Copy
            + Send
            + Sync
            + 'static
            + Default
            + PartialEq
            + PartialOrd
            + One
            + Sum
            + Add<Output = T>,
    {
        Tensor::new(data, shape).expect("Failed to create test tensor")
    }

    #[test]
    fn test_sum_all() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[], false).unwrap();
        assert_eq!(result.shape(), vec![], "Result shape should be scalar");
        let result_val = result.get(&[]).unwrap();
        assert_relative_eq!(result_val, 21.0, epsilon = 1e-7);
    }

    #[test]
    fn test_sum_axis_0() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[0], false).unwrap();
        assert_eq!(result.shape(), vec![3]);
        let expected_data = vec![5.0, 7.0, 9.0]; // [1+4, 2+5, 3+6]
        let res_data = result.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_axis_1() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = sum_axes(&t, &[1], false).unwrap();
        assert_eq!(result.shape(), vec![2]);
        let expected_data = vec![6.0, 15.0]; // [1+2+3, 4+5+6]
        let res_data = result.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_axes_multiple() {
        let t = create_test_tensor(
            (1..=24).map(|x| x as f64).collect(),
            vec![2, 3, 4],
        );
        // Sum over axes 0 and 2
        let result = sum_axes(&t, &[0, 2], false).unwrap();
        // Expected output shape [3]
        // Block 0, Row 0: 1+2+3+4 = 10
        // Block 0, Row 1: 5+6+7+8 = 26
        // Block 0, Row 2: 9+10+11+12 = 42
        // Block 1, Row 0: 13+14+15+16 = 58
        // Block 1, Row 1: 17+18+19+20 = 74
        // Block 1, Row 2: 21+22+23+24 = 90
        // Sum Block 0 + Block 1 for each row:
        // [10+58, 26+74, 42+90] = [68, 100, 132]
        assert_eq!(result.shape(), vec![3]);
        let expected_data = vec![68.0, 100.0, 132.0];
        let res_data = result.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());
    }

    #[test]
    fn test_sum_keep_dims() {
        let t = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Sum over axis 0, keep dims
        let result0 = sum_axes(&t, &[0], true).unwrap();
        assert_eq!(result0.shape(), vec![1, 3]);
        let expected_data0 = vec![5.0, 7.0, 9.0];
        let res_data0 = result0.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(res_data0.as_slice(), expected_data0.as_slice());

        // Sum over axis 1, keep dims
        let result1 = sum_axes(&t, &[1], true).unwrap();
        assert_eq!(result1.shape(), vec![2, 1]);
        let expected_data1 = vec![6.0, 15.0];
        let res_data1 = result1.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(res_data1.as_slice(), expected_data1.as_slice());

        // Sum all, keep dims
        let result_all = sum_axes(&t, &[], true).unwrap();
        assert_eq!(result_all.shape(), vec![1, 1]); // Should this be [1, 1] or input shape [2,3]? Let's assume [1,1]
        let expected_data_all = vec![21.0];
        let res_data_all = result_all.read_data().data.cpu_data().unwrap().clone();
        assert_relative_eq!(res_data_all.as_slice(), expected_data_all.as_slice());
    }

    #[test]
    fn test_sum_invalid_axis() {
        let t = create_test_tensor(vec![1.0, 2.0], vec![2]);
        let result = sum_axes(&t, &[1], false);
        assert!(matches!(result, Err(NeuraRustError::IndexOutOfBounds { .. })));

        let result_empty = sum_axes(&t, &[0, 1, 2], false);
        assert!(matches!(result_empty, Err(NeuraRustError::IndexOutOfBounds { .. })));
    }
}


#[cfg(test)]
mod autograd_tests {
    // REMOVED: use super::*;
    use crate::ops::reduction::sum::sum_axes; // Explicit import
    use crate::autograd::grad_check::check_grad;
    use crate::error::NeuraRustError;
    use crate::tensor::Tensor;
    use crate::utils::testing::create_test_tensor_with_grad;

    #[test]
    fn test_sum_axes_backward_simple_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[0], true)
        };

        let output_shape = vec![1, 3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

     #[test]
    fn test_sum_axes_backward_simple_no_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[1], false)
        };

        let output_shape = vec![2];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

    #[test]
    fn test_sum_all_backward_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f32>]| -> Result<Tensor<f32>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[], true)
        };

        let output_shape = vec![1, 1];
        let output_grad = Tensor::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 5e-2;

        let grad_check_result = check_grad(func, &[input.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

    #[test]
    fn test_sum_all_backward_no_keep_dims() {
        let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f32>]| -> Result<Tensor<f32>, NeuraRustError> {
             assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[], false)
        };

        let output_shape = vec![];
        let output_grad = Tensor::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 5e-2;

        let grad_check_result = check_grad(func, &[input.clone()], &output_grad, epsilon, tolerance);
         assert!(grad_check_result.is_ok(), "Gradient check failed: {:?}", grad_check_result.err());
    }

     #[test]
    fn test_sum_multiple_axes_backward() {
        let input = create_test_tensor_with_grad((1..=24).map(|x| x as f64).collect::<Vec<_>>(), vec![2, 3, 4]);
        input.set_requires_grad(true).unwrap();

        let func = |inputs: &[Tensor<f64>]| -> Result<Tensor<f64>, NeuraRustError> {
            assert_eq!(inputs.len(), 1);
            sum_axes(&inputs[0], &[0, 2], false)
        };

        let output_shape = vec![3];
        let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
        let epsilon = 1e-5;
        let tolerance = 1e-7;

        let grad_check_result = check_grad(func, &[input], &output_grad, epsilon, tolerance);
        assert!(grad_check_result.is_ok(), "Gradient check failed (f64): {:?}", grad_check_result.err());
    }

      #[test]
     fn test_sum_no_reduction_backward() {
         let input = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
         input.set_requires_grad(true).unwrap();

         let epsilon = 1e-5;
         let tolerance = 5e-3;

         let func1 = |inputs: &[Tensor<f32>]| {
             assert_eq!(inputs.len(), 1);
             sum_axes(&inputs[0], &[], false)
        };
         let output1_shape = vec![];
         let output1_grad = Tensor::ones(output1_shape).unwrap();
         let check1 = check_grad(func1, &[input.clone()], &output1_grad, epsilon, tolerance);
         assert!(check1.is_ok(), "Sum all (scalar) grad check failed: {:?}", check1.err());
         input.clear_grad();

         let func2 = |inputs: &[Tensor<f32>]| {
             assert_eq!(inputs.len(), 1);
             sum_axes(&inputs[0], &[], true)
         };
         let output2_shape = vec![1];
         let output2_grad = Tensor::ones(output2_shape).unwrap();
         let check2 = check_grad(func2, &[input.clone()], &output2_grad, epsilon, tolerance);
          assert!(check2.is_ok(), "Sum all (keep_dims) grad check failed: {:?}", check2.err());
     }
} 