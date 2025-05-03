use super::*;
use crate::autograd::grad_check::check_grad;
use approx::assert_relative_eq;
use std::panic;

// Helper to create tensors for tests
fn create_test_tensor<
    T: Clone
        + Debug
        + PartialEq
        + Zero
        + One
        + AddAssign
        + Copy
        + Add<Output = T>
        + Default
        + Sum
        + PartialOrd
        + Send
        + Sync
        + 'static,
>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> Tensor<T> {
    Tensor::new(data, shape).expect("Failed to create test tensor")
}

// Helper to create tensors with requires_grad = true
fn create_test_tensor_with_grad<T>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> Tensor<T>
where
    T: Default + Debug + Clone + Copy + PartialEq + Zero + One + Sum + AddAssign + PartialOrd + Send + Sync + std::iter::Product + 'static
{
    let t = create_test_tensor(data, shape);
    t.set_requires_grad(true)
        .expect("Failed to set requires_grad");
    t
}

#[test]
fn test_add_tensors_ok() {
    // Test case 1: Simple addition
    let a = create_test_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b = create_test_tensor(vec![4.0, 5.0, 6.0], vec![3]);
    let result = add_op(&a, &b).unwrap();
    let expected_data = vec![5.0, 7.0, 9.0];
    assert_eq!(result.shape(), vec![3]);
    let res_data = result.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data.as_slice(), expected_data.as_slice());

    // Test case 2: Scalar addition
    let a_scalar = create_test_tensor(vec![10.0], vec![]);
    let b_scalar = create_test_tensor(vec![5.0], vec![]);
    let result_scalar = add_op(&a_scalar, &b_scalar).unwrap();
    assert_eq!(result_scalar.shape(), vec![]);
    assert_relative_eq!(result_scalar.get(&[]).unwrap(), 15.0);
}

#[test]
fn test_add_tensors_shape_mismatch() {
    // Cas où les dimensions finales diffèrent et aucune n'est 1
    let a = create_test_tensor(vec![1.0, 2.0], vec![2]);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = add_op(&a, &b);
    assert!(matches!(
        result,
        Err(NeuraRustError::BroadcastError { .. })
    ));

    // Retiré : Le cas [2, 2] + [2] est valide par broadcast
    // let c = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // let d = create_test_tensor(vec![1.0, 2.0], vec![2]);
    // let result_cd = add_op(&c, &d);
    // assert!(matches!(result_cd, Err(NeuraRustError::BroadcastError { .. })));
}

#[test]
fn test_add_broadcasting() {
    // Test case 1: Vector + Scalar
    let a = create_test_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b_scalar = create_test_tensor(vec![10.0], vec![]);
    let result1 = add_op(&a, &b_scalar).unwrap();
    assert_eq!(result1.shape(), vec![3]);
    let expected_data1 = vec![11.0, 12.0, 13.0];
    let res_data1 = result1.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data1.as_slice(), expected_data1.as_slice());

    // Test case 2: Matrix [2, 2] + Vector [2]
    // Le vecteur [2] est broadcasté en [1, 2] puis en [2, 2]
    let matrix = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let vector = create_test_tensor(vec![10.0, 20.0], vec![2]); // Shape [2]
    // MODIFIED: Check for successful operation and correct result assuming broadcast to [1, 2]
    let result2 = add_op(&matrix, &vector).unwrap();
    assert_eq!(result2.shape(), vec![2, 2]);
    let expected_data2 = vec![11.0, 22.0, 13.0, 24.0]; // Same as result3, assuming broadcast to [1, 2]
    let res_data2 = result2.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data2.as_slice(), expected_data2.as_slice());

    // Let's test the intended broadcast explicitly: Matrix + Row Vector
    let row_vector = create_test_tensor(vec![10.0, 20.0], vec![1, 2]); // Shape [1, 2]
    let result3 = add_op(&matrix, &row_vector).unwrap();
    assert_eq!(result3.shape(), vec![2, 2]);
    let expected_data3 = vec![11.0, 22.0, 13.0, 24.0]; // [[1+10, 2+20], [3+10, 4+20]]
    let res_data3 = result3.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data3.as_slice(), expected_data3.as_slice());

    // Test case 3: Matrix + Column Vector
    let col_vector = create_test_tensor(vec![10.0, 20.0], vec![2, 1]); // Shape [2, 1]
    let result4 = add_op(&matrix, &col_vector).unwrap();
    assert_eq!(result4.shape(), vec![2, 2]);
    let expected_data4 = vec![11.0, 12.0, 23.0, 24.0]; // [[1+10, 2+10], [3+20, 4+20]]
    let res_data4 = result4.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data4.as_slice(), expected_data4.as_slice());

     // Test case 4: Higher dimensions
    let t1 = create_test_tensor((1..=12).map(|x| x as f64).collect(), vec![2, 3, 2]);
    let t2 = create_test_tensor(vec![10.0, 20.0], vec![1, 1, 2]);
    let result5 = add_op(&t1, &t2).unwrap();
    assert_eq!(result5.shape(), vec![2, 3, 2]);
    let expected_data5 = vec![
        11.0, 22.0, 13.0, 24.0, 15.0, 26.0, // First 3x2 block
        17.0, 28.0, 19.0, 30.0, 21.0, 32.0, // Second 3x2 block
    ];
    let res_data5 = result5.read_data().data.cpu_data().unwrap().clone();
    assert_relative_eq!(res_data5.as_slice(), expected_data5.as_slice());
}

// --- Autograd Tests ---

#[test]
fn test_add_backward_simple() {
    let a = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
    let b = create_test_tensor_with_grad(vec![4.0, 5.0, 6.0], vec![3]);

    let func = |inputs: &[Tensor<f64>]| add_op(&inputs[0], &inputs[1]);

    let output_shape = vec![3];
    let output_grad = Tensor::<f64>::ones(output_shape).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result = check_grad(func, &[a, b], &output_grad, epsilon, tolerance);

    // Check if gradients were computed and stored correctly (optional detailed check)
    // let a_grad = a.grad().unwrap();
    // let b_grad = b.grad().unwrap();
    // let expected_grad = vec![1.0, 1.0, 1.0];
    // assert_relative_eq!(a_grad.data().as_slice(), expected_grad.as_slice());
    // assert_relative_eq!(b_grad.data().as_slice(), expected_grad.as_slice());

     assert!(grad_check_result.is_ok(), "Simple add backward grad check failed: {:?}", grad_check_result.err());
}

#[test]
fn test_add_backward_broadcast() {
    // Test case 1: Matrix + Row Vector
    let matrix = create_test_tensor_with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let row_vector = create_test_tensor_with_grad(vec![10.0, 20.0], vec![1, 2]);

    let func1 = |inputs: &[Tensor<f64>]| add_op(&inputs[0], &inputs[1]);

    let output_shape1 = vec![2, 2];
    let output_grad1 = Tensor::<f64>::ones(output_shape1).unwrap();
    let epsilon = 1e-5;
    let tolerance = 1e-7;

    let grad_check_result1 = check_grad(func1, &[matrix.clone(), row_vector.clone()], &output_grad1, epsilon, tolerance);
    assert!(grad_check_result1.is_ok(), "Broadcast add (matrix+row) backward grad check failed: {:?}", grad_check_result1.err());

    // Optional detailed check for row_vector gradient (should sum along dim 0)
    // let row_grad = row_vector.grad().unwrap();
    // assert_eq!(row_grad.shape(), vec![1, 2]);
    // let expected_row_grad = vec![2.0, 2.0]; // Sum of 1s along dim 0
    // assert_relative_eq!(row_grad.data().as_slice(), expected_row_grad.as_slice());

    // Test case 2: Matrix + Column Vector
    let col_vector = create_test_tensor_with_grad(vec![10.0, 20.0], vec![2, 1]);

    let func2 = |inputs: &[Tensor<f64>]| add_op(&inputs[0], &inputs[1]);

    let output_shape2 = vec![2, 2];
    let output_grad2 = Tensor::<f64>::ones(output_shape2).unwrap();

    let grad_check_result2 = check_grad(func2, &[matrix.clone(), col_vector.clone()], &output_grad2, epsilon, tolerance);
     assert!(grad_check_result2.is_ok(), "Broadcast add (matrix+col) backward grad check failed: {:?}", grad_check_result2.err());

      // Optional detailed check for col_vector gradient (should sum along dim 1)
    // let col_grad = col_vector.grad().unwrap();
    // assert_eq!(col_grad.shape(), vec![2, 1]);
    // let expected_col_grad = vec![2.0, 2.0]; // Sum of 1s along dim 1
    // assert_relative_eq!(col_grad.data().as_slice(), expected_col_grad.as_slice());
} 