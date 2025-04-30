use crate::Dataset;
use neurarust_core::tensor::Tensor;

/// A simple dataset implementation backed by Vectors of Tensors.
///
/// Assumes that the i-th element of `inputs` corresponds to the i-th element
/// of `targets`.
#[derive(Debug, Clone)]
pub struct VecDataset<InputType, TargetType>
where
    InputType: Clone + 'static,
    TargetType: Clone + 'static,
{
    inputs: Vec<Tensor<InputType>>,
    targets: Vec<Tensor<TargetType>>,
}

impl<InputType, TargetType> VecDataset<InputType, TargetType>
where
    InputType: Clone + 'static,
    TargetType: Clone + 'static,
{
    /// Creates a new VecDataset from input and target vectors.
    ///
    /// # Panics
    /// Panics if the lengths of `inputs` and `targets` vectors are different.
    pub fn new(inputs: Vec<Tensor<InputType>>, targets: Vec<Tensor<TargetType>>) -> Self {
        assert_eq!(inputs.len(), targets.len(),
            "Input and target vectors must have the same length. Got {} and {}.",
            inputs.len(), targets.len()
        );
        VecDataset { inputs, targets }
    }
}

impl<InputType, TargetType> Dataset for VecDataset<InputType, TargetType>
where
    InputType: Clone + 'static,
    TargetType: Clone + 'static,
{
    /// The item type is a tuple containing cloned Tensors for input and target.
    type Item = (Tensor<InputType>, Tensor<TargetType>);

    /// Returns clones of the input and target tensors at the specified index.
    fn get(&self, index: usize) -> Self::Item {
        assert!(index < self.len(), "Index out of bounds: {} >= {}", index, self.len());
        // Clone the tensors to return owned copies
        (self.inputs[index].clone(), self.targets[index].clone())
    }

    /// Returns the number of samples (length of the input/target vectors).
    fn len(&self) -> usize {
        self.inputs.len() // inputs and targets have the same length due to assert in new()
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use neurarust_core::tensor::Tensor;

    #[test]
    fn test_vec_dataset_creation_and_len() {
        // Handle Result from Tensor::new using expect
        let inputs = vec![
            Tensor::new(vec![1.0f32, 2.0], vec![2]).expect("Input 1 failed"),
            Tensor::new(vec![3.0, 4.0], vec![2]).expect("Input 2 failed")
        ];
        let targets = vec![
            Tensor::new(vec![0i32], vec![1]).expect("Target 1 failed"), // Specify i32 for target type clarity
            Tensor::new(vec![1i32], vec![1]).expect("Target 2 failed")
        ];
        // Now inputs/targets are Vec<Tensor<T>>, VecDataset::new should work
        let dataset = VecDataset::new(inputs, targets);
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());

        let empty_dataset: VecDataset<f32, i32> = VecDataset::new(vec![], vec![]);
        assert_eq!(empty_dataset.len(), 0);
        assert!(empty_dataset.is_empty());
    }

    #[test]
    #[should_panic(expected = "Input and target vectors must have the same length")]
    fn test_vec_dataset_creation_panic() {
        // Handle Result
        let inputs = vec![Tensor::new(vec![1.0f32], vec![1]).expect("Input creation failed")];
        let targets: Vec<Tensor<i32>> = vec![];
        let _dataset = VecDataset::new(inputs, targets);
    }

    #[test]
    fn test_vec_dataset_get() {
        // Handle Result
        let input1 = Tensor::new(vec![1.0f32, 2.0], vec![2]).expect("Input 1 failed");
        let target1 = Tensor::new(vec![0i32], vec![1]).expect("Target 1 failed"); // Specify i32
        let input2 = Tensor::new(vec![3.0, 4.0], vec![2]).expect("Input 2 failed");
        let target2 = Tensor::new(vec![1i32], vec![1]).expect("Target 2 failed"); // Specify i32

        // Clone Tensors, not Results
        let dataset = VecDataset::new(
            vec![input1.clone(), input2.clone()],
            vec![target1.clone(), target2.clone()]
        );

        let (retrieved_input1, retrieved_target1) = dataset.get(0);
        // Compare Tensor with Tensor
        assert_eq!(retrieved_input1, input1);
        assert_eq!(retrieved_target1, target1);

        let (retrieved_input2, retrieved_target2) = dataset.get(1);
        assert_eq!(retrieved_input2, input2);
        assert_eq!(retrieved_target2, target2);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_vec_dataset_get_panic() {
        let empty_dataset: VecDataset<f32, i32> = VecDataset::new(vec![], vec![]);
        empty_dataset.get(0); // This should panic
    }
}