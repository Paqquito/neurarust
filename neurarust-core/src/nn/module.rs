use crate::tensor::Tensor;
use std::fmt::Debug;
use crate::error::NeuraRustError;
use crate::nn::Parameter;

/// The base trait for all neural network modules (layers, containers, etc.).
pub trait Module<T>: Debug {
    /// Performs the forward pass of the module.
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError>;

    /// Returns a list of all learnable parameters within the module.
    /// Should return owned Parameters for optimizer updates.
    fn parameters(&self) -> Vec<Parameter<T>>;
}

// Example of how a simple container might implement it (not needed yet)
/*
use std::collections::BTreeMap;

pub struct Sequential<T> {
    modules: BTreeMap<String, Box<dyn Module<T>>>,
}

impl<T: 'static> Module<T> for Sequential<T> {
    fn forward(&self, input: &Tensor<T>) -> Tensor<T> {
        let mut current_input = input.clone(); // Start with the initial input
        for (_name, module) in &self.modules {
            current_input = module.forward(&current_input);
        }
        current_input
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        let mut params = Vec::new();
        for (_name, module) in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::types::{DType, NeuraNumeric};
    use crate::nn::Parameter;
    use std::fmt::Debug;

    // Définir un DType par défaut pour T générique
    trait DefaultDType {
        fn default_dtype() -> DType;
    }

    impl DefaultDType for f32 {
        fn default_dtype() -> DType { DType::F32 }
    }

    impl DefaultDType for f64 {
        fn default_dtype() -> DType { DType::F64 }
    }

    #[derive(Debug)]
    struct DummyModule<T: NeuraNumeric + DefaultDType> {
        param: Parameter<T>,
    }

    impl<T: NeuraNumeric + Copy + Debug + Default + DefaultDType> DummyModule<T> {
        fn new() -> Self {
            // Crée un tenseur simple avec le type T et son DType par défaut
            let tensor_t = Tensor::<T>::zeros(vec![1], T::default_dtype());
            DummyModule {
                param: Parameter::new(tensor_t),
            }
        }
    }

    impl<T: NeuraNumeric + Copy + Debug + Default + DefaultDType> Module<T> for DummyModule<T> {
        fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, NeuraRustError> {
            // Opération factice : retourne l'input cloné (pour tester la signature)
            // Une vraie opération pourrait impliquer self.param
            Ok(input.clone()) 
        }

        fn parameters(&self) -> Vec<Parameter<T>> {
            vec![self.param.clone()]
        }
    }

    #[test]
    fn test_dummy_module_f32_compiles_and_runs() {
        let module = DummyModule::<f32>::new();
        let input_tensor = Tensor::<f32>::ones(vec![2, 2], DType::F32);
        
        // Test forward
        let output = module.forward(&input_tensor).expect("Forward pass failed");
        assert_eq!(output.shape(), input_tensor.shape(), "Output shape mismatch");
        assert_eq!(output.dtype(), input_tensor.dtype(), "Output DType mismatch");

        // Test parameters
        let params = module.parameters();
        assert_eq!(params.len(), 1, "Incorrect number of parameters");
        assert!(params[0].requires_grad(), "Parameter should require grad");
        assert_eq!(params[0].shape(), &vec![1], "Parameter shape mismatch");
    }

    #[test]
    fn test_dummy_module_f64_compiles_and_runs() {
        let module = DummyModule::<f64>::new();
        let input_tensor = Tensor::<f64>::ones(vec![3, 1], DType::F64);
        
        // Test forward
        let output = module.forward(&input_tensor).expect("Forward pass failed");
        assert_eq!(output.shape(), input_tensor.shape(), "Output shape mismatch");
        assert_eq!(output.dtype(), input_tensor.dtype(), "Output DType mismatch");

        // Test parameters
        let params = module.parameters();
        assert_eq!(params.len(), 1, "Incorrect number of parameters");
        assert!(params[0].requires_grad(), "Parameter should require grad");
        assert_eq!(params[0].shape(), &vec![1], "Parameter shape mismatch");
    }
} 