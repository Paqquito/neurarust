use crate::device::StorageDevice as Device;
use crate::tensor::Tensor;
use crate::nn::Parameter;
use crate::error::NeuraRustError;
use crate::types::DType;
// use crate::tensor::{zeros, zeros_f64, ones, ones_f64}; // Import global supprimé

/// The base trait for all neural network modules (layers, containers, etc.).
///
/// This trait defines the fundamental operations that any neural network module
/// should support, such as performing a forward pass and accessing its parameters.
/// It is designed to be generic over the data type `T` (e.g., `f32`, `f64`).
pub trait Module: std::fmt::Debug + Send + Sync {
    /// Performs a forward pass of the module.
    ///
    /// # Arguments
    /// * `input`: A reference to the input `Tensor` for the module.
    ///
    /// # Returns
    /// A `Result` containing the output `Tensor` of the module, or a `NeuraRustError`
    /// if an error occurs during the forward pass.
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError>;

    /// Returns a vector of all learnable parameters (`Parameter` instances) of the module.
    ///
    /// Parameters are typically weights and biases of layers that are adjusted during training.
    /// This method should collect all such parameters, including those from sub-modules.
    fn parameters(&self) -> Vec<&Parameter>;

    /// Returns a vector of all learnable parameters (`Parameter` instances) of the module
    /// along with their names.
    /// Names should be unique within the module and follow a hierarchical structure for nested modules
    /// (e.g., "layer1.weight", "layer1.bias").
    fn named_parameters(&self) -> Vec<(String, &Parameter)>;

    /// Returns a vector of direct child `Module`s.
    /// For modules that do not contain other modules, this should return an empty vector.
    fn children(&self) -> Vec<&dyn Module> {
        Vec::new()
    }

    /// Returns a vector of direct child `Module`s along with their names.
    /// Names are typically the field names under which the children are stored.
    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// Returns an iterator over all modules in the tree (self + all descendants), depth-first.
    fn modules(&self) -> Vec<&dyn Module>;

    /// Sets the device for all parameters of the module.
    ///
    /// # Arguments
    /// * `device`: The target `Device` (e.g., `Device::Cpu`, `Device::Gpu`).
    ///
    /// # Errors
    /// Returns `NeuraRustError` if any parameter fails to move to the specified device.
    fn to_device(&mut self, _device: Device) -> Result<(), NeuraRustError> {
        for _param in self.parameters() {
            // param.to_device(device)?; // Commenté temporairement
        }
        Ok(())
    }

    /// Sets the data type for all parameters of the module.
    ///
    /// # Arguments
    /// * `dtype`: The target `DType` (e.g., `DType::F32`, `DType::F64`).
    ///
    /// # Errors
    /// Returns `NeuraRustError` if any parameter fails to change its data type.
    fn to_dtype(&mut self, _dtype: DType) -> Result<(), NeuraRustError> {
        for _param in self.parameters() {
            // param.to_dtype(dtype)?; // Commenté temporairement
        }
        Ok(())
    }
}

// Example of how a simple container might implement it (not needed yet)
/*
use std::collections::BTreeMap;

pub struct Sequential {
    modules: BTreeMap<String, Box<dyn Module>>,
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let mut current_input = input.clone(); // Start with the initial input
        for (_name, module) in &self.modules {
            current_input = module.forward(&current_input)?;
        }
        Ok(current_input)
    }

    fn parameters(&self) -> Vec<Parameter> {
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
    use super::*; // Garde les imports du module parent comme Module, Parameter, DType, Device
    use crate::tensor::{zeros, zeros_f64, ones, ones_f64}; // Ajout des imports spécifiques au test
    use crate::nn::parameter::Parameter; // Parameter est déjà dans super::*, mais peut être gardé pour clarté ou si super::* change
    use crate::types::DType; // Idem pour DType
    use crate::device::StorageDevice as Device; // Idem pour Device

    // Mock Module pour les tests
    #[derive(Debug)]
    struct MockModule {
        param: Parameter,
    }

    impl MockModule {
        fn new(dtype: DType) -> Result<Self, NeuraRustError> {
            let tensor = match dtype {
                DType::F32 => zeros(&[1])?,
                DType::F64 => zeros_f64(&[1])?,
            };
            Ok(Self { param: Parameter::new_unnamed(tensor) })
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter> {
            vec![&self.param]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter)> {
            let name = self.param.name().unwrap_or("param").to_string();
            vec![(name, &self.param)]
        }

        fn to_device(&mut self, device: Device) -> Result<(), NeuraRustError> {
            self.param.to_device(device)
        }

        fn to_dtype(&mut self, dtype: DType) -> Result<(), NeuraRustError> {
            self.param.to_dtype(dtype)
        }

        fn modules(&self) -> Vec<&dyn Module> {
            vec![self]
        }
    }

    #[test]
    fn test_module_parameters_retrieval() -> Result<(), NeuraRustError> {
        let module_f32 = MockModule::new(DType::F32)?;
        let input_f32 = ones(&[2, 2])?;
        let _ = module_f32.forward(&input_f32)?;
        let params_f32 = module_f32.parameters();
        assert_eq!(params_f32.len(), 1, "Expected 1 parameter for F32 module");
        assert_eq!(params_f32[0].shape(), &[1], "Parameter shape mismatch");
        assert_eq!(params_f32[0].dtype(), DType::F32, "Parameter DType mismatch");

        let module_f64 = MockModule::new(DType::F64)?;
        let input_f64 = ones_f64(&[3, 1])?;
        let _ = module_f64.forward(&input_f64)?;
        let params_f64 = module_f64.parameters();
        assert_eq!(params_f64.len(), 1, "Expected 1 parameter for F64 module");
        assert_eq!(params_f64[0].shape(), &[1], "Parameter shape mismatch");
        assert_eq!(params_f64[0].dtype(), DType::F64, "Parameter DType mismatch");

        Ok(())
    }

    #[test]
    fn test_module_to_device() -> Result<(), NeuraRustError> {
        let mut module = MockModule::new(DType::F32)?;
        module.to_device(Device::CPU)?;
        assert_eq!(module.parameters()[0].device(), Device::CPU, "Parameter should be on CPU");
        Ok(())
    }

    #[test]
    fn test_module_to_dtype() -> Result<(), NeuraRustError> {
        let mut module = MockModule::new(DType::F32)?;
        assert_eq!(module.parameters()[0].dtype(), DType::F32, "Initial DType should be F32");

        module.to_dtype(DType::F64)?;
        assert_eq!(module.parameters()[0].dtype(), DType::F64, "DType should be F64 after conversion");
        let params_f64 = module.parameters();
        let data_f64 = params_f64[0].get_f64_data()?;
        assert_eq!(data_f64, vec![0.0f64], "Data should be 0.0_f64 after conversion");

        module.to_dtype(DType::F32)?;
        assert_eq!(module.parameters()[0].dtype(), DType::F32, "DType should be F32 after converting back");
        let params_f32 = module.parameters();
        let data_f32 = params_f32[0].get_f32_data()?;
        assert_eq!(data_f32, vec![0.0f32], "Data should be 0.0_f32 after conversion back");

        module.to_dtype(DType::F32)?;
        assert_eq!(module.parameters()[0].dtype(), DType::F32, "DType should remain F32");
        Ok(())
    }

    #[test]
    fn test_mock_module_named_parameters() -> Result<(), NeuraRustError> {
        let tensor = zeros(&[1])?;
        let named_param = Parameter::new(tensor.clone(), Some("custom_mock_name".to_string()));
        let module_named = MockModule { param: named_param };
        let named_params1 = module_named.named_parameters();
        assert_eq!(named_params1.len(), 1);
        assert_eq!(named_params1[0].0, "custom_mock_name");
        assert_eq!(named_params1[0].1.name(), Some("custom_mock_name"));

        let unnamed_param = Parameter::new_unnamed(tensor);
        let module_unnamed = MockModule { param: unnamed_param };
        let named_params2 = module_unnamed.named_parameters();
        assert_eq!(named_params2.len(), 1);
        assert_eq!(named_params2[0].0, "param"); // Default name
        assert_eq!(named_params2[0].1.name(), None);
        Ok(())
    }

    #[test]
    fn test_mock_module_children_and_modules() -> Result<(), NeuraRustError> {
        let module = MockModule::new(DType::F32)?;

        let children = module.children();
        assert!(children.is_empty(), "MockModule (leaf) should have no children");

        let named_children = module.named_children();
        assert!(named_children.is_empty(), "MockModule (leaf) should have no named children");

        let modules = module.modules();
        assert_eq!(modules.len(), 1, "MockModule (leaf) should return itself in modules()");
        // Pour vérifier que c'est bien self, on peut comparer les adresses, mais c'est un peu délicat avec &dyn Module.
        // On peut se contenter de vérifier le type si possible, ou indirectement via parameters.
        assert_eq!(modules[0].parameters().len(), 1, "The module in modules() should be the MockModule itself");
        Ok(())
    }
} 