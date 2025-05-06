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
pub trait Module {
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
    fn parameters(&self) -> Vec<Parameter>;

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
    struct MockModule {
        param: Parameter,
    }

    impl MockModule {
        fn new(dtype: DType) -> Result<Self, NeuraRustError> {
            let tensor = match dtype {
                DType::F32 => zeros(&[1])?,
                DType::F64 => zeros_f64(&[1])?,
            };
            Ok(Self { param: Parameter::new(tensor) })
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<Parameter> {
            vec![self.param.clone()]
        }

        fn to_device(&mut self, device: Device) -> Result<(), NeuraRustError> {
            self.param.to_device(device)
        }

        fn to_dtype(&mut self, dtype: DType) -> Result<(), NeuraRustError> {
            self.param.to_dtype(dtype)
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
} 