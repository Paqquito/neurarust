use crate::device::StorageDevice as Device;
use crate::tensor::Tensor;
use crate::nn::Parameter;
use crate::error::NeuraRustError;
use crate::types::DType;
use std::sync::{Arc, RwLock};
use std::any::Any;
// use crate::tensor::{zeros, zeros_f64, ones, ones_f64}; // Import global supprimé

/// The base trait for all neural network modules (layers, containers, etc.).
///
/// This trait defines the fundamental operations that any neural network module
/// should support, such as performing a forward pass and accessing its parameters.
/// It is designed to be generic over the data type `T` (e.g., `f32`, `f64`).
pub trait Module: std::fmt::Debug + Send + Sync + Any {
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
    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>>;

    /// Returns a vector of all learnable parameters (`Parameter` instances) of the module
    /// along with their names.
    /// Names should be unique within the module and follow a hierarchical structure for nested modules
    /// (e.g., "layer1.weight", "layer1.bias").
    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)>;

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

    /// Applies a function `f` recursively to each sub-module as well as to itself.
    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module));
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
    use crate::tensor::Tensor; 
    // Pas d'import spécifique de `zeros` ici, on utilisera le chemin complet ou un import de module
    use crate::types::DType;
    use crate::device::StorageDevice;
    use crate::NeuraRustError;
    use crate::nn::parameter::Parameter;
    use std::sync::{Arc, RwLock};
    use crate::tensor::create::{zeros, zeros_f64};
    use std::any::Any; // Any est importé implicitement via le trait mais l'avoir ici peut aider

    // Mock Module pour les tests
    #[derive(Debug)]
    struct MockModule {
        param: Arc<RwLock<Parameter>>,
        device: StorageDevice,
        dtype: DType,
    }

    impl MockModule {
        fn new(dtype: DType) -> Result<Self, NeuraRustError> {
            let data = match dtype {
                DType::F32 => zeros(&[1])?,
                DType::F64 => zeros_f64(&[1])?,
                DType::I32 | DType::I64 | DType::Bool => todo!("module: non supporté pour ce DType"),
            };
            let param = Parameter::new(data, Some("mock_param".to_string()));
            Ok(MockModule {
                param: Arc::new(RwLock::new(param)),
                device: StorageDevice::CPU,
                dtype,
            })
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
            vec![Arc::clone(&self.param)]
        }

        fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
            let lock = self.param.read().expect("Failed to acquire read lock");
            let name = lock.name().unwrap_or("param").to_string();
            drop(lock);
            vec![(name, Arc::clone(&self.param))]
        }

        fn children(&self) -> Vec<&dyn Module> {
            vec![]
        }

        fn named_children(&self) -> Vec<(String, &dyn Module)> {
            vec![]
        }

        fn modules(&self) -> Vec<&dyn Module> {
            vec![self]
        }

        fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
            f(self)
        }
    }

    #[test]
    fn test_module_parameter_collection() -> Result<(), NeuraRustError> {
        let module_f32 = MockModule::new(DType::F32)?;
        let params_f32 = module_f32.parameters();
        assert_eq!(params_f32.len(), 1, "Expected 1 parameter for F32 module");
        assert_eq!(params_f32[0].read().unwrap().shape(), &[1], "Parameter shape mismatch");
        assert_eq!(params_f32[0].read().unwrap().dtype(), DType::F32, "Parameter DType mismatch");

        let module_f64 = MockModule::new(DType::F64)?;
        let params_f64 = module_f64.parameters();
        assert_eq!(params_f64.len(), 1, "Expected 1 parameter for F64 module");
        assert_eq!(params_f64[0].read().unwrap().shape(), &[1], "Parameter shape mismatch");
        assert_eq!(params_f64[0].read().unwrap().dtype(), DType::F64, "Parameter DType mismatch");

        Ok(())
    }

    #[test]
    fn test_module_named_parameter_collection() -> Result<(), NeuraRustError> {
        let tensor = zeros(&[1])?;
        let named_param = Parameter::new(tensor.clone(), Some("custom_mock_name".to_string()));
        let module_named = MockModule {
            param: Arc::new(RwLock::new(named_param)),
            device: StorageDevice::CPU,
            dtype: DType::F32,
        };
        let named_params1 = module_named.named_parameters();
        assert_eq!(named_params1.len(), 1);
        assert_eq!(named_params1[0].0, "custom_mock_name");
        assert!(Arc::ptr_eq(&named_params1[0].1, &module_named.param));

        let unnamed_param = Parameter::new_unnamed(tensor);
        let module_unnamed = MockModule {
            param: Arc::new(RwLock::new(unnamed_param)),
            device: StorageDevice::CPU,
            dtype: DType::F32,
        };
        let named_params2 = module_unnamed.named_parameters();
        assert_eq!(named_params2.len(), 1);
        assert_eq!(named_params2[0].0, "param");
        assert!(Arc::ptr_eq(&named_params2[0].1, &module_unnamed.param));

        Ok(())
    }

    #[test]
    fn test_module_device_and_dtype_initial() -> Result<(), NeuraRustError> {
        let module_f32 = MockModule::new(DType::F32)?;
        assert_eq!(module_f32.device, StorageDevice::CPU, "Initial device should be CPU");
        assert_eq!(module_f32.dtype, DType::F32, "Initial dtype should be F32");
        assert_eq!(module_f32.parameters()[0].read().unwrap().device(), StorageDevice::CPU, "Parameter initial device check");
        assert_eq!(module_f32.parameters()[0].read().unwrap().dtype(), DType::F32, "Parameter initial dtype check");

        let module_f64 = MockModule::new(DType::F64)?;
        assert_eq!(module_f64.device, StorageDevice::CPU);
        assert_eq!(module_f64.dtype, DType::F64);
        assert_eq!(module_f64.parameters()[0].read().unwrap().device(), StorageDevice::CPU);
        assert_eq!(module_f64.parameters()[0].read().unwrap().dtype(), DType::F64);

        Ok(())
    }

    #[test]
    fn test_module_modules() -> Result<(), NeuraRustError> {
        let module = MockModule::new(DType::F32)?;
        let modules = module.modules();
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].parameters().len(), 1);
        Ok(())
    }

    #[test]
    fn test_module_apply() -> Result<(), NeuraRustError> {
        let mut module = MockModule::new(DType::F32)?;
        let initial_name = module.named_parameters()[0].0.clone();
        assert_eq!(initial_name, "mock_param");

        let mut rename_count = 0;
        let mut rename_fn = |m: &mut dyn Module| {
            // Forcer la conversion en &mut dyn Any avant d'appeler downcast_mut
            let any_ref: &mut dyn Any = m;
            if let Some(mock) = any_ref.downcast_mut::<MockModule>() {
                let new_name = format!("renamed_{}", rename_count);
                // Accéder directement au champ `name` pour le modifier
                mock.param.write().unwrap().name = Some(new_name);
                rename_count += 1;
            }
        };

        module.apply(&mut rename_fn);

        assert_eq!(rename_count, 1);
        let final_name = module.named_parameters()[0].0.clone();
        assert_eq!(final_name, "renamed_0");

        Ok(())
    }
} 