use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use std::sync::{Arc, RwLock};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
    named_modules: BTreeMap<String, usize>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            modules: Vec::new(),
            named_modules: BTreeMap::new(),
        }
    }

    pub fn add_module(&mut self, name: &str, module: Box<dyn Module>) {
        let index = self.modules.len();
        self.modules.push(module);
        self.named_modules.insert(name.to_string(), index);
    }

    pub fn modules_list(&self) -> &Vec<Box<dyn Module>> {
        &self.modules
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let mut current_input = input.clone();
        for module in &self.modules {
            current_input = module.forward(&current_input)?;
        }
        Ok(current_input)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
        let mut params = Vec::new();
        for (name, &index) in &self.named_modules {
            if let Some(module) = self.modules.get(index) {
                for (param_name, param_arc) in module.named_parameters() {
                    params.push((format!("{}.{}", name, param_name), param_arc));
                }
            }
        }
        params
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.modules.iter().map(|m| m.as_ref()).collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children_vec = Vec::new();
        for (name, &index) in &self.named_modules {
             if let Some(module) = self.modules.get(index) {
                children_vec.push((name.clone(), module.as_ref()));
            }
        }
        children_vec
    }
    
    fn modules(&self) -> Vec<&dyn Module> {
        let mut all_modules = vec![self as &dyn Module];
        for module_box in &self.modules {
            all_modules.extend(module_box.modules());
        }
        all_modules
    }

    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
        // f(self); // Apply to self first, then children
        for module_box in &mut self.modules {
            module_box.apply(f);
        }
        f(self); // Ou apply to self last
    }
} 