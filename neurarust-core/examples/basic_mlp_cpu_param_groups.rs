// examples/basic_mlp_cpu_param_groups.rs
//!
//! This example demonstrates training a simple Multi-Layer Perceptron (MLP) on the CPU
//! using different learning rates for different parameter groups (weights vs. biases).
//! It showcases how to manually assign parameters to different groups and configure
//! optimizer options per group.

use neurarust_core::{
    device::StorageDevice,
    error::NeuraRustError,
    model::sequential::Sequential,
    nn::{
        layers::{linear::Linear, relu::ReLU},
        losses::mse::MSELoss,
        module::Module,
        parameter::Parameter,
    },
    optim::{
        optimizer_trait::Optimizer,
        param_group::{ParamGroup, ParamGroupOptions},
        sgd::SgdOptimizer,
    },
    tensor::{create::randn, Tensor},
    types::DType,
};
use std::sync::{Arc, RwLock};

// Define a simple MLP
#[derive(Debug)]
struct SimpleMLP {
    seq: Sequential,
}

impl SimpleMLP {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
    ) -> Result<Self, NeuraRustError> {
        let l1 = Linear::new(in_features, hidden_features, true, DType::F32)?;
        let relu1 = ReLU::new();
        let l2 = Linear::new(hidden_features, out_features, true, DType::F32)?;

        let mut seq = Sequential::new();
        seq.add_module("l1", Box::new(l1));
        seq.add_module("relu1", Box::new(relu1));
        seq.add_module("l2", Box::new(l2));

        Ok(SimpleMLP { seq })
    }
}

impl Module for SimpleMLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        self.seq.forward(input)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        self.seq.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
        self.seq.named_parameters()
    }

    fn modules(&self) -> Vec<&dyn Module> {
        let mut mods = vec![self as &dyn Module];
        mods.extend(self.seq.modules());
        mods
    }

    fn children(&self) -> Vec<&dyn Module> {
        vec![&self.seq as &dyn Module]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![("seq".to_string(), &self.seq as &dyn Module)]
    }

    fn to_device(&mut self, device: StorageDevice) -> Result<(), NeuraRustError> {
        self.seq.to_device(device)
    }

    fn to_dtype(&mut self, dtype: DType) -> Result<(), NeuraRustError> {
        self.seq.to_dtype(dtype)
    }

    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
        f(self);
        self.seq.apply(f);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting MLP example with parameter groups...");

    // Hyperparameters
    let input_dim = 10;
    let hidden_dim = 20;
    let output_dim = 5;
    let num_samples = 100;
    let epochs = 50;
    let base_lr = 0.1;
    let bias_lr_multiplier = 0.5;

    // Create model
    let model = Arc::new(RwLock::new(SimpleMLP::new(
        input_dim,
        hidden_dim,
        output_dim,
    )?));
    println!("SimpleMLP created.");

    // Create synthetic data
    let x_data = randn(vec![num_samples, input_dim])?;
    let y_true_data = randn(vec![num_samples, output_dim])?;
    println!("Synthetic data created.");

    // Define Loss Function
    let loss_fn = MSELoss::new("mean");

    // --- Optimizer Setup with Parameter Groups ---
    let model_read_guard = model.read().unwrap();
    
    let mut weight_params: Vec<Arc<RwLock<Parameter>>> = Vec::new();
    let mut bias_params: Vec<Arc<RwLock<Parameter>>> = Vec::new();

    for (name, param) in model_read_guard.named_parameters() {
        if name.ends_with(".weight") {
            weight_params.push(param.clone());
        } else if name.ends_with(".bias") {
            bias_params.push(param.clone());
        }
    }
    
    println!("Number of weight parameters: {}", weight_params.len());
    assert!(!weight_params.is_empty(), "No weight parameters found!");
    println!("Number of bias parameters: {}", bias_params.len());
    assert!(!bias_params.is_empty(), "No bias parameters found!");

    // Initialize optimizer with the first group (weights)
    let mut optimizer = SgdOptimizer::new(
        weight_params, 
        base_lr,
        0.0, 0.0, 0.0, false,
    )?;
    println!("Optimizer created with weight parameters (LR: {}).", base_lr);

    // Add the second group (biases) with different options
    if !bias_params.is_empty() {
        let mut bias_group_options = ParamGroupOptions::default();
        bias_group_options.lr = Some(base_lr * bias_lr_multiplier);
        
        let mut bias_param_group = ParamGroup::new(bias_params);
        bias_param_group.options = bias_group_options;

        optimizer.add_param_group(bias_param_group);
        println!("Bias parameter group added to optimizer (LR: {}).", base_lr * bias_lr_multiplier);
    }
    
    drop(model_read_guard); 

    // --- Training Loop ---
    println!("\nStarting training loop...");
    for epoch in 0..epochs {
        let model_write_guard = model.write().unwrap();
        
        let y_pred = model_write_guard.forward(&x_data)?;
        let loss_val_tensor = loss_fn.calculate(&y_pred, &y_true_data)?;
        let loss_val = loss_val_tensor.item_f32()?;

        loss_val_tensor.backward(None)?;
        optimizer.step()?;
        optimizer.zero_grad();
        
        // Affichage à chaque epoch
        println!("Epoch [{}/{}], Loss: {:.4}", epoch + 1, epochs, loss_val);
    }
    println!("Training finished.");
    Ok(())
} 