use std::sync::{Arc, RwLock};
use neurarust_core::{
    device::StorageDevice,
    types::DType,
    error::NeuraRustError,
    nn::{
        layers::linear::Linear, 
        layers::relu::ReLU, 
        losses::mse::MSELoss, 
        module::Module,
        parameter::Parameter,
    },
    optim::{
        adam::AdamOptimizer,
        lr_scheduler::{StepLR, LRScheduler},
        optimizer_trait::Optimizer,
        param_group::ParamGroup,
        sgd::SgdOptimizer,
        clip_grad_norm_,
    },
    tensor::Tensor,
    tensor::from_vec_f32,
};

// Define a simple MLP (Multi-Layer Perceptron)
// Input (batch, in_features) -> Linear -> ReLU -> Linear -> Output (batch, out_features)
#[derive(Debug)]
struct SimpleMLP {
    linear1: Linear,
    relu1: ReLU,
    linear2: Linear,
    name: String,
}

impl SimpleMLP {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize, _device: StorageDevice) -> Result<Self, NeuraRustError> {
        let linear1 = Linear::new(in_features, hidden_features, true, DType::F32)?;
        let relu1 = ReLU::new();
        let linear2 = Linear::new(hidden_features, out_features, true, DType::F32)?;
        Ok(SimpleMLP {
            linear1,
            relu1,
            linear2,
            name: "mlp".to_string(),
        })
    }
}

impl Module for SimpleMLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let x1 = self.linear1.forward(input)?;
        let x2 = self.relu1.forward(&x1)?;
        self.linear2.forward(&x2)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>) > {
        let mut named_params = Vec::new();
        for (param_local_name, param_arc) in self.linear1.named_parameters() {
            named_params.push((format!("{}.linear1.{}", self.name, param_local_name), param_arc));
        }
        for (param_local_name, param_arc) in self.linear2.named_parameters() {
            named_params.push((format!("{}.linear2.{}", self.name, param_local_name), param_arc));
        }
        named_params
    }

    fn children(&self) -> Vec<&dyn Module> {
        vec![&self.linear1, &self.relu1, &self.linear2]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            (format!("{}.linear1", self.name), &self.linear1 as &dyn Module),
            (format!("{}.relu1", self.name), &self.relu1 as &dyn Module),
            (format!("{}.linear2", self.name), &self.linear2 as &dyn Module),
        ]
    }
    
    fn modules(&self) -> Vec<&dyn Module> {
        let mut modules_vec: Vec<&dyn Module> = vec![self as &dyn Module];
        modules_vec.extend(self.linear1.modules());
        modules_vec.extend(self.relu1.modules());
        modules_vec.extend(self.linear2.modules());
        modules_vec
    }

    fn to_device(&mut self, device: StorageDevice) -> Result<(), NeuraRustError> {
        self.linear1.to_device(device)?;
        self.linear2.to_device(device)?;
        Ok(())
    }

    fn to_dtype(&mut self, dtype: DType) -> Result<(), NeuraRustError> {
        self.linear1.to_dtype(dtype)?;
        self.linear2.to_dtype(dtype)?;
        Ok(())
    }

    fn apply(&mut self, f: &mut dyn for<'a> FnMut(&'a mut (dyn Module + 'static))) {
        f(self);
        self.linear1.apply(f);
        self.relu1.apply(f);
        self.linear2.apply(f);
    }
}

fn main() -> Result<(), NeuraRustError> {
    println!("Starting Advanced Training Techniques Example...");

    // Configuration
    let batch_size = 64;
    let input_features = 10;
    let hidden_features = 20;
    let output_features = 1;
    let num_epochs = 10;
    let base_learning_rate = 0.01;
    let device = StorageDevice::CPU;

    // 1. Create Synthetic Data (simple linear relationship with noise)
    // X: (batch_size, input_features)
    // Y: (batch_size, output_features)
    let mut x_data = Vec::with_capacity(batch_size * input_features);
    let mut y_data = Vec::with_capacity(batch_size * output_features);
    let true_weights: Vec<f32> = (0..input_features).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect(); // e.g. -0.5, 0.0, 0.5 ...
    let true_bias: f32 = 0.5;

    for i in 0..batch_size {
        let mut y_val: f32 = true_bias;
        for j in 0..input_features {
            let val = ((i * input_features + j) % 100) as f32 / 50.0 - 1.0; // Values between -1 and 1
            x_data.push(val);
            y_val += true_weights[j] * val;
        }
        // Add some noise to y_val
        y_data.push(y_val + ( (i % 10) as f32 / 20.0 - 0.25) ); // Small noise
    }

    let x_train = from_vec_f32(x_data, vec![batch_size, input_features])?;
    let y_train = from_vec_f32(y_data, vec![batch_size, output_features])?;

    // 2. Instantiate Model and Loss
    let mut model = SimpleMLP::new(input_features, hidden_features, output_features, device)?;
    model.to_device(device)?;
    let mse_loss = MSELoss::new("mean");

    // --- Optimizer Setup (Demonstrating SGD with Parameter Groups) ---
    println!("\n--- Using SGD Optimizer with Parameter Groups ---");
    let mut weight_params_vec = Vec::new();
    let mut bias_params_vec = Vec::new();

    for (name, param_arc) in model.named_parameters() {
        if name.ends_with(".weight") { 
            weight_params_vec.push(param_arc.clone());
        } else if name.ends_with(".bias") {
            bias_params_vec.push(param_arc.clone());
        } else {
            weight_params_vec.push(param_arc.clone()); 
        }
    }
    
    let bias_lr_val = base_learning_rate * 2.0;

    let mut weight_pg = ParamGroup::new(weight_params_vec);
    weight_pg.options.lr = Some(base_learning_rate);
    weight_pg.options.weight_decay = Some(0.001);

    let mut bias_pg = ParamGroup::new(bias_params_vec);
    bias_pg.options.lr = Some(bias_lr_val);
    bias_pg.options.weight_decay = Some(0.0);

    let mut sgd_optimizer = SgdOptimizer::new(
        weight_pg.params.clone(),
        base_learning_rate,
        0.0,
        0.0,
        weight_pg.options.weight_decay.unwrap_or(0.0),
        false
    )?;
    sgd_optimizer.add_param_group(bias_pg);
    
    // --- LR Scheduler Setup (StepLR with SGD) ---
    let step_size = 5;
    let gamma = 0.5;   
    let mut sgd_scheduler = StepLR::new(&mut sgd_optimizer, step_size, gamma);


    // --- Training Loop ---
    println!("\nStarting training with SGD, Param Groups, LR Scheduler, and Grad Clipping...");
    let max_grad_norm = 1.0;
    let grad_norm_type = 2.0;

    for epoch in 0..num_epochs {
        // Get current LRs for logging - via scheduler.optimizer()
        let current_lrs_sgd: Vec<f32> = sgd_scheduler.optimizer().param_groups()
            .iter()
            .filter_map(|pg| pg.options.lr)
            .collect();
        // Alternativement, si get_last_lr() est à jour avant le step du scheduler:
        // let current_lrs_sgd = sgd_scheduler.get_last_lr(); 
        let lrs_str_sgd: Vec<String> = current_lrs_sgd.iter().map(|lr| format!("{:.4e}", lr)).collect();

        let predictions = model.forward(&x_train)?;
        let loss_tensor = mse_loss.calculate(&predictions, &y_train)?;
        let loss_val = loss_tensor.item_f32()?;

        // Access optimizer methods via scheduler
        sgd_scheduler.optimizer_mut().zero_grad();
        loss_tensor.backward(None)?;

        let total_norm = clip_grad_norm_(
            model.parameters().into_iter(), // model.parameters() est toujours ok
            max_grad_norm,
            grad_norm_type
        )?;
        
        sgd_scheduler.optimizer_mut().step()?;
        // Scheduler step (modifie les LRs dans l'optimizer via sa réf mut)
        sgd_scheduler.step(Some(epoch as u64), None)?;

        println!(
            "Epoch: {:>3} (SGD), Loss: {:.6e}, Total Grad Norm: {:.4e}, LRs: [{}]", 
            epoch + 1, 
            loss_val, 
            total_norm, 
            lrs_str_sgd.join(", ")
        );
    }

    println!("\nTraining finished with SGD.");

    // --- Optional: Demonstrate AdamW --- 
    println!("\n--- Now trying with AdamW Optimizer (re-initialize model) ---");
    let mut model_adam = SimpleMLP::new(input_features, hidden_features, output_features, device)?;
    model_adam.to_device(device)?;
    
    let mut adam_optimizer = AdamOptimizer::new(
        model_adam.parameters(), 
        base_learning_rate,
        0.9,    // beta1
        0.999,  // beta2
        1e-8,   // eps
        0.01,   // weight_decay
        false   // amsgrad
    )?;

    let mut adam_scheduler = StepLR::new(&mut adam_optimizer, step_size, gamma);
    
    // --- Training Loop (AdamW) ---
    println!("\nStarting training with AdamW, LR Scheduler, and Grad Clipping...");
    for epoch in 0..num_epochs {
        // Get current LRs for Adam optimizer - via scheduler.optimizer()
        let current_lrs_adam: Vec<f32> = adam_scheduler.optimizer().param_groups()
            .iter()
            .filter_map(|pg| pg.options.lr)
            .collect();
        let lrs_str_adam: Vec<String> = current_lrs_adam.iter().map(|lr| format!("{:.4e}", lr)).collect();

        let predictions = model_adam.forward(&x_train)?;
        let loss_tensor = mse_loss.calculate(&predictions, &y_train)?;
        let loss_val = loss_tensor.item_f32()?;

        adam_scheduler.optimizer_mut().zero_grad();
        loss_tensor.backward(None)?;

        let total_norm_adam: f64 = clip_grad_norm_(
            model_adam.parameters().into_iter(),
            max_grad_norm,
            grad_norm_type
        )?;

        adam_scheduler.optimizer_mut().step()?;
        adam_scheduler.step(Some(epoch as u64), Some(loss_val))?;

        println!(
            "Epoch {:>3} (AdamW), Loss: {:.6e}, Total Grad Norm: {:.4e}, LRs: [{}]",
            epoch + 1, 
            loss_val, 
            total_norm_adam,
            lrs_str_adam.join(", ")
        );
    }
    println!("\nTraining finished with AdamW.");

    println!("\nAdvanced Training Techniques Example Finished Successfully!");
    Ok(())
} 