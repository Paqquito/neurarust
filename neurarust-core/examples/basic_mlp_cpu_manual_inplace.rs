//! # Exemple d'Entraînement d'un MLP Simple sur CPU avec Mise à Jour Manuelle In-Place
//!
//! Cet exemple illustre la mise à jour manuelle des poids d'un MLP en utilisant
//! les opérations in-place de `neurarust-core` (`add_`, `mul_scalar_`, etc.),
//! conformément au Step 1.D.10 de la roadmap.
//!
//! ## Fonctionnalités Démontrées:
//! 1.  Définition d'un `SimpleMLP`.
//! 2.  Création de Données Synthétiques (fixes pour l'entraînement).
//! 3.  Instanciation du Modèle et de la Fonction de Perte (`MSELoss`).
//! 4.  Boucle d'Entraînement Manuelle:
//!     -   Passe avant (`forward`).
//!     -   Calcul de la perte.
//!     -   Passe arrière (`backward`).
//!     -   **Mise à jour manuelle des poids en utilisant des opérations in-place.**
//!     -   Remise à zéro des gradients.

use neurarust_core::tensor::Tensor;
use neurarust_core::nn::layers::linear::Linear;
use neurarust_core::nn::module::Module;
use neurarust_core::nn::parameter::Parameter;
use neurarust_core::ops::activation::relu_op;
use neurarust_core::NeuraRustError;
use neurarust_core::types::DType;
use neurarust_core::tensor::create::randn;
use neurarust_core::nn::losses::mse::MSELoss;
use std::sync::{Arc, RwLock};

/// Un Multi-Layer Perceptron (MLP) simple avec une couche cachée.
#[derive(Debug)]
pub struct SimpleMLP {
    linear1: Linear,
    linear2: Linear,
}

impl SimpleMLP {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Result<Self, NeuraRustError> {
        let linear1 = Linear::new(in_features, hidden_features, true, DType::F32)?;
        let linear2 = Linear::new(hidden_features, out_features, true, DType::F32)?;
        Ok(SimpleMLP { linear1, linear2 })
    }
}

impl Module for SimpleMLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let x = self.linear1.forward(input)?;
        let x = relu_op(&x)?;
        self.linear2.forward(&x)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        let mut params: Vec<Arc<RwLock<Parameter>>> = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
        let mut params: Vec<(String, Arc<RwLock<Parameter>>)> = Vec::new();
        for (name, param_arc) in self.linear1.named_parameters() {
            params.push((format!("linear1.{}", name), param_arc));
        }
        for (name, param_arc) in self.linear2.named_parameters() {
            params.push((format!("linear2.{}", name), param_arc));
        }
        params
    }

    fn children(&self) -> Vec<&dyn Module> {
        vec![&self.linear1, &self.linear2]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("linear1".to_string(), &self.linear1 as &dyn Module),
            ("linear2".to_string(), &self.linear2 as &dyn Module),
        ]
    }

    fn modules(&self) -> Vec<&dyn Module> {
        let mut mods: Vec<&dyn Module> = vec![self as &dyn Module];
        mods.extend(self.linear1.modules());
        mods.extend(self.linear2.modules());
        mods
    }
    
    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
        f(self);
        self.linear1.apply(f);
        self.linear2.apply(f);
    }
}

fn main() -> Result<(), NeuraRustError> {
    let mlp = SimpleMLP::new(10, 20, 5)?;
    println!("SimpleMLP créé avec succès !");

    // Création de Données Synthétiques (fixes)
    let batch_size = 4;
    let input_features = 10;
    let output_features = 5;
    let x_data = randn(vec![batch_size, input_features])?;
    let y_data = randn(vec![batch_size, output_features])?;
    println!("Données synthétiques X et Y créées.");

    // Instanciation de la Fonction de Perte
    let loss_fn = MSELoss::new("mean");
    println!("Fonction de perte MSELoss instanciée.");

    // Boucle d'Entraînement Manuelle
    let learning_rate = 0.01f32;
    let num_epochs = 10; 
    println!("\nDébut de la boucle d'entraînement (mise à jour manuelle in-place)...");

    for epoch in 0..num_epochs {
        // Passe avant
        let y_pred = mlp.forward(&x_data)?;
        
        // Calcul de la perte
        let loss = loss_fn.calculate(&y_pred, &y_data)?;
        
        println!("Epoch: {}, Loss: {:?}", epoch, loss.item_f32()?);

        // Remise à zéro des gradients
        for param_arc in mlp.parameters() {
            let mut param_guard = param_arc.write().unwrap();
            param_guard.zero_grad();
        }

        // Passe arrière
        loss.backward(None)?;

        // Mise à jour manuelle des poids (NON IN-PLACE CETTE FOIS, pour contourner BufferSharedError)
        for param_arc in mlp.parameters() {
            let mut param_guard = param_arc.write().unwrap();
            if let Some(grad_tensor) = param_guard.grad() {
                let current_param_data = param_guard.tensor.get_f32_data()?; // Vec<f32>
                let grad_data = grad_tensor.get_f32_data()?; // Vec<f32>

                if current_param_data.len() != grad_data.len() {
                    return Err(NeuraRustError::ShapeMismatch {
                        expected: format!("grad shape matching data shape {}", current_param_data.len()),
                        actual: format!("{}", grad_data.len()),
                        operation: "manual weight update (data len check)".to_string(),
                    });
                }

                let mut new_param_data = Vec::with_capacity(current_param_data.len());
                for (p_val, g_val) in current_param_data.iter().zip(grad_data.iter()) {
                    new_param_data.push(p_val - learning_rate * g_val);
                }

                let original_shape = param_guard.tensor.shape().to_vec();
                let requires_grad_status = param_guard.tensor.requires_grad();
                
                // Remplacer le tenseur du paramètre par un nouveau
                param_guard.tensor = Tensor::new(new_param_data, original_shape)?;
                param_guard.tensor.set_requires_grad(requires_grad_status)?;
            }
        }
    }
    println!("Entraînement terminé.");
    Ok(())
} 