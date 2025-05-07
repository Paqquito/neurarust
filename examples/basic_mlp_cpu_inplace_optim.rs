//! # Exemple d'Entraînement d'un MLP Simple sur CPU avec Optimiseur In-Place
//!
//! Cet exemple illustre les étapes fondamentales pour entraîner un petit réseau de neurones
//! (Multi-Layer Perceptron) en utilisant `neurarust-core`, avec une mise à jour
//! des poids optimisée grâce aux opérations in-place.
//!
//! ## Fonctionnalités Démontrées:
//! 1.  **Définition d'un Module de Réseau de Neurones** (`SimpleMLP`):
//!     -   Implémentation du trait `Module`.
//!     -   Utilisation de couches `Linear` et de fonctions d'activation (`relu_op`).
//!     -   Collecte des paramètres du modèle.
//! 2.  **Création de Données Synthétiques**: Tenseurs `X` (entrées) et `Y` (cibles).
//! 3.  **Instanciation du Modèle et de la Fonction de Perte** (`MSELoss`).
//! 4.  **Mécanisme `zero_grad`**: Remise à zéro des gradients des paramètres.
//! 5.  **Boucle d'Entraînement Manuelle**:
//!     -   Passe avant (`forward`).
//!     -   Calcul de la perte (`loss_fn.calculate`).
//!     -   Passe arrière (`backward`) pour calculer les gradients.
//!     -   **Mise à jour des poids du modèle utilisant des opérations in-place** (`sub_`, `mul_scalar`).
//!     -   Remise à zéro des gradients pour la prochaine itération.
//!
//! ## Exécution
//! Pour exécuter cet exemple, utilisez la commande :
//! `cargo run --example basic_mlp_cpu_inplace_optim`
//!

use neurarust_core::tensor::Tensor;
use neurarust_core::nn::layers::linear::Linear;
use neurarust_core::nn::module::Module;
use neurarust_core::nn::parameter::Parameter;
use neurarust_core::ops::activation::relu_op; // Opération ReLU
use neurarust_core::NeuraRustError;
use neurarust_core::types::DType; 
use neurarust_core::tensor::create::randn; 
use neurarust_core::nn::losses::mse::MSELoss; 

/// Un Multi-Layer Perceptron (MLP) simple avec une couche cachée.
/// Architecture: Linear -> ReLU -> Linear
#[derive(Debug)] 
pub struct SimpleMLP {
    linear1: Linear,
    linear2: Linear,
}

impl SimpleMLP {
    /// Crée un nouveau SimpleMLP.
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Result<Self, NeuraRustError> {
        let linear1 = Linear::new(in_features, hidden_features, true, DType::F32)?;
        let linear2 = Linear::new(hidden_features, out_features, true, DType::F32)?;
        Ok(SimpleMLP { linear1, linear2 })
    }
}

impl Module for SimpleMLP {
    /// Effectue une passe avant à travers le MLP.
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let x = self.linear1.forward(input)?;
        let x = relu_op(&x)?;
        self.linear2.forward(&x)
    }

    /// Retourne une liste des paramètres clonés du module.
    fn parameters(&self) -> Vec<&Parameter> {
        let mut model_params: Vec<&Parameter> = Vec::new();
        model_params.extend(self.linear1.parameters());
        model_params.extend(self.linear2.parameters());
        model_params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter)> {
        let mut params = Vec::new();
        for (name, param) in self.linear1.named_parameters() {
            params.push((format!("linear1.{}", name), param));
        }
        for (name, param) in self.linear2.named_parameters() {
            params.push((format!("linear2.{}", name), param));
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
        mods.push(&self.linear1 as &dyn Module);
        mods.push(&self.linear2 as &dyn Module);
        mods
    }
}


fn main() -> Result<(), NeuraRustError> {
    let mut mlp = SimpleMLP::new(10, 20, 5)?;
    println!("SimpleMLP (pour optim in-place) créé avec succès !");

    // Affichage des paramètres nommés pour vérification
    let named_params = mlp.named_parameters();
    println!("Paramètres nommés dans le MLP (in-place optim):");
    for (name, _param) in &named_params {
        println!("- {}", name);
    }
    assert_eq!(named_params.len(), 4); // Devrait aussi être 4

    // Test de children()
    let children = mlp.children();
    println!("Nombre d'enfants directs dans le MLP (in-place optim): {}", children.len());
    assert_eq!(children.len(), 2);

    // Test de named_children()
    let named_children = mlp.named_children();
    println!("Enfants nommés dans le MLP (in-place optim):");
    for (name, _module) in &named_children {
        println!("- {}", name);
    }
    assert_eq!(named_children.len(), 2);
    assert!(named_children.iter().any(|(name, _)| name == "linear1"));
    assert!(named_children.iter().any(|(name, _)| name == "linear2"));

    // Test de modules()
    let modules = mlp.modules();
    println!("Nombre total de modules (self + descendants) dans le MLP (in-place optim): {}", modules.len());
    assert_eq!(modules.len(), 3);

    let input_tensor = randn(vec![1, 10])?;
    let _output = mlp.forward(&input_tensor)?; // Vérifier que forward passe

    let batch_size = 4;
    let input_features = 10;
    let output_features = 5;

    let x_data = randn(vec![batch_size, input_features])?;
    let y_data = randn(vec![batch_size, output_features])?;

    let loss_fn = MSELoss::new("mean");

    // Remise à zéro initiale des gradients (bonne pratique avant la boucle)
    mlp.linear1.weight_mut().zero_grad();
    if let Some(b) = mlp.linear1.bias_mut() { b.zero_grad(); }
    mlp.linear2.weight_mut().zero_grad();
    if let Some(b) = mlp.linear2.bias_mut() { b.zero_grad(); }
    
    let learning_rate = 0.01f32; // Apprentissage pour F32
    let num_epochs = 10;

    println!("\nDébut de la boucle d'entraînement avec optimiseur in-place...");

    for epoch in 0..num_epochs {
        // --- Passe Avant ---
        let y_pred = mlp.forward(&x_data)?;

        // --- Calcul de la Perte ---
        let loss = loss_fn.calculate(&y_pred, &y_data)?;
        
        match loss.item_f32() {
            Ok(loss_value) => {
                println!("Epoch: {}, Loss: {}", epoch, loss_value);
            }
            Err(e) => {
                eprintln!("Impossible d'extraire la valeur de la perte à l'epoch {}: {:?}. Shape de la perte: {:?}", epoch, e, loss.shape());
            }
        }

        // --- Passe Arrière ---
        loss.backward(None)?;

        // --- Mise à Jour des Poids (Utilisant les opérations in-place) ---
        // Accès direct aux paramètres mutables et mise à jour

        // Paramètres de linear1
        if let Some(grad_tensor) = mlp.linear1.weight().tensor().grad() {
            let update_value = match grad_tensor.dtype() {
                DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor.contiguous()?, learning_rate)?,
                _ => return Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: grad_tensor.dtype(),
                    operation: "Optimizer step (linear1 weight mul_op_scalar)".to_string(),
                }),
            };
            let weight_param_tensor = mlp.linear1.weight_mut().tensor_mut();
            let detached_weight_data = weight_param_tensor.detach();
            let updated_weight_data_from_op = neurarust_core::ops::arithmetic::sub::sub_op(&detached_weight_data, &update_value)?;
            let final_updated_data = updated_weight_data_from_op.detach();
            final_updated_data.set_requires_grad(true)?;
            *weight_param_tensor = final_updated_data;
        }

        if let Some(bias_param_mut) = mlp.linear1.bias_mut() {
            if let Some(grad_tensor) = bias_param_mut.tensor().grad() {
                let update_value = match grad_tensor.dtype() {
                    DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor.contiguous()?, learning_rate)?,
                    _ => return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32,
                        actual: grad_tensor.dtype(),
                        operation: "Optimizer step (linear1 bias mul_op_scalar)".to_string(),
                    }),
                };
                let bias_tensor_mut = bias_param_mut.tensor_mut();
                let detached_bias_data = bias_tensor_mut.detach();
                let updated_bias_data_from_op = neurarust_core::ops::arithmetic::sub::sub_op(&detached_bias_data, &update_value)?;
                let final_updated_bias_data = updated_bias_data_from_op.detach();
                final_updated_bias_data.set_requires_grad(true)?;
                *bias_tensor_mut = final_updated_bias_data;
            }
        }

        // Paramètres de linear2
        if let Some(grad_tensor) = mlp.linear2.weight().tensor().grad() {
            let update_value = match grad_tensor.dtype() {
                DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor.contiguous()?, learning_rate)?,
                _ => return Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: grad_tensor.dtype(),
                    operation: "Optimizer step (linear2 weight mul_op_scalar)".to_string(),
                }),
            };
            let weight_param_tensor = mlp.linear2.weight_mut().tensor_mut();
            let detached_weight_data = weight_param_tensor.detach();
            let updated_weight_data_from_op = neurarust_core::ops::arithmetic::sub::sub_op(&detached_weight_data, &update_value)?;
            let final_updated_data = updated_weight_data_from_op.detach();
            final_updated_data.set_requires_grad(true)?;
            *weight_param_tensor = final_updated_data;
        }

        if let Some(bias_param_mut) = mlp.linear2.bias_mut() {
            if let Some(grad_tensor) = bias_param_mut.tensor().grad() {
                let update_value = match grad_tensor.dtype() {
                    DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor.contiguous()?, learning_rate)?,
                    _ => return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32,
                        actual: grad_tensor.dtype(),
                        operation: "Optimizer step (linear2 bias mul_op_scalar)".to_string(),
                    }),
                };
                let bias_tensor_mut = bias_param_mut.tensor_mut();
                let detached_bias_data = bias_tensor_mut.detach();
                let updated_bias_data_from_op = neurarust_core::ops::arithmetic::sub::sub_op(&detached_bias_data, &update_value)?;
                let final_updated_bias_data = updated_bias_data_from_op.detach();
                final_updated_bias_data.set_requires_grad(true)?;
                *bias_tensor_mut = final_updated_bias_data;
            }
        }

        // --- Remise à Zéro des Gradients ---
        mlp.linear1.weight_mut().zero_grad();
        if let Some(b) = mlp.linear1.bias_mut() { b.zero_grad(); }
        mlp.linear2.weight_mut().zero_grad();
        if let Some(b) = mlp.linear2.bias_mut() { b.zero_grad(); }
    }

    println!("\nEntraînement terminé.");
    Ok(())
}

// Notes additionnelles pour la structure de Parameter et Linear pour que cela fonctionne:
// 1. `Parameter` doit implémenter `zero_grad()` et exposer son `Tensor` de manière mutable.
//    (Ex: `tensor_mut()` ou via `DerefMut`). Actuellement, `Parameter(Tensor)` avec `DerefMut` existe.
//    `zero_grad` sur `Parameter` mettrait `tensor.grad = None`.
// 2. `