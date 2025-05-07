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
        todo!("Implement named_parameters for SimpleMLP (inplace optim example)")
    }

    fn modules(&self) -> Vec<&dyn Module> {
        todo!("Implement modules for SimpleMLP (inplace optim example)")
    }
}


fn main() -> Result<(), NeuraRustError> {
    let mut mlp = SimpleMLP::new(10, 20, 5)?;
    println!("SimpleMLP (pour optim in-place) créé avec succès !");

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
                DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor, learning_rate)?,
                _ => return Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: grad_tensor.dtype(),
                    operation: "Optimizer step (linear1 weight mul_op_scalar)".to_string(),
                }),
            };
            mlp.linear1.weight_mut().sub_(&update_value)?;
        }

        if let Some(bias_param_mut) = mlp.linear1.bias_mut() {
            if let Some(grad_tensor) = bias_param_mut.tensor().grad() {
                let update_value = match grad_tensor.dtype() {
                    DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor, learning_rate)?,
                    _ => return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32,
                        actual: grad_tensor.dtype(),
                        operation: "Optimizer step (linear1 bias mul_op_scalar)".to_string(),
                    }),
                };
                bias_param_mut.sub_(&update_value)?;
            }
        }

        // Paramètres de linear2
        if let Some(grad_tensor) = mlp.linear2.weight().tensor().grad() {
            let update_value = match grad_tensor.dtype() {
                DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor, learning_rate)?,
                _ => return Err(NeuraRustError::DataTypeMismatch {
                    expected: DType::F32,
                    actual: grad_tensor.dtype(),
                    operation: "Optimizer step (linear2 weight mul_op_scalar)".to_string(),
                }),
            };
            mlp.linear2.weight_mut().sub_(&update_value)?;
        }

        if let Some(bias_param_mut) = mlp.linear2.bias_mut() {
            if let Some(grad_tensor) = bias_param_mut.tensor().grad() {
                let update_value = match grad_tensor.dtype() {
                    DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor, learning_rate)?,
                    _ => return Err(NeuraRustError::DataTypeMismatch {
                        expected: DType::F32,
                        actual: grad_tensor.dtype(),
                        operation: "Optimizer step (linear2 bias mul_op_scalar)".to_string(),
                    }),
                };
                bias_param_mut.sub_(&update_value)?;
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
// 2. `Linear` doit fournir une méthode comme `get_mutable_parameters(&mut self) -> Vec<&mut Parameter>`.
//    Ou alors `SimpleMLP::parameters()` devrait retourner `Vec<Arc<RwLock<Parameter>>>`
//    et la boucle d'optimisation devrait utiliser `param_lock.write().unwrap()` pour obtenir
//    une garde mutable sur `Parameter`.
//    L'exemple `basic_mlp_cpu.rs` faisait: `mlp.linear1.weight_mut()` qui retourne `&mut Parameter`.
//    Donc, `Linear` doit avoir `weight_mut()` et `bias_mut()`.
//    `SimpleMLP::get_mutable_parameters` est une façon de collecter ces `&mut Parameter`.

// Pour la méthode `mul_scalar_op`:
// Si `Tensor` n'a pas de `mul_scalar_op` non in-place, il faudrait l'ajouter:
// impl Tensor {
//     pub fn mul_scalar_op(&self, scalar: f32) -> Result<Tensor, NeuraRustError> {
//         if self.dtype() != DType::F32 { /* error */ }
//         // Logique pour créer un nouveau tenseur résultat de self * scalar
//         // Pourrait utiliser `ops::arithmetic::mul::mul_op_scalar(self, scalar)`
//         ops::arithmetic::mul::mul_op_scalar(self, scalar) // Si mul_op_scalar accepte f32 directement
                                                          // ou si une conversion est faite.
                                                          // ops::arithmetic::mul::mul_op_scalar<f32>(self, scalar)
//     }
// }
// Le `ops::arithmetic::mul::mul_op_scalar` semble être la bonne voie.
// Il est défini comme `pub fn mul_op_scalar<T: NeuraNumeric>(tensor: &Tensor, scalar: T) -> Result<Tensor, NeuraRustError>`.
// Donc, `ops::arithmetic::mul::mul_op_scalar(&grad_tensor_owned, learning_rate)` devrait fonctionner.

// Révision de la boucle d'update avec `ops::arithmetic::mul::mul_op_scalar`:
// ... dans la boucle for ...
// if let Some(grad_tensor_owned) = param_ref_mut.tensor().grad() {
//     let update_value = match grad_tensor_owned.dtype() {
//         DType::F32 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor_owned, learning_rate)?,
//         DType::F64 => neurarust_core::ops::arithmetic::mul::mul_op_scalar(&grad_tensor_owned, learning_rate as f64)?, // Cast lr si besoin
//         _ => { /* error */ }
//     };
//     param_ref_mut.sub_(&update_value)?;
// }
// ...
// Cette approche est plus propre car elle utilise directement une fonction d'op existante.
// Il faudra s'assurer que `learning_rate` a le bon type (f32 ou f64) pour correspondre au paramètre.
// L'exemple étant F32, `learning_rate` est déjà f32.
