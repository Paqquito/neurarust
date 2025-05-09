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

use neurarust_core::{NeuraRustError, nn::layers::linear::Linear, nn::layers::ReLU, model::sequential::Sequential, nn::module::Module, nn::parameter::Parameter, types::DType, tensor::create::randn, nn::losses::mse::MSELoss, optim::Optimizer, optim::AdamOptimizer, optim::lr_scheduler::{StepLR, LRScheduler}, tensor::Tensor};
use std::sync::{Arc, RwLock};

/// Un Multi-Layer Perceptron (MLP) simple avec une couche cachée.
/// Architecture: Linear -> ReLU -> Linear
#[derive(Debug)]
struct SimpleMLP {
    layers: Sequential,
}

impl SimpleMLP {
    /// Crée un nouveau SimpleMLP.
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self, NeuraRustError> {
        let mut layers = Sequential::new();
        layers.add_module(
            "linear1",
            Box::new(Linear::new(input_size, hidden_size, true, DType::F32)?)
        );
        layers.add_module("relu1", Box::new(ReLU::new()));
        layers.add_module(
            "linear2",
            Box::new(Linear::new(hidden_size, output_size, true, DType::F32)?)
        );
        Ok(SimpleMLP { layers })
    }
}

impl Module for SimpleMLP {
    /// Effectue une passe avant à travers le MLP.
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        self.layers.forward(input)
    }

    /// Retourne une liste des paramètres clonés du module.
    /// Modifié pour retourner le type attendu par les optimiseurs.
    fn parameters(&self) -> Vec<Arc<RwLock<Parameter>>> {
        self.layers.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Arc<RwLock<Parameter>>)> {
        self.layers.named_parameters()
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.layers.children()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.layers.named_children()
    }

    fn modules(&self) -> Vec<&dyn Module> {
        let mut mods = vec![self as &dyn Module];
        mods.extend(self.layers.modules());
        mods
    }

    fn apply(&mut self, f: &mut dyn FnMut(&mut dyn Module)) {
        self.layers.apply(f);
    }
}

fn main() -> Result<(), NeuraRustError> {
    // La variable device n'est pas utilisée dans cet exemple pour le moment.
    // let device = StorageDevice::CPU;
    let mlp = SimpleMLP::new(10, 20, 5)?;
    println!("SimpleMLP créé avec succès !");

    let mut optimizer = AdamOptimizer::new(
        mlp.parameters(),
        0.001, // lr
        0.9,   // beta1
        0.999, // beta2
        1e-8, // eps
        0.01, // weight_decay
        false, // amsgrad
    )?;

    let mut scheduler = StepLR::new(&mut optimizer, 5, 0.5); 

    println!("\nDébut de la boucle d'entraînement...");
    let loss_fn = MSELoss::new("mean");
    let num_epochs = 10;
    for epoch in 0..num_epochs {
        // Simuler un batch de données
        let x_data = randn(vec![4, 10])?; 
        let y_data = randn(vec![4, 5])?;  

        // Passe avant
        let y_pred = mlp.forward(&x_data)?;

        // Calcul de la perte
        let loss = loss_fn.calculate(&y_pred, &y_data)?;

        // Passe arrière
        loss.backward(None)?;

        // Accéder à l'optimiseur via le scheduler
        scheduler.optimizer_mut().zero_grad();
        scheduler.optimizer_mut().step()?;

        // Mettre à jour le learning rate
        scheduler.step(Some(epoch as u64), None)?;

        // Afficher la perte et le LR à chaque epoch
        let current_lr = scheduler.get_last_lr(); 
        match loss.item_f32() { // Utiliser item_f32 pour obtenir Result<f32>
            Ok(loss_value) => {
                println!(
                    "Epoch [{}/{}], Loss: {:.4}, LR: {:?}", // Utiliser {:.4} ou :? pour f32
                    epoch + 1,
                    num_epochs,
                    loss_value, // Pas besoin de .unwrap() ici, le format :? fonctionne
                    current_lr
                );
            },
            Err(e) => {
                eprintln!("Impossible d'extraire la valeur de la perte à l'epoch {}: {:?}. Shape de la perte: {:?}", epoch, e, loss.shape());
            }
        }
    }

    println!("\nEntraînement terminé.");
    Ok(())
}

// Notes additionnelles pour la structure de Parameter et Linear pour que cela fonctionne:
// 1. `Parameter` doit implémenter `zero_grad()` et exposer son `Tensor` de manière mutable.
//    (Ex: `tensor_mut()` ou via `DerefMut`). Actuellement, `Parameter(Tensor)` avec `DerefMut` existe.
//    `zero_grad` sur `Parameter` mettrait `tensor.grad = None`.
// 2. `