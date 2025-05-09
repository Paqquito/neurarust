//! # Exemple d'Entraînement d'un MLP Simple sur CPU
//!
//! Cet exemple illustre les étapes fondamentales pour entraîner un petit réseau de neurones
//! (Multi-Layer Perceptron) en utilisant `neurarust-core`.
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
//!     -   Mise à jour manuelle (inefficace) des poids du modèle.
//!     -   Remise à zéro des gradients pour la prochaine itération.
//!
//! ## Exécution
//! Pour exécuter cet exemple, utilisez la commande :
//! `cargo run --example basic_mlp_cpu`
//!
//! ## Prochaines Étapes (selon la Roadmap)
//! -   L'étape suivante (Sub-Phase 1.D) introduira des opérations en place (`add_`, `sub_`, etc.)
//!     pour une mise à jour des poids plus efficace, se rapprochant des pratiques de PyTorch.

use neurarust_core::tensor::Tensor;
use neurarust_core::nn::layers::linear::Linear;
use neurarust_core::nn::module::Module;
use neurarust_core::nn::parameter::Parameter;
use neurarust_core::ops::activation::relu_op; // Opération ReLU
use neurarust_core::NeuraRustError;
use neurarust_core::types::DType; // Ajout de l'import pour DType
use neurarust_core::tensor::create::randn; // Ajout de l'import pour randn
use neurarust_core::nn::losses::mse::MSELoss; // Ajout de l'import pour MSELoss
use std::sync::{Arc, RwLock};

/// Un Multi-Layer Perceptron (MLP) simple avec une couche cachée.
/// Architecture: Linear -> ReLU -> Linear
#[derive(Debug)] // Ajout de Debug pour permettre l'affichage si nécessaire
pub struct SimpleMLP {
    linear1: Linear,
    linear2: Linear,
}

impl SimpleMLP {
    /// Crée un nouveau SimpleMLP.
    ///
    /// # Arguments
    ///
    /// * `in_features`: Nombre de caractéristiques d'entrée.
    /// * `hidden_features`: Nombre de caractéristiques dans la couche cachée.
    /// * `out_features`: Nombre de caractéristiques de sortie.
    ///
    /// # Errors
    ///
    /// Retourne une erreur si la création des couches linéaires échoue.
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Result<Self, NeuraRustError> {
        let linear1 = Linear::new(in_features, hidden_features, true, DType::F32)?; // Avec biais et DType
        let linear2 = Linear::new(hidden_features, out_features, true, DType::F32)?; // Avec biais et DType
        Ok(SimpleMLP { linear1, linear2 })
    }
}

impl Module for SimpleMLP {
    /// Effectue une passe avant à travers le MLP.
    ///
    /// # Arguments
    ///
    /// * `input`: Le tenseur d'entrée.
    ///
    /// # Returns
    ///
    /// Le tenseur de sortie.
    ///
    /// # Errors
    ///
    /// Retourne une erreur si une opération échoue pendant la passe avant.
    fn forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError> {
        let x = self.linear1.forward(input)?;
        let x = relu_op(&x)?;
        self.linear2.forward(&x)
    }

    /// Retourne une liste des paramètres du module.
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

// Un petit main pour tester la structure pour l'instant
// Ce sera étendu dans les étapes suivantes de 1.C
fn main() -> Result<(), NeuraRustError> {
    // Test de création du MLP
    let mut mlp = SimpleMLP::new(10, 20, 5)?;
    println!("SimpleMLP créé avec succès !");

    // Test de la passe avant avec un tenseur factice
    let input_tensor = randn(vec![1, 10])?; // Utilisation de create::randn, shape en Vec<usize>
    println!("Tenseur d'entrée aléatoire créé : {:?}", input_tensor.shape());

    let output = mlp.forward(&input_tensor)?;
    println!("Sortie du MLP (shape): {:?}", output.shape());
    assert_eq!(output.shape(), &[1, 5]);

    // Test de la récupération des paramètres
    let params = mlp.parameters();
    println!("Nombre de paramètres dans le MLP: {}", params.len());
    // Chaque couche linéaire a un poids et un biais (si activé)
    // Donc, 2 paramètres pour linear1 + 2 pour linear2 = 4
    assert_eq!(params.len(), 4); 

    println!("Paramètres récupérés avec succès.");

    // Affichage des paramètres nommés pour vérification
    let named_params = mlp.named_parameters();
    println!("Paramètres nommés dans le MLP:");
    for (name, _param) in &named_params {
        println!("- {}", name);
    }
    assert_eq!(named_params.len(), 4); // Devrait aussi être 4

    // Test de children()
    let children = mlp.children();
    println!("Nombre d'enfants directs dans le MLP: {}", children.len());
    assert_eq!(children.len(), 2);

    // Test de named_children()
    let named_children = mlp.named_children();
    println!("Enfants nommés dans le MLP:");
    for (name, _module) in &named_children {
        println!("- {}", name);
    }
    assert_eq!(named_children.len(), 2);
    assert!(named_children.iter().any(|(name, _)| name == "linear1"));
    assert!(named_children.iter().any(|(name, _)| name == "linear2"));

    // Test de modules()
    let modules = mlp.modules();
    println!("Nombre total de modules (self + descendants) dans le MLP: {}", modules.len());
    // Devrait être 3: SimpleMLP lui-même, linear1, linear2
    assert_eq!(modules.len(), 3);

    // Step 1.C.2: Create Synthetic Data
    let batch_size = 4;
    let input_features = 10;
    let output_features = 5;

    let x_data = randn(vec![batch_size, input_features])?;
    let y_data = randn(vec![batch_size, output_features])?;

    println!("Données synthétiques X créées (shape): {:?}", x_data.shape());
    println!("Données synthétiques Y créées (shape): {:?}", y_data.shape());

    // Step 1.C.3: Instantiate Model and Loss
    // Le modèle (mlp) est déjà instancié plus haut.

    let loss_fn = MSELoss::new("mean"); // Utilise la réduction "mean", retrait du ?
    println!("Fonction de perte MSELoss instanciée.");

    // Step 1.C.4: Implement zero_grad Mechanism
    let first_weight_param_arc = mlp.linear1.weight(); // Ceci est &Arc<RwLock<Parameter>>
    {
        let first_weight_param_guard = first_weight_param_arc.read().unwrap();
        let shape = first_weight_param_guard.tensor.shape().to_vec();
        let numel = first_weight_param_guard.tensor.numel();
        let dummy_grad_data_vec = vec![0.5f32; numel];
        let dummy_grad_tensor_for_test = Tensor::new(dummy_grad_data_vec, shape)?;
        
        // acc_grad est sur Tensor, donc on accède à .tensor
        first_weight_param_guard.tensor.acc_grad(dummy_grad_tensor_for_test)?; 
        assert!(first_weight_param_guard.tensor.grad().is_some(), "Le gradient du premier poids devrait être Some avant zero_grad");
    }
    println!("Gradient factice assigné au premier paramètre via acc_grad.");

    // Appel de zero_grad sur tous les paramètres
    // zero_grad est sur Parameter, accessible via write lock
    mlp.linear1.weight_mut().write().unwrap().zero_grad();
    if let Some(bias_arc) = mlp.linear1.bias_mut() {
        bias_arc.write().unwrap().zero_grad();
    }
    mlp.linear2.weight_mut().write().unwrap().zero_grad();
    if let Some(bias_arc) = mlp.linear2.bias_mut() {
        bias_arc.write().unwrap().zero_grad();
    }
    println!("Méthode zero_grad appelée sur tous les paramètres.");

    // Vérification
    assert!(mlp.linear1.weight().read().unwrap().tensor.grad().is_none(), "Le gradient du premier poids devrait être None après zero_grad");
    if let Some(bias_param_arc) = mlp.linear1.bias() { 
        assert!(bias_param_arc.read().unwrap().tensor.grad().is_none(), "Le gradient du premier biais devrait être None après zero_grad");
    }
    assert!(mlp.linear2.weight().read().unwrap().tensor.grad().is_none(), "Le gradient du second poids devrait être None après zero_grad");
    if let Some(bias_param_arc) = mlp.linear2.bias() { 
        assert!(bias_param_arc.read().unwrap().tensor.grad().is_none(), "Le gradient du second biais devrait être None après zero_grad");
    }
    println!("Gradients vérifiés et remis à zéro avec succès.");

    // Step 1.C.5: Implement Manual Training Loop
    let learning_rate = 0.01f32;
    let num_epochs = 10; 
    println!("\nDébut de la boucle d'entraînement...");

    for epoch in 0..num_epochs {
        let y_pred = mlp.forward(&x_data)?;
        let loss = loss_fn.calculate(&y_pred, &y_data)?;
        
        if epoch % 2 == 0 || epoch == num_epochs - 1 {
            println!("Epoch: {}, Loss: {:?}", epoch, loss.get_f32_data()?);
        }

        // --- Remise à zéro des gradients avant la passe arrière ---
        // mlp.parameters() retourne Vec<Arc<RwLock<Parameter>>>
        for param_arc in mlp.parameters() {
            param_arc.write().unwrap().zero_grad();
        }

        // --- Passe Arrière ---
        loss.backward(None)?;

        // --- Mise à jour manuelle des poids ---
        // update_parameter_manually attend &mut Parameter
        update_parameter_manually(&mut mlp.linear1.weight_mut().write().unwrap(), learning_rate)?;
        if let Some(bias_arc) = mlp.linear1.bias_mut() {
            update_parameter_manually(&mut bias_arc.write().unwrap(), learning_rate)?;
        }
        update_parameter_manually(&mut mlp.linear2.weight_mut().write().unwrap(), learning_rate)?;
        if let Some(bias_arc) = mlp.linear2.bias_mut() {
            update_parameter_manually(&mut bias_arc.write().unwrap(), learning_rate)?;
        }
    }
    println!("Entraînement terminé.");

    Ok(())
}

// Fonction d'aide pour la mise à jour manuelle (simplifiée)
fn update_parameter_manually(
    param: &mut Parameter, 
    learning_rate: f32,
) -> Result<(), NeuraRustError> {
    let original_shape = param.tensor.shape(); 

    let new_data_opt = { 
        // Se ranger à l'avis du linter pour grad()
        let grad_tensor_opt = param.tensor.grad(); 

        if let Some(grad_tensor) = grad_tensor_opt {
            let current_tensor_data_guard = param.tensor.read_data();
            let current_data_slice = current_tensor_data_guard.buffer().try_get_cpu_f32()?;

            if grad_tensor.dtype() != param.tensor.dtype() {
                return Err(NeuraRustError::DataTypeMismatch {
                    expected: param.tensor.dtype(),
                    actual: grad_tensor.dtype(),
                    operation: "manual gradient update (grad dtype check)".to_string(),
                });
            }

            let grad_tensor_data_guard = grad_tensor.read_data();
            let grad_data_slice = grad_tensor_data_guard.buffer().try_get_cpu_f32()?;

            if current_data_slice.len() != grad_data_slice.len() {
                return Err(NeuraRustError::ShapeMismatch {
                    expected: format!("grad shape matching data shape {}", current_data_slice.len()),
                    actual: format!("{}", grad_data_slice.len()),
                    operation: "manual weight update".to_string(),
                });
            }
            
            let mut new_data_vec = Vec::with_capacity(current_data_slice.len());
            for (val, g_val) in current_data_slice.iter().zip(grad_data_slice.iter()) {
                new_data_vec.push(val - learning_rate * g_val);
            }
            Some(new_data_vec)
        } else {
            None 
        }
    }; 

    if let Some(new_data) = new_data_opt {
        param.tensor = Tensor::new(new_data, original_shape)?;
        param.tensor.set_requires_grad(true)?;
    }
    Ok(())
} 