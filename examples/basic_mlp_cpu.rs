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
    /// Actuellement, collecte les poids et les biais des couches linéaires.
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params: Vec<&Parameter> = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
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

    // Pour tester zero_grad, nous avons besoin d'un moyen de définir un grad initial.
    // Dans un vrai scénario, le backward pass peuplerait les gradients.
    // Ici, nous utilisons acc_grad sur le tensor du premier paramètre pour définir un gradient factice.
    let first_weight_param = mlp.linear1.weight();
    let shape = first_weight_param.tensor().shape().to_vec();
    let numel = first_weight_param.tensor().numel();
    let dummy_grad_data_vec = vec![0.5f32; numel];
    let dummy_grad_tensor_for_test = Tensor::new(dummy_grad_data_vec, shape)?;
    
    first_weight_param.tensor().acc_grad(dummy_grad_tensor_for_test)?; // Utilise acc_grad pour initialiser

    assert!(mlp.linear1.weight().tensor().grad().is_some(), "Le gradient du premier poids devrait être Some avant zero_grad");
    println!("Gradient factice assigné au premier paramètre via acc_grad.");

    // Appel de zero_grad sur tous les paramètres
    mlp.linear1.weight_mut().zero_grad();
    if let Some(bias) = mlp.linear1.bias_mut() {
        bias.zero_grad();
    }
    mlp.linear2.weight_mut().zero_grad();
    if let Some(bias) = mlp.linear2.bias_mut() {
        bias.zero_grad();
    }

    println!("Méthode zero_grad appelée sur tous les paramètres.");

    // Vérification
    assert!(mlp.linear1.weight().tensor().grad().is_none(), "Le gradient du premier poids devrait être None après zero_grad");
    if let Some(bias_param) = mlp.linear1.bias() { // Utilise l'accesseur non mutable pour la vérification
        assert!(bias_param.tensor().grad().is_none(), "Le gradient du premier biais devrait être None après zero_grad");
    }
    assert!(mlp.linear2.weight().tensor().grad().is_none(), "Le gradient du second poids devrait être None après zero_grad");
    if let Some(bias_param) = mlp.linear2.bias() { // Utilise l'accesseur non mutable pour la vérification
        assert!(bias_param.tensor().grad().is_none(), "Le gradient du second biais devrait être None après zero_grad");
    }

    println!("Gradients vérifiés et remis à zéro avec succès.");

    // Step 1.C.5: Implement Manual Training Loop
    let learning_rate = 0.01f32;
    let num_epochs = 10; // Petit nombre pour l'exemple

    println!("\nDébut de la boucle d'entraînement...");

    for epoch in 0..num_epochs {
        // --- Passe Avant ---
        let y_pred = mlp.forward(&x_data)?;

        // --- Calcul de la Perte ---
        let loss = loss_fn.calculate(&y_pred, &y_data)?;
        
        // Affichage de la perte (optionnel, mais utile)
        // Assurez-vous que la perte est un scalaire ou peut être réduite à un scalaire
        // MSELoss avec réduction "Mean" ou "Sum" devrait produire un tenseur scalaire.
        match loss.item_f32() {
            Ok(loss_value) => {
                println!("Epoch: {}, Loss: {}", epoch, loss_value);
            }
            Err(e) => {
                eprintln!("Impossible d'extraire la valeur de la perte à l'epoch {}: {:?}. Shape de la perte: {:?}", epoch, e, loss.shape());
                // Peut-être que la perte n'est pas scalaire, ou autre problème.
                // Pour l'instant, on continue.
            }
        }

        // --- Passe Arrière ---
        loss.backward(None)?; // Utilise None pour un gradient initial de 1.0 (pour les pertes scalaires)

        // --- Mise à Jour des Poids (Manuelle et Inefficace) ---
        // Fonction helper temporaire pour la mise à jour manuelle d'un paramètre
        fn update_parameter_manually(
            param: &mut Parameter, 
            learning_rate: f32,
        ) -> Result<(), NeuraRustError> {
            let tensor = param.tensor_mut(); // Obtient &mut Tensor du Parameter

            if let Some(grad_tensor) = tensor.grad() { // grad() retourne Option<Tensor> (clone)
                // S'assurer que les opérations se font sur des données F32
                // Ceci est une simplification; une vraie implémentation gérerait les DTypes
                let weight_data = tensor.get_f32_data()?;
                let grad_data = grad_tensor.get_f32_data()?;

                if weight_data.len() != grad_data.len() {
                    return Err(NeuraRustError::ShapeMismatch {
                        expected: format!("data len {}", weight_data.len()),
                        actual: format!("grad data len {}", grad_data.len()),
                        operation: "manual weight update (data length)".to_string(),
                    });
                }

                let mut new_weight_data = Vec::with_capacity(weight_data.len());
                for (w, g) in weight_data.iter().zip(grad_data.iter()) {
                    new_weight_data.push(w - learning_rate * g);
                }

                let updated_tensor = Tensor::new(new_weight_data, tensor.shape().to_vec())?;
                let detached_updated_tensor = updated_tensor.detach(); // Détacher pour ne pas affecter le graphe
                
                *tensor = detached_updated_tensor; // Remplace le Tensor interne du Parameter
            }
            Ok(())
        }

        update_parameter_manually(mlp.linear1.weight_mut(), learning_rate)?;
        if let Some(bias_param) = mlp.linear1.bias_mut() {
            update_parameter_manually(bias_param, learning_rate)?;
        }
        update_parameter_manually(mlp.linear2.weight_mut(), learning_rate)?;
        if let Some(bias_param) = mlp.linear2.bias_mut() {
            update_parameter_manually(bias_param, learning_rate)?;
        }

        // --- Remise à Zéro des Gradients ---
        mlp.linear1.weight_mut().zero_grad();
        if let Some(bias) = mlp.linear1.bias_mut() {
            bias.zero_grad();
        }
        mlp.linear2.weight_mut().zero_grad();
        if let Some(bias) = mlp.linear2.bias_mut() {
            bias.zero_grad();
        }
    }

    println!("Boucle d'entraînement terminée.");

    Ok(())
} 