// neurarust-core/src/optim/sgd.rs

use crate::tensor::Tensor;
use crate::optim::optimizer::Optimizer; // Importer le trait
use std::ops::{Mul, Sub, AddAssign}; // Supprimé: SubAssign
use num_traits::FromPrimitive; // Pour convertir lr en T
use std::fmt::Debug;

/// Implémente l'algorithme de descente de gradient stochastique.
/// 
/// Met à jour les paramètres `p` selon la règle:
/// `p = p - lr * grad(p)`
#[derive(Debug)]
pub struct SGD<T: Clone> { // T doit être Clone pour stocker lr
    params: Vec<Tensor<T>>,
    lr: T, // Taux d'apprentissage
}

impl<T> SGD<T>
where
    T: Copy + Clone + FromPrimitive + Debug, // FromPrimitive pour convertir le lr f32/f64 initial
{
    /// Crée une nouvelle instance de l'optimiseur SGD.
    ///
    /// # Arguments
    ///
    /// * `params` - Une collection itérable de Tensors qui doivent être optimisés.
    /// * `lr` - Le taux d'apprentissage.
    pub fn new(params: impl IntoIterator<Item = Tensor<T>>, lr: f64) -> Self {
        let lr_t = T::from_f64(lr).expect("Impossible de convertir le taux d'apprentissage (lr) vers le type T.");
        SGD {
            params: params.into_iter().collect(),
            lr: lr_t,
        }
    }
}

impl<T> Optimizer<T> for SGD<T> 
where
    // Bounds pour step: Sub, Mul (pour T*T), Copy, Clone, Debug, AddAssign (si on utilise += dans SubAssign)
    // Bounds pour new: FromPrimitive
    // Bounds pour zero_grad: (aucune spécifique ici)
    T: Copy + Clone + Debug + FromPrimitive + 
       Sub<Output=T> + Mul<Output=T> + AddAssign, // Minimum pour calcul manuel de mise à jour
{
    fn step(&mut self) {
        for param in &self.params {
            // Crée un scope pour limiter la durée de vie de l'emprunt de `borrow_grad`
            let grad_data_option = {
                let grad_option = param.borrow_grad(); // Emprunt immuable
                if let Some(grad_tensor) = grad_option.as_ref() {
                    // Clone les données du gradient pendant que l'emprunt est valide
                    Some(grad_tensor.data().to_vec()) // Clone les données
                } else {
                    None
                }
                // `grad_option` (et l'emprunt immuable) est libéré ici
            };

            // Si un gradient existait (et nous avons cloné ses données)
            if let Some(grad_data) = grad_data_option {
                // Calcule la mise à jour en utilisant les données clonées
                let update_data: Vec<T> = grad_data.iter().map(|&g| g * self.lr).collect();

                // Maintenant, l'emprunt mutable est sûr
                let mut param_td_mut = param.borrow_tensor_data_mut(); // Emprunt mutable

                assert_eq!(param_td_mut.data.len(), update_data.len(), "Shape mismatch during SGD step");

                // Applique la mise à jour
                param_td_mut.data.iter_mut().zip(update_data.iter()).for_each(|(p, &u)| *p = *p - u);
                // L'emprunt mutable est libéré ici
            }
            // Si grad_data_option est None, ne rien faire
        }
    }

    fn zero_grad(&mut self) {
        for param in self.params.iter() {
            // Prend un emprunt mutable pour modifier le champ grad
            let mut param_td_mut = param.borrow_tensor_data_mut(); 
            param_td_mut.grad = None;
            // L'emprunt mutable est relâché ici
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import SGD, Optimizer
    use crate::tensor::Tensor;

    // Helper pour créer des tenseurs f32
    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape)
    }
    
    // Helper pour vérifier l'égalité approximative des données
    fn check_data_approx(tensor: &Tensor<f32>, expected_data: &[f32]) {
        assert_eq!(tensor.data().len(), expected_data.len());
        for (a, b) in tensor.data().iter().zip(expected_data.iter()) {
            assert!((a - b).abs() < 1e-6, "Data mismatch: expected {:?}, got {:?}", expected_data, tensor.data());
        }
    }

    #[test]
    fn test_sgd_zero_grad() {
        let p1 = create_tensor_f32(vec![1., 2.], vec![2]);
        let p2 = create_tensor_f32(vec![3., 4.], vec![2]);
        // Donner un gradient initial à p1
        p1.borrow_tensor_data_mut().grad = Some(create_tensor_f32(vec![0.1, 0.2], vec![2]));
        
        let params = vec![p1.clone(), p2.clone()];
        let mut optim = SGD::new(params, 0.1);

        assert!(p1.borrow_grad().is_some());
        assert!(p2.borrow_grad().is_none());

        optim.zero_grad();

        assert!(p1.borrow_grad().is_none(), "Grad de p1 devrait être None après zero_grad");
        assert!(p2.borrow_grad().is_none(), "Grad de p2 devrait être None après zero_grad");
    }

    #[test]
    fn test_sgd_step() {
        let p1 = create_tensor_f32(vec![1.0, 2.0], vec![2]);
        let p2 = create_tensor_f32(vec![3.0, 4.0], vec![1, 2]); // Shape différente
        let p3 = create_tensor_f32(vec![5.0], vec![1]); // Sans gradient

        // Donner des gradients
        let grad1 = create_tensor_f32(vec![10.0, -20.0], vec![2]);
        let grad2 = create_tensor_f32(vec![0.5, -0.5], vec![1, 2]);
        p1.borrow_tensor_data_mut().grad = Some(grad1);
        p2.borrow_tensor_data_mut().grad = Some(grad2);

        let params = vec![p1.clone(), p2.clone(), p3.clone()];
        let mut optim = SGD::new(params, 0.1); // lr = 0.1

        optim.step();

        // Vérifier p1: p1 = p1 - lr * grad1 = [1, 2] - 0.1 * [10, -20] = [1, 2] - [1, -2] = [0, 4]
        check_data_approx(&p1, &[0.0, 4.0]);

        // Vérifier p2: p2 = p2 - lr * grad2 = [3, 4] - 0.1 * [0.5, -0.5] = [3, 4] - [0.05, -0.05] = [2.95, 4.05]
        check_data_approx(&p2, &[2.95, 4.05]);

        // Vérifier p3: pas de gradient, ne devrait pas changer
        check_data_approx(&p3, &[5.0]);
    }
} 