/// Trait de base pour tous les optimiseurs.
/// Un optimiseur détient une référence aux paramètres du modèle
/// et implémente la logique pour mettre à jour ces paramètres
/// en fonction de leurs gradients.
pub trait Optimizer<T> {
    /// Effectue une seule étape d'optimisation (mise à jour des paramètres).
    fn step(&mut self);

    /// Remet à zéro les gradients de tous les paramètres gérés par l'optimiseur.
    fn zero_grad(&mut self);
} 