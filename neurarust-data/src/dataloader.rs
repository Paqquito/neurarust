// dataloader.rs
//! # DataLoader
//!
//! Le module `DataLoader` fournit une structure générique pour le chargement, le batching et le sampling efficace de données dans des projets de machine learning en Rust.
//!
//! ## Exemple d'utilisation basique
//!
//! ```rust
//! use neurarust_data::dataloader::DataLoader;
//! use neurarust_data::datasets::VecDataset;
//! use neurarust_data::samplers::sequential_sampler::SequentialSampler;
//!
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let dataset = VecDataset::new(data);
//! let sampler = SequentialSampler::new();
//! let mut loader = DataLoader::new(dataset, 2, sampler, false, None);
//! for batch in loader {
//!     let batch = batch.expect("Pas d'erreur attendue");
//!     println!("Batch : {:?}", batch);
//! }
//! ```
//!
//! ## Fonctionnalités principales
//!
//! - Batching automatique des données
//! - Sampling flexible via le trait `Sampler`
//! - Fonction de collation personnalisable
//! - Option pour ignorer le dernier batch incomplet (`drop_last`)
//!
//! ## Types supportés
//!
//! Le DataLoader est générique sur le type de dataset (`D: Dataset`) et le sampler (`S: Sampler`). Il peut donc fonctionner avec n'importe quel type de données ou stratégie de sampling compatible.

use crate::datasets::Dataset;
use crate::samplers::Sampler;
use neurarust_core::NeuraRustError;

/// Type pour la fonction de collation personnalisée.
///
/// Cette fonction prend un vecteur de samples (issus du dataset) et retourne un batch prêt à être utilisé par le modèle.
/// Par défaut, il s'agit d'une simple agrégation sous forme de `Vec`, mais on peut fournir une fonction pour, par exemple, empiler des tenseurs.
pub type CollateFn<D> = Box<dyn Fn(Vec<<D as Dataset>::Item>) -> Result<Vec<<D as Dataset>::Item>, NeuraRustError> + Send + Sync>;

/// DataLoader générique pour le batching et le sampling de données.
///
/// # Paramètres de type
/// - `D`: Le type du dataset, qui doit implémenter le trait [`Dataset`].
/// - `S`: Le type du sampler, qui doit implémenter le trait [`Sampler`].
///
/// # Champs principaux
/// - `dataset`: Le dataset source.
/// - `batch_size`: La taille des batches.
/// - `sampler`: Le sampler utilisé pour générer les indices.
/// - `drop_last`: Si vrai, le dernier batch est ignoré s'il est incomplet.
/// - `collate_fn`: Fonction de collation optionnelle pour assembler les samples en batch.
///
/// # Exemple
///
/// Voir l'exemple en haut de fichier pour une utilisation typique.
pub struct DataLoader<D: Dataset, S: Sampler> {
    /// Le dataset source.
    pub dataset: D,
    /// La taille des batches.
    pub batch_size: usize,
    /// Le sampler utilisé pour générer les indices.
    pub sampler: S,
    /// Si vrai, le dernier batch est ignoré s'il est incomplet.
    pub drop_last: bool,
    /// Fonction de collation optionnelle pour assembler les samples en batch.
    pub collate_fn: Option<CollateFn<D>>,
    indices_iter: Box<dyn Iterator<Item = usize> + Send + Sync>,
}

impl<D: Dataset, S: Sampler> DataLoader<D, S> {
    /// Crée un nouveau DataLoader.
    ///
    /// # Arguments
    /// - `dataset`: Le dataset à utiliser.
    /// - `batch_size`: La taille des batches.
    /// - `sampler`: Le sampler pour générer les indices.
    /// - `drop_last`: Si vrai, le dernier batch est ignoré s'il est incomplet.
    /// - `collate_fn`: Fonction de collation personnalisée (optionnelle).
    ///
    /// # Exemple
    ///
    /// ```rust
    /// # use neurarust_data::dataloader::DataLoader;
    /// # use neurarust_data::datasets::VecDataset;
    /// # use neurarust_data::samplers::sequential_sampler::SequentialSampler;
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let dataset = VecDataset::new(data);
    /// let sampler = SequentialSampler::new();
    /// let loader = DataLoader::new(dataset, 2, sampler, false, None);
    /// ```
    pub fn new(dataset: D, batch_size: usize, sampler: S, drop_last: bool, collate_fn: Option<CollateFn<D>>) -> Self {
        let indices_iter = sampler.iter(dataset.len());
        Self {
            dataset,
            batch_size,
            sampler,
            drop_last,
            collate_fn,
            indices_iter,
        }
    }

    /// Crée un DataLoader avec la fonction de collation par défaut (simple agrégation en `Vec`).
    ///
    /// # Exemple
    ///
    /// ```rust
    /// # use neurarust_data::dataloader::DataLoader;
    /// # use neurarust_data::datasets::VecDataset;
    /// # use neurarust_data::samplers::sequential_sampler::SequentialSampler;
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let dataset = VecDataset::new(data);
    /// let sampler = SequentialSampler::new();
    /// let loader = DataLoader::with_default_collate(dataset, 2, sampler, false);
    /// ```
    pub fn with_default_collate(dataset: D, batch_size: usize, sampler: S, drop_last: bool) -> Self 
    where
        <D as Dataset>::Item: Clone,
    {
        let collate_fn = Box::new(|batch: Vec<<D as Dataset>::Item>| Ok(batch));
        Self::new(dataset, batch_size, sampler, drop_last, Some(collate_fn))
    }
}

impl<D: Dataset, S: Sampler> Iterator for DataLoader<D, S> {
    type Item = Result<Vec<<D as Dataset>::Item>, NeuraRustError>;

    /// Renvoie le prochain batch de données.
    ///
    /// Cette méthode est appelée automatiquement lors de l'itération sur le DataLoader.
    /// Elle utilise le sampler pour générer les indices, récupère les items dans le dataset,
    /// et applique la fonction de collation si elle est définie.
    ///
    /// # Retour
    /// - `Some(Ok(batch))` : Un batch de données prêt à l'emploi.
    /// - `Some(Err(e))` : Une erreur lors de la récupération d'un item.
    /// - `None` : Plus de données à itérer.
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(idx) = self.indices_iter.next() {
                match self.dataset.get(idx) {
                    Ok(item) => batch.push(item),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                break;
            }
        }
        if batch.is_empty() || (self.drop_last && batch.len() < self.batch_size) {
            return None;
        }
        if let Some(ref collate_fn) = self.collate_fn {
            Some(collate_fn(batch))
        } else {
            Some(Ok(batch))
        }
    }
} 