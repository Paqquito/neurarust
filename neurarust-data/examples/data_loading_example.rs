//! Exemple d'utilisation du DataLoader avec TensorDataset, SequentialSampler et RandomSampler
//!
//! Ce script montre comment charger des données batchées pour l'entraînement d'un modèle.

// use neurarust_core::Tensor;
use neurarust_data::datasets::tensor_dataset::TensorDataset;
use neurarust_data::dataloader::DataLoader;
use neurarust_data::samplers::sequential_sampler::SequentialSampler;
use neurarust_data::samplers::random_sampler::RandomSampler;
use neurarust_core::tensor::create::rand;

fn main() {
    // Création de tenseurs synthétiques (features et labels)
    let features = rand(vec![10, 3]).expect("Création features ok"); // 10 exemples, 3 features
    let labels = rand(vec![10, 1]).expect("Création labels ok");   // 10 labels scalaires

    // Création du TensorDataset
    let dataset = TensorDataset::new(vec![features, labels]).expect("Tensors compatibles");

    // SequentialSampler : itération ordonnée
    let seq_sampler = SequentialSampler::new();
    let mut seq_loader = DataLoader::with_default_collate(dataset.clone(), 4, seq_sampler, false);
    println!("\n--- Batching avec SequentialSampler ---");
    for (i, batch) in seq_loader.by_ref().enumerate() {
        let batch = batch.expect("Pas d'erreur attendue");
        println!("Batch {i} :");
        for (j, sample) in batch.iter().enumerate() {
            println!("  Sample {j} :");
            for (k, tensor) in sample.iter().enumerate() {
                println!("    Tensor {k} shape: {:?}", tensor.shape());
            }
        }
    }

    // RandomSampler : itération aléatoire
    let rand_sampler = RandomSampler::new(false, None);
    let mut rand_loader = DataLoader::with_default_collate(dataset, 4, rand_sampler, false);
    println!("\n--- Batching avec RandomSampler ---");
    for (i, batch) in rand_loader.by_ref().enumerate() {
        let batch = batch.expect("Pas d'erreur attendue");
        println!("Batch {i} :");
        for (j, sample) in batch.iter().enumerate() {
            println!("  Sample {j} :");
            for (k, tensor) in sample.iter().enumerate() {
                println!("    Tensor {k} shape: {:?}", tensor.shape());
            }
        }
    }

    // (Optionnel) Intégration dans une boucle d'entraînement fictive
    // Ici, on montre juste comment on pourrait itérer sur les batchs pour entraîner un modèle
    // for batch in seq_loader {
    //     let batch = batch.expect("Pas d'erreur attendue");
    //     // batch : Vec<Vec<Tensor>>
    //     // Ici, on séparerait features et labels, puis on appellerait model.forward(), loss, backward, etc.
    // }
} 