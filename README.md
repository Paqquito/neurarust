# NeuraRust 🦀🧠

**Objectif** : Créer un framework de deep learning performant, sûr et ergonomique en Rust, inspiré par PyTorch mais exploitant les avantages uniques de Rust.

[![Rust](https://github.com/Paqquito/NeuraRust/actions/workflows/rust.yml/badge.svg)](https://github.com/Paqquito/NeuraRust/actions/workflows/rust.yml)

---

## ✨ Fonctionnalités Principales (basées sur Objectifs.md)

NeuraRust vise à fournir une expérience similaire à PyTorch tout en tirant parti de la puissance de Rust :

*   **Tenseurs (`neurarust-core::Tensor`)**: Une structure `Tensor` multi-dimensionnelle performante avec gestion explicite de la mémoire et sécurité garantie par Rust.
*   **Différentiation Automatique (`neurarust-core::Autograd`)**: Moteur d'autodifférentiation (dynamique) pour calculer les gradients automatiquement via `.backward()`.
*   **Modules de Réseaux Neuronaux (`neurarust-nn`)** *(Futur)*: Blocs de construction (couches linéaires, convolutives, etc.) et fonctions d'activation/perte.
*   **Optimiseurs (`neurarust-optim`)** *(Futur)*: Algorithmes d'optimisation standards (SGD, Adam...). 
*   **Gestion des Données (`neurarust-data`)** *(Futur)*: Outils pour charger et prétraiter les données (`Dataset`, `DataLoader`).
*   **Support Accélérateurs** *(Futur)*: Intégration GPU (CUDA, etc.) pour des calculs rapides.
*   **Interopérabilité & Déploiement** *(Futur)*: Export ONNX, bindings Python (PyO3), compilation WASM et binaire natif.

## 🎯 Avantages Clés de Rust

*   **Performance :** Vitesse native proche du C/C++, contrôle fin de la mémoire.
*   **Sécurité :** Garantie d'absence de data races et de nombreuses erreurs mémoire grâce au compilateur.
*   **Concurrence :** Parallélisme "sans crainte" pour l'accélération multi-cœurs (ex: data loading, autograd).
*   **Déploiement :** Compilation vers WASM, binaires natifs légers et autonomes.

## 🚧 État Actuel (Selon la Roadmap)

Le projet est actuellement dans les **premières phases (Phase 0 & 1)** :

*   ✅ **Phase 0 : Fondations et Tenseur de Base (CPU)**
    *   Structure du projet (workspace Cargo).
    *   Implémentation initiale de `Tensor` (données, shape).
    *   Opérations CPU fondamentales (arithmétique élément par élément).
    *   Tests unitaires pour `Tensor` et opérations de base.
*   ⏳ **Phase 1 : Autograd et Blocs de Construction NN**
    *   Bases du moteur Autograd (structure `BackwardOp`, graphe de calcul via `Rc<RefCell>`, `.backward()` initiée).
    *   *Prochaines étapes : Compléter la passe backward, définir les modules `nn`.* 

Voir [`Objectifs.md`](Objectifs.md) pour la roadmap complète.

## 🚀 Commencer

1.  **Prérequis :** Assurez-vous d'avoir [Rust installé](https://www.rust-lang.org/tools/install).
2.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/Paqquito/NeuraRust.git # Mettre l'URL correcte
    cd NeuraRust
    ```
3.  **Compiler :**
    ```bash
    cargo build
    ```
4.  **Exécuter les tests :**
    ```bash
    cargo test
    ```

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez consulter [`CONTRIBUTING.md`](CONTRIBUTING.md) pour les directives.

## 📜 Licence

Ce projet est sous licence [MIT](LICENSE). 