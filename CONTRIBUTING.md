# Guide de Contribution à NeuraRust 🦀🧠

Merci de votre intérêt pour NeuraRust ! Nous apprécions toutes les contributions, qu'il s'agisse de rapports de bugs, de demandes de fonctionnalités, d'améliorations de la documentation ou de code.

## Comment Contribuer

### 🐞 Signaler des Bugs

*   **Vérifiez les Issues Existantes :** Avant de soumettre un nouveau bug, veuillez rechercher dans les [issues GitHub](https://github.com/Paqquito/NeuraRust/issues) pour voir s'il n'a pas déjà été signalé.
*   **Créez une Issue Claire :** Si le bug n'est pas déjà signalé, créez une nouvelle issue.
    *   Utilisez un titre clair et descriptif.
    *   Décrivez précisément les étapes pour reproduire le bug.
    *   Indiquez le comportement attendu et le comportement observé.
    *   Incluez des informations sur votre environnement (version de Rust, système d'exploitation).
    *   Si possible, fournissez un exemple de code minimal reproductible.

### ✨ Proposer des Améliorations ou de Nouvelles Fonctionnalités

*   **Discutez d'abord :** Pour les changements majeurs ou les nouvelles fonctionnalités importantes, il est préférable d'ouvrir une issue [GitHub](https://github.com/Paqquito/NeuraRust/issues) pour discuter de l'idée avant de commencer à travailler dessus. Cela permet de s'assurer que la proposition s'aligne avec les objectifs du projet.
*   **Soyez Clair et Concis :** Décrivez clairement la fonctionnalité ou l'amélioration proposée et pourquoi elle serait bénéfique pour NeuraRust.

### 📝 Contribuer au Code ou à la Documentation (Pull Requests)

1.  **Forkez le Dépôt :** Créez un fork du dépôt [NeuraRust](https://github.com/Paqquito/NeuraRust) sur votre compte GitHub.
2.  **Clonez votre Fork :** Clonez votre fork localement : `git clone https://github.com/VOTRE_NOM_UTILISATEUR/NeuraRust.git`
3.  **Créez une Branche :** Créez une branche descriptive pour vos modifications : `git checkout -b ma-super-fonctionnalite`
4.  **Codez !** Faites vos modifications. 
    *   Suivez le style de code existant (utilisez `cargo fmt` pour formater).
    *   Ajoutez des tests unitaires pour les nouvelles fonctionnalités ou corrections de bugs.
    *   Assurez-vous que tous les tests passent (`cargo test`).
    *   Mettez à jour la documentation si nécessaire.
5.  **Committez vos Changements :** Utilisez des messages de commit clairs et descriptifs (par exemple, en suivant les [Conventional Commits](https://www.conventionalcommits.org/)).
    ```bash
    git add .
    git commit -m "feat: Ajout de la fonctionnalité X"
    ```
6.  **Poussez vers votre Fork :** `git push origin ma-super-fonctionnalite`
7.  **Ouvrez une Pull Request (PR) :** Allez sur la page du dépôt NeuraRust original et ouvrez une Pull Request de votre branche vers la branche `main` de NeuraRust.
    *   Donnez un titre clair à votre PR.
    *   Décrivez les changements effectués et liez l'issue correspondante si applicable (ex: `Closes #123`).
8.  **Revue de Code :** Les mainteneurs examineront votre PR. Il se peut qu'on vous demande d'apporter des modifications. Soyez réactif aux commentaires.
9.  **Fusion :** Une fois approuvée, votre PR sera fusionnée !

## Style de Code

Nous utilisons les outils standard de l'écosystème Rust :

*   **Formatage :** `cargo fmt`
*   **Linting :** `cargo clippy`

Veuillez exécuter ces commandes avant de soumettre une PR.

## Conduite

Nous attendons de tous les contributeurs qu'ils respectent notre [Code de Conduite](CODE_OF_CONDUCT.md) (à créer).

Merci encore pour votre contribution ! 