use crate::error::NeuraRustError; // En supposant que NeuraRustError est dans crate::error

// --- Placeholders --- 
// Ces traits sont des substituts temporaires. Ils devront être remplacés par les
// véritables définitions de l'Optimizer et de ParamGroup une fois que l'accès aux fichiers sera rétabli
// ou que leurs structures exactes seront fournies. L'Optimizer devra permettre d'accéder
// et de modifier les learning rates de ses groupes de paramètres.

/// Trait substitut pour un Optimizer.
pub trait OptimizerInterface {
    /// Type substitut pour un groupe de paramètres.
    type ParamGroup: ParamGroupInterface + Send + Sync;
    /// Retourne une tranche mutable des groupes de paramètres.
    fn param_groups_mut(&mut self) -> &mut [Self::ParamGroup];
    /// Retourne une tranche immuable des groupes de paramètres.
    fn param_groups(&self) -> &[Self::ParamGroup];
}

/// Trait substitut pour un ParamGroup.
pub trait ParamGroupInterface {
    /// Obtient le taux d'apprentissage actuel.
    fn lr(&self) -> f32;
    /// Définit le taux d'apprentissage.
    fn set_lr(&mut self, lr: f32);
}
// --- Fin des Placeholders ---

/// Définit l'interface pour les ordonnanceurs de taux d'apprentissage (Learning Rate Schedulers).
///
/// Les ordonnanceurs ajustent le taux d'apprentissage pendant l'entraînement
/// en fonction d'un calendrier ou d'une politique définie.
pub trait LRScheduler<O: OptimizerInterface> {
    /// Effectue une étape dans le calendrier du taux d'apprentissage.
    ///
    /// Cette méthode doit être appelée après chaque époque d'entraînement, ou itération,
    /// en fonction de la politique de l'ordonnanceur.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Un numéro d'époque actuel optionnel. Certains ordonnanceurs l'utilisent.
    /// * `metrics` - Une valeur métrique optionnelle (par exemple, la perte de validation). Utilisée par les
    ///               ordonnanceurs comme `ReduceLROnPlateau`.
    ///
    /// # Retours
    ///
    /// * `Result<(), NeuraRustError>` - Ok si l'étape a réussi, ou une erreur.
    fn step(&mut self, epoch: Option<usize>, metrics: Option<f32>) -> Result<(), NeuraRustError>;

    /// Retourne les derniers taux d'apprentissage calculés pour chaque groupe de paramètres.
    ///
    /// L'ordre des taux d'apprentissage dans le vecteur retourné doit correspondre
    /// à l'ordre des groupes de paramètres dans l'optimiseur.
    ///
    /// # Retours
    ///
    /// * `Vec<f32>` - Un vecteur des derniers taux d'apprentissage.
    fn get_last_lr(&self) -> Vec<f32>;

    /// Retourne une référence à l'optimiseur associé à cet ordonnanceur.
    fn optimizer(&self) -> &O;

    /// Retourne une référence mutable à l'optimiseur associé à cet ordonnanceur.
    fn optimizer_mut(&mut self) -> &mut O;
}

/// Mode for ReduceLROnPlateau scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceLROnPlateauMode {
    Min, // Reduce LR when the metric stops decreasing
    Max, // Reduce LR when the metric stops increasing
}

/// Threshold mode for ReduceLROnPlateau scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceLROnPlateauThresholdMode {
    Rel, // Relative threshold
    Abs, // Absolute threshold
}

/// Implements the ReduceLROnPlateau learning rate scheduler.
///
/// Reduces learning rate when a metric has stopped improving.
#[derive(Debug)]
pub struct ReduceLROnPlateau<O: OptimizerInterface> {
    optimizer: O,
    mode: ReduceLROnPlateauMode,
    factor: f32,
    patience: usize,
    threshold: f32,
    threshold_mode: ReduceLROnPlateauThresholdMode,
    cooldown: usize,
    min_lr: f32,
    eps: f32,

    // Internal state
    best: f32,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    last_epoch: usize,
    // _base_lrs: Vec<f32>, // Could be useful for min_lr per group
}

impl<O: OptimizerInterface> ReduceLROnPlateau<O> {
    /// Creates a new `ReduceLROnPlateau` scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Wrapped optimizer.
    /// * `mode` - One of `Min`, `Max`. In `Min` mode, lr will be reduced when the quantity monitored has stopped decreasing; in `Max` mode it will be reduced when the quantity monitored has stopped increasing.
    /// * `factor` - Factor by which the learning rate will be reduced. `new_lr = lr * factor`. Default: 0.1.
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced. Default: 10.
    /// * `threshold` - Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
    /// * `threshold_mode` - One of `Rel`, `Abs`. In `Rel` mode, dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or best * ( 1 - threshold ) in `min` mode. In `Abs` mode, dynamic_threshold = best + threshold in `max` mode or best - threshold in `min` mode. Default: `Rel`.
    /// * `cooldown` - Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
    /// * `min_lr` - A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
    /// * `eps` - Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    pub fn new(
        optimizer: O,
        mode: ReduceLROnPlateauMode,
        factor: Option<f32>,
        patience: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<ReduceLROnPlateauThresholdMode>,
        cooldown: Option<usize>,
        min_lr: Option<f32>,
        eps: Option<f32>,
    ) -> Self {
        let factor = factor.unwrap_or(0.1);
        if !(0.0..1.0).contains(&factor) {
            panic!("ReduceLROnPlateau: factor must be between 0.0 and 1.0.");
        }

        let initial_best = match mode {
            ReduceLROnPlateauMode::Min => f32::INFINITY,
            ReduceLROnPlateauMode::Max => f32::NEG_INFINITY,
        };

        ReduceLROnPlateau {
            optimizer,
            mode,
            factor,
            patience: patience.unwrap_or(10),
            threshold: threshold.unwrap_or(1e-4),
            threshold_mode: threshold_mode.unwrap_or(ReduceLROnPlateauThresholdMode::Rel),
            cooldown: cooldown.unwrap_or(0),
            min_lr: min_lr.unwrap_or(0.0),
            eps: eps.unwrap_or(1e-8),
            best: initial_best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_epoch: 0,
            // _base_lrs: optimizer.param_groups().iter().map(|pg| pg.lr()).collect(),
        }
    }

    // Helper function to check if the metric has improved
    fn is_better(&self, current_metric: f32, best_metric: f32) -> bool {
        if current_metric.is_nan() || current_metric.is_infinite() {
            return false; // NaN or Inf is never better
        }
        let threshold_val = match self.threshold_mode {
            ReduceLROnPlateauThresholdMode::Rel => best_metric * self.threshold,
            ReduceLROnPlateauThresholdMode::Abs => self.threshold,
        };

        match self.mode {
            ReduceLROnPlateauMode::Min => current_metric < best_metric - threshold_val.abs(),
            ReduceLROnPlateauMode::Max => current_metric > best_metric + threshold_val.abs(),
        }
    }

    // Helper function to reduce learning rates
    fn reduce_lr(&mut self, epoch: Option<usize>) {
        for pg in self.optimizer.param_groups_mut().iter_mut() {
            let old_lr = pg.lr();
            let mut new_lr = old_lr * self.factor;

            let current_min_lr = self.min_lr;
            
            if old_lr <= current_min_lr + self.eps {
                if old_lr - current_min_lr <= self.eps {
                    self.cooldown_counter = self.cooldown;
                    self.num_bad_epochs = 0;
                    continue;
                }
            }
            new_lr = new_lr.max(current_min_lr);
            

            if old_lr - new_lr > self.eps {
                pg.set_lr(new_lr);
                if let Some(ep) = epoch {
                    println!(
                        "Epoch {}: reducing learning rate of group to {:.4e}.",
                        ep,
                        new_lr
                    );
                } else {
                    println!(
                        "Reducing learning rate of group to {:.4e}.",
                        new_lr
                    );
                }
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }
    }
}

impl<O: OptimizerInterface> LRScheduler<O> for ReduceLROnPlateau<O> {
    fn step(&mut self, _epoch: Option<usize>, metrics: Option<f32>) -> Result<(), NeuraRustError> {
        let current_metric = match metrics {
            Some(m) => m,
            None => {
                return Err(NeuraRustError::ConfigurationError(
                    "ReduceLROnPlateau scheduler requires a metric value passed to step().".to_string(),
                ));
            }
        };

        self.last_epoch += 1;
        
        // Vérifier si c'est la première métrique valide (best est encore infini)
        let is_initial_best = match self.mode {
            ReduceLROnPlateauMode::Min => self.best.is_infinite() && self.best.is_sign_positive(),
            ReduceLROnPlateauMode::Max => self.best.is_infinite() && self.best.is_sign_negative(),
        };

        if is_initial_best {
             if current_metric.is_finite() {
                self.best = current_metric;
                self.num_bad_epochs = 0;
             } // else : la première métrique est NaN/Inf, on ne peut rien faire
        } else if self.is_better(current_metric, self.best) {
            self.best = current_metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0; // Reset bad epochs count during cooldown
        }

        if self.num_bad_epochs > self.patience {
            // Reduce LR if patience is exceeded and not in cooldown
            if self.cooldown_counter == 0 { 
                self.reduce_lr(Some(self.last_epoch));
                self.cooldown_counter = self.cooldown; // Start cooldown
                self.num_bad_epochs = 0; // Reset bad epochs count after reduction
            }
        }

        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.optimizer
            .param_groups()
            .iter()
            .map(|pg| pg.lr())
            .collect()
    }

    fn optimizer(&self) -> &O {
        &self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

/// Implements the StepLR learning rate scheduler.
///
/// Decays the learning rate of each parameter group by gamma every `step_size` epochs.
#[derive(Debug)]
pub struct StepLR<O: OptimizerInterface> {
    optimizer: O,
    step_size: usize,
    gamma: f32,
    last_epoch: usize, // To keep track of the number of steps or epochs
    _base_lrs: Vec<f32>, // To store initial learning rates for resetting or calculations (prefix with _)
}

impl<O: OptimizerInterface> StepLR<O> {
    /// Creates a new `StepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer.
    /// * `step_size` - Period of learning rate decay.
    /// * `gamma` - Multiplicative factor of learning rate decay.
    ///
    /// # Panics
    ///
    /// Panics if `step_size` is 0.
    pub fn new(optimizer: O, step_size: usize, gamma: f32) -> Self {
        if step_size == 0 {
            panic!("StepLR: step_size cannot be zero.");
        }
        let base_lrs = optimizer
            .param_groups()
            .iter()
            .map(|pg| pg.lr())
            .collect();

        StepLR {
            optimizer,
            step_size,
            gamma,
            last_epoch: 0,
            _base_lrs: base_lrs, // updated field name
        }
    }
}

impl<O: OptimizerInterface> LRScheduler<O> for StepLR<O> {
    fn step(&mut self, _epoch: Option<usize>, _metrics: Option<f32>) -> Result<(), NeuraRustError> {
        // Si l'époque est fournie, nous l'utilisons. Sinon, nous incrémentons notre compteur interne.
        // La roadmap mentionne "step(&mut self, epoch: Option<usize>, metrics: Option<f32>)"
        // PyTorch met à jour last_epoch basé sur les appels à step, pas sur la valeur de l'epoch passée.
        // Nous allons suivre une approche similaire: incrémenter last_epoch à chaque appel.
        // L'argument `epoch` peut être utilisé par d'autres schedulers, ou pour un reset.

        self.last_epoch += 1;

        if self.last_epoch % self.step_size == 0 {
            // Appliquer la décroissance du LR
            for (_i, pg) in self.optimizer.param_groups_mut().iter_mut().enumerate() {
                // On pourrait se baser sur pg.lr() * self.gamma ou sur base_lrs * gamma^n
                // PyTorch semble multiplier le lr courant.
                let new_lr = pg.lr() * self.gamma;
                pg.set_lr(new_lr);
            }
        }
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.optimizer
            .param_groups()
            .iter()
            .map(|pg| pg.lr())
            .collect()
    }

    fn optimizer(&self) -> &O {
        &self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

/// Implements the MultiStepLR learning rate scheduler.
///
/// Decays the learning rate of each parameter group by gamma once the number of epoch
/// reaches one of the milestones.
#[derive(Debug)]
pub struct MultiStepLR<O: OptimizerInterface> {
    optimizer: O,
    milestones: Vec<usize>,
    gamma: f32,
    last_epoch: usize,
    // _base_lrs: Vec<f32>, // Peut être utile si on veut un comportement différent ou pour reset.
}

impl<O: OptimizerInterface> MultiStepLR<O> {
    /// Creates a new `MultiStepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer.
    /// * `milestones` - A list of epoch indices. Must be increasing.
    /// * `gamma` - Multiplicative factor of learning rate decay.
    ///
    /// # Panics
    ///
    /// Panics if `milestones` is not sorted in increasing order.
    pub fn new(optimizer: O, mut milestones: Vec<usize>, gamma: f32) -> Self {
        // S'assurer que les milestones sont triés pour une vérification efficace.
        // On pourrait aussi retirer les doublons si nécessaire.
        let mut sorted_milestones = milestones.clone();
        sorted_milestones.sort_unstable();
        if milestones != sorted_milestones {
            // Ou on pourrait choisir de les trier automatiquement et d'émettre un warning.
            // Pour l'instant, exigeons qu'ils soient déjà triés pour être stricts.
            panic!("MultiStepLR: milestones must be sorted in increasing order.");
        }
        // Retirer les doublons pour éviter des décroissances multiples à la même époque si l'utilisateur les a fournis.
        milestones.dedup();

        // let base_lrs = optimizer.param_groups().iter().map(|pg| pg.lr()).collect();

        MultiStepLR {
            optimizer,
            milestones,
            gamma,
            last_epoch: 0,
            // _base_lrs: base_lrs,
        }
    }
}

impl<O: OptimizerInterface> LRScheduler<O> for MultiStepLR<O> {
    fn step(&mut self, _epoch: Option<usize>, _metrics: Option<f32>) -> Result<(), NeuraRustError> {
        self.last_epoch += 1;

        // Vérifier si l'epoch actuelle est un milestone
        // La méthode `binary_search` est efficace sur un Vec trié.
        if self.milestones.binary_search(&self.last_epoch).is_ok() {
            for pg in self.optimizer.param_groups_mut().iter_mut() {
                let new_lr = pg.lr() * self.gamma;
                pg.set_lr(new_lr);
            }
        }
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.optimizer
            .param_groups()
            .iter()
            .map(|pg| pg.lr())
            .collect()
    }

    fn optimizer(&self) -> &O {
        &self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuraRustError;

    // --- Mock Optimizer et ParamGroup pour les tests ---
    #[derive(Debug)]
    struct MockParamGroup {
        lr: f32,
    }

    impl ParamGroupInterface for MockParamGroup {
        fn lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    #[derive(Debug)]
    struct MockOptimizer {
        param_groups: Vec<MockParamGroup>,
    }

    impl OptimizerInterface for MockOptimizer {
        type ParamGroup = MockParamGroup;

        fn param_groups_mut(&mut self) -> &mut [Self::ParamGroup] {
            &mut self.param_groups
        }

        fn param_groups(&self) -> &[Self::ParamGroup] {
            &self.param_groups
        }
    }

    impl MockOptimizer {
        fn new(lrs: Vec<f32>) -> Self {
            MockOptimizer {
                param_groups: lrs.into_iter().map(|lr| MockParamGroup { lr }).collect(),
            }
        }
    }
    // --- Fin des Mocks ---

    #[test]
    fn test_step_lr_new() {
        let optimizer = MockOptimizer::new(vec![0.1, 0.01]);
        let scheduler = StepLR::new(optimizer, 3, 0.1);
        assert_eq!(scheduler.step_size, 3);
        assert_eq!(scheduler.gamma, 0.1);
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler._base_lrs, vec![0.1, 0.01]); // updated field name in test
    }

    #[test]
    #[should_panic(expected = "StepLR: step_size cannot be zero.")]
    fn test_step_lr_new_panic_on_zero_step_size() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        StepLR::new(optimizer, 0, 0.1); // Doit paniquer
    }

    #[test]
    fn test_step_lr_basic_decay() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = StepLR::new(optimizer, 2, 0.5);

        // Step 1: epoch 1, no decay
        scheduler.step(Some(1), None)?;
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);
        assert_eq!(scheduler.last_epoch, 1);

        // Step 2: epoch 2, decay should happen (0.1 * 0.5 = 0.05)
        scheduler.step(Some(2), None)?;
        assert_eq!(scheduler.get_last_lr(), vec![0.05]);
        assert_eq!(scheduler.last_epoch, 2);

        // Step 3: epoch 3, no decay
        scheduler.step(Some(3), None)?;
        assert_eq!(scheduler.get_last_lr(), vec![0.05]);
        assert_eq!(scheduler.last_epoch, 3);

        // Step 4: epoch 4, decay should happen (0.05 * 0.5 = 0.025)
        scheduler.step(Some(4), None)?;
        assert_eq!(scheduler.get_last_lr(), vec![0.025]);
        assert_eq!(scheduler.last_epoch, 4);

        Ok(())
    }

    #[test]
    fn test_step_lr_multiple_param_groups() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0, 0.5]);
        let mut scheduler = StepLR::new(optimizer, 1, 0.1);

        // Step 1: decay should happen (1.0*0.1=0.1, 0.5*0.1=0.05)
        scheduler.step(Some(1), None)?;
        let lrs = scheduler.get_last_lr();
        assert!((lrs[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs[1] - 0.05).abs() < f32::EPSILON);
        assert_eq!(scheduler.last_epoch, 1);

        // Step 2: decay again (0.1*0.1=0.01, 0.05*0.1=0.005)
        scheduler.step(Some(2), None)?;
        let lrs2 = scheduler.get_last_lr();
        assert!((lrs2[0] - 0.01).abs() < f32::EPSILON);
        assert!((lrs2[1] - 0.005).abs() < f32::EPSILON);
        assert_eq!(scheduler.last_epoch, 2);
        Ok(())
    }

    #[test]
    fn test_step_lr_get_optimizer_refs() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = StepLR::new(optimizer, 2, 0.5);

        // Accès immuable
        let _opt_ref = scheduler.optimizer();
        // Accès mutable
        scheduler.optimizer_mut().param_groups_mut()[0].set_lr(0.99);
        assert_eq!(scheduler.get_last_lr(), vec![0.99]);
    }
}

#[cfg(test)]
mod multistep_lr_tests { // Nouveau module de test pour éviter les conflits de nom de mock
    use super::*; // Accède à LRScheduler, MultiStepLR, etc.
    use crate::error::NeuraRustError;
    // Réutiliser les mocks définis dans le module de test de StepLR ou les redéfinir si nécessaire.
    // Pour la propreté, et si les mocks sont identiques, il serait mieux de les extraire
    // dans un module helper de test commun au sein de lr_scheduler.rs ou optim/mod.rs.
    // Pour l'instant, je vais les copier ici pour que ce bloc soit autonome.

    // --- Mock Optimizer et ParamGroup pour les tests (copié) ---
    #[derive(Debug, Clone)] // Ajout de Clone pour certains tests
    struct MockParamGroup {
        lr: f32,
    }

    impl ParamGroupInterface for MockParamGroup {
        fn lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    #[derive(Debug, Clone)] // Ajout de Clone
    struct MockOptimizer {
        param_groups: Vec<MockParamGroup>,
    }

    impl OptimizerInterface for MockOptimizer {
        type ParamGroup = MockParamGroup;

        fn param_groups_mut(&mut self) -> &mut [Self::ParamGroup] {
            &mut self.param_groups
        }

        fn param_groups(&self) -> &[Self::ParamGroup] {
            &self.param_groups
        }
    }

    impl MockOptimizer {
        fn new(lrs: Vec<f32>) -> Self {
            MockOptimizer {
                param_groups: lrs.into_iter().map(|lr| MockParamGroup { lr }).collect(),
            }
        }
    }
    // --- Fin des Mocks (copié) ---

    #[test]
    fn test_multistep_lr_new() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let milestones = vec![2, 5, 8];
        let scheduler = MultiStepLR::new(optimizer, milestones.clone(), 0.1);
        assert_eq!(scheduler.milestones, milestones);
        assert_eq!(scheduler.gamma, 0.1);
        assert_eq!(scheduler.last_epoch, 0);
    }

    #[test]
    #[should_panic(expected = "MultiStepLR: milestones must be sorted in increasing order.")]
    fn test_multistep_lr_new_panic_unsorted_milestones() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        MultiStepLR::new(optimizer, vec![5, 2, 8], 0.1); // Non trié
    }
    
    #[test]
    fn test_multistep_lr_new_deduplicates_milestones() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = MultiStepLR::new(optimizer, vec![2, 2, 3, 5, 5, 5, 8], 0.1);
        assert_eq!(scheduler.milestones, vec![2, 3, 5, 8]);
    }

    #[test]
    fn test_multistep_lr_basic_decay() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0]);
        let mut scheduler = MultiStepLR::new(optimizer, vec![2, 4], 0.1);

        // Epoch 1: last_epoch = 1. No decay.
        scheduler.step(None, None)?;
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);

        // Epoch 2: last_epoch = 2. Milestone! Decay: 1.0 * 0.1 = 0.1
        scheduler.step(None, None)?;
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < f32::EPSILON);

        // Epoch 3: last_epoch = 3. No decay.
        scheduler.step(None, None)?;
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < f32::EPSILON);

        // Epoch 4: last_epoch = 4. Milestone! Decay: 0.1 * 0.1 = 0.01
        scheduler.step(None, None)?;
        assert!((scheduler.get_last_lr()[0] - 0.01).abs() < f32::EPSILON);

        // Epoch 5: last_epoch = 5. No decay.
        scheduler.step(None, None)?;
        assert!((scheduler.get_last_lr()[0] - 0.01).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_multistep_lr_multiple_param_groups() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0, 0.5]);
        let mut scheduler = MultiStepLR::new(optimizer, vec![1, 3], 0.1);

        // Epoch 1: Milestone! Decay.
        // LR1: 1.0 * 0.1 = 0.1
        // LR2: 0.5 * 0.1 = 0.05
        scheduler.step(None, None)?;
        let lrs1 = scheduler.get_last_lr();
        assert!((lrs1[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs1[1] - 0.05).abs() < f32::EPSILON);

        // Epoch 2: No decay.
        scheduler.step(None, None)?;
        let lrs2 = scheduler.get_last_lr();
        assert!((lrs2[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs2[1] - 0.05).abs() < f32::EPSILON);

        // Epoch 3: Milestone! Decay.
        // LR1: 0.1 * 0.1 = 0.01
        // LR2: 0.05 * 0.1 = 0.005
        scheduler.step(None, None)?;
        let lrs3 = scheduler.get_last_lr();
        assert!((lrs3[0] - 0.01).abs() < f32::EPSILON);
        assert!((lrs3[1] - 0.005).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_multistep_lr_no_milestones() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0]);
        let mut scheduler = MultiStepLR::new(optimizer, vec![], 0.1); // No milestones

        scheduler.step(None, None)?;
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);
        scheduler.step(None, None)?;
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);
        Ok(())
    }

    #[test]
    fn test_multistep_lr_milestone_at_zero_or_one() -> Result<(), NeuraRustError> {
        // Test milestone at epoch 1 (last_epoch becomes 1)
        let optimizer1 = MockOptimizer::new(vec![1.0]);
        let mut scheduler1 = MultiStepLR::new(optimizer1, vec![1], 0.1);
        scheduler1.step(None, None)?;
        assert!((scheduler1.get_last_lr()[0] - 0.1).abs() < f32::EPSILON);

        // Test milestone at epoch 0 behavior (last_epoch starts at 0, increments to 1 on first step)
        // So a milestone at 0 effectively means decay on the first step too.
        // Our current logic means last_epoch is 1 on the first step, so milestone 0 won't be hit.
        // Let's test current behavior for milestone 0 (effectively ignored unless last_epoch starts at -1)
        let optimizer0 = MockOptimizer::new(vec![1.0]);
        let mut scheduler0 = MultiStepLR::new(optimizer0, vec![0, 1], 0.1); // includes 0
        scheduler0.step(None, None)?;
        assert!((scheduler0.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "Decay should happen at epoch 1");
        Ok(())
    }
}

#[cfg(test)]
mod reduce_lr_plateau_tests { // Nouveau module de test
    use super::*; // Accède à LRScheduler, ReduceLROnPlateau, Enums, etc.
    use crate::error::NeuraRustError;
    // Réutiliser ou redéfinir les mocks...

    // --- Mock Optimizer et ParamGroup pour les tests (copié) ---
    #[derive(Debug, Clone)]
    struct MockParamGroup {
        lr: f32,
    }

    impl ParamGroupInterface for MockParamGroup {
        fn lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    #[derive(Debug, Clone)]
    struct MockOptimizer {
        param_groups: Vec<MockParamGroup>,
    }

    impl OptimizerInterface for MockOptimizer {
        type ParamGroup = MockParamGroup;

        fn param_groups_mut(&mut self) -> &mut [Self::ParamGroup] {
            &mut self.param_groups
        }

        fn param_groups(&self) -> &[Self::ParamGroup] {
            &self.param_groups
        }
    }

    impl MockOptimizer {
        fn new(lrs: Vec<f32>) -> Self {
            MockOptimizer {
                param_groups: lrs.into_iter().map(|lr| MockParamGroup { lr }).collect(),
            }
        }
    }
    // --- Fin des Mocks (copié) ---

    #[test]
    fn test_reduce_lr_plateau_new_defaults() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = ReduceLROnPlateau::new(
            optimizer,
            ReduceLROnPlateauMode::Min,
            None, None, None, None, None, None, None
        );
        assert_eq!(scheduler.mode, ReduceLROnPlateauMode::Min);
        assert_eq!(scheduler.factor, 0.1);
        assert_eq!(scheduler.patience, 10);
        assert_eq!(scheduler.threshold, 1e-4);
        assert_eq!(scheduler.threshold_mode, ReduceLROnPlateauThresholdMode::Rel);
        assert_eq!(scheduler.cooldown, 0);
        assert_eq!(scheduler.min_lr, 0.0);
        assert_eq!(scheduler.eps, 1e-8);
        assert_eq!(scheduler.best, f32::INFINITY);
        assert_eq!(scheduler.num_bad_epochs, 0);
        assert_eq!(scheduler.cooldown_counter, 0);
        assert_eq!(scheduler.last_epoch, 0);
    }

    #[test]
    #[should_panic(expected = "ReduceLROnPlateau: factor must be between 0.0 and 1.0.")]
    fn test_reduce_lr_plateau_new_invalid_factor() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        ReduceLROnPlateau::new(
            optimizer, ReduceLROnPlateauMode::Min, Some(1.5), None, None, None, None, None, None
        );
    }

    #[test]
    fn test_reduce_lr_plateau_step_requires_metric() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = ReduceLROnPlateau::new(optimizer, ReduceLROnPlateauMode::Min, None, None, None, None, None, None, None);
        let result = scheduler.step(None, None);
        assert!(result.is_err());
        match result {
            Err(NeuraRustError::ConfigurationError(msg)) => {
                assert!(msg.contains("requires a metric value"));
            }
            _ => panic!("Expected ConfigurationError"),
        }
    }

    #[test]
    fn test_reduce_lr_plateau_min_lr_clamp() -> Result<(), NeuraRustError> {
        // Test spécifiquement le plafonnement à min_lr
        let optimizer = MockOptimizer::new(vec![0.0125]); // Start just above min_lr
        let mut scheduler = ReduceLROnPlateau::new(
            optimizer, // Passe par valeur, MockOptimizer implémente OptimizerInterface
            ReduceLROnPlateauMode::Min,
            Some(0.5),    // factor
            Some(0),      // patience (reduce immediately on bad epoch)
            Some(1e-4),   // threshold
            None,         // threshold_mode (Rel)
            Some(0),      // cooldown
            Some(0.01),   // min_lr
            Some(1e-9),   // eps (très petit)
        );

        // Simuler une première époque pour établir un "best"
        scheduler.step(None, Some(10.0))?;
        assert_eq!(scheduler.best, 10.0);
        assert_eq!(scheduler.get_last_lr(), vec![0.0125]);

        // Simuler une mauvaise époque, patience=0 -> réduction immédiate
        // Expected: new_lr = (0.0125 * 0.5).max(0.01) = 0.00625.max(0.01) = 0.01
        scheduler.step(None, Some(11.0))?;
        let current_lr = scheduler.get_last_lr()[0];
        assert!((current_lr - 0.01).abs() < 1e-7, "LR should clamp to 0.01");

        // Simuler une autre mauvaise époque, devrait rester à min_lr
        scheduler.step(None, Some(12.0))?;
        let current_lr_after_clamp = scheduler.get_last_lr()[0];
        assert!((current_lr_after_clamp - 0.01).abs() < 1e-7, "LR should stay clamped to 0.01");

        Ok(())
    }

    // Supprimer ou modifier l'ancien test `test_reduce_lr_plateau_min_mode_decay`
    // pour éviter la redondance ou la confusion.
    // Pour l'instant, je le laisse mais il faudra peut-être le fusionner/supprimer.

    // ... reste des tests ...
} 