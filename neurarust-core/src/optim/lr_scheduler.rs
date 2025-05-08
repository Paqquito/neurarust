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
    /// Retourne les noms des paramètres.
    fn param_names(&self) -> Vec<String>;
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
/// Decays the learning rate of each parameter group by gamma every step_size epochs.
#[derive(Debug)]
pub struct StepLR<O: OptimizerInterface> {
    optimizer: O,
    step_size: usize,
    gamma: f32,
    last_epoch: usize, // To keep track of the number of steps or epochs, 0-indexed.
    _base_lrs: Vec<f32>, // To store initial learning rates
}

impl<O: OptimizerInterface> StepLR<O> {
    /// Creates a new `StepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Wrapped optimizer.
    /// * `step_size` - Period of learning rate decay.
    /// * `gamma` - Multiplicative factor of learning rate decay. Default: 0.1.
    ///
    /// # Panics
    ///
    /// Panics if `step_size` is 0.
    pub fn new(optimizer: O, step_size: usize, gamma: f32) -> Self {
        if step_size == 0 {
            panic!("StepLR: step_size must be greater than 0.");
        }
        // Store initial LRs. In PyTorch, LRScheduler class does this in its constructor.
        let base_lrs: Vec<f32> = optimizer.param_groups().iter().map(|pg| pg.lr()).collect();
        
        StepLR {
            optimizer,
            step_size,
            gamma,
            last_epoch: 0, // PyTorch's last_epoch is -1 at init, step() increments it first.
                           // Let's make last_epoch the count of steps taken, starting at 0 after the first step.
                           // So, when new() is called, no steps taken yet. First call to step() will make it 1.
                           // Or, simpler: last_epoch = 0 initially. After first step, last_epoch = 1.
                           // LR calculation: base_lr * gamma^(floor(last_epoch / step_size))
                           // If last_epoch is number of calls to step():
                           // Call 0: last_epoch = 0. exponent = 0. lr = base_lr.
                           // Call 1 (step_size=3): last_epoch=1. exp=0. lr=base_lr
                           // Call 2 (step_size=3): last_epoch=2. exp=0. lr=base_lr
                           // Call 3 (step_size=3): last_epoch=3. exp=1. lr=base_lr*gamma
                           // This seems correct if last_epoch is number of steps *taken*.
                           // PyTorch actually increments last_epoch *before* get_lr() in its step method.
                           // So if last_epoch = 0, after first step() call, it becomes 1, then get_lr() is called.
                           // For consistency with PyTorch's model, let's start last_epoch at 0 representing number of steps *completed*.
                           // And step() will effectively calculate for last_epoch+1 or simply update based on current last_epoch.
            _base_lrs: base_lrs,
        }
    }
}

impl<O: OptimizerInterface> LRScheduler<O> for StepLR<O> {
    fn step(&mut self, _epoch: Option<usize>, _metrics: Option<f32>) -> Result<(), NeuraRustError> {
        self.last_epoch += 1;

        // Le calcul du LR se base sur last_epoch et les _base_lrs.
        // Si last_epoch / step_size est différent de (last_epoch-1) / step_size, alors un "pas" a été franchi.
        // Cependant, il est plus simple de recalculer le LR à chaque fois basé sur la formule.
        // La condition if n'est nécessaire que si on veut optimiser pour ne pas appeler set_lr si le lr ne change pas.
        // Pour l'instant, recalculons et réappliquons toujours, c'est plus simple.

        // Vérifier si un pas de step_size a été franchi depuis la dernière mise à jour de LR.
        // Ex: step_size = 3. last_epoch = 1,2 -> no change. last_epoch = 3 -> change.
        // PyTorch le fait en calculant la valeur cible du LR et en la comparant à l'actuelle,
        // ou plus simplement, get_lr() calcule toujours la valeur correcte pour self.last_epoch.
        
        // Calculer le facteur d'échelle basé sur le nombre de pas complets de step_size
        let num_steps_taken = self.last_epoch / self.step_size;
        let scale_factor = self.gamma.powi(num_steps_taken as i32);

        let param_groups = self.optimizer.param_groups_mut();
        if self._base_lrs.len() != param_groups.len() {
            // Cela ne devrait pas arriver si l'optimizer n'est pas modifié après la création du scheduler
            return Err(NeuraRustError::ConfigurationError(
                "StepLR: Number of base_lrs and param_groups mismatch. Optimizer structure changed?".to_string()
            ));
        }

        for (pg, base_lr) in param_groups.iter_mut().zip(&self._base_lrs) {
            let new_lr = base_lr * scale_factor;
            pg.set_lr(new_lr);
        }
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.optimizer.param_groups().iter().map(|pg| pg.lr()).collect()
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
/// Decays the learning rate of each parameter group by gamma once the epoch reaches one of the milestones.
#[derive(Debug)]
pub struct MultiStepLR<O: OptimizerInterface> {
    optimizer: O,
    milestones: Vec<usize>, // Sorted list of epoch indices at which to decay LR.
    gamma: f32,
    last_epoch: usize,      // Number of times step() has been called.
    _base_lrs: Vec<f32>,
}

impl<O: OptimizerInterface> MultiStepLR<O> {
    /// Creates a new `MultiStepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Wrapped optimizer.
    /// * `milestones` - List of epoch indices. Must be increasing.
    /// * `gamma` - Multiplicative factor of learning rate decay. Default: 0.1.
    ///
    /// # Panics
    ///
    /// Panics if `milestones` is not sorted in strictly increasing order.
    pub fn new(optimizer: O, mut milestones: Vec<usize>, gamma: f32) -> Self {
        // Vérifier si les jalons sont triés. Si ce n'est pas le cas, paniquer.
        // Ou, on pourrait les trier et les dédoublonner, comme le fait PyTorch.
        // PyTorch: sorts and dedups milestones. Let's do that for robustness.
        milestones.sort_unstable();
        milestones.dedup();

        let base_lrs: Vec<f32> = optimizer.param_groups().iter().map(|pg| pg.lr()).collect();

        MultiStepLR {
            optimizer,
            milestones,
            gamma,
            last_epoch: 0,
            _base_lrs: base_lrs,
        }
    }
}

impl<O: OptimizerInterface> LRScheduler<O> for MultiStepLR<O> {
    fn step(&mut self, _epoch: Option<usize>, _metrics: Option<f32>) -> Result<(), NeuraRustError> {
        self.last_epoch += 1;

        // Compte combien de milestones ont été atteints ou dépassés par last_epoch.
        // self.milestones est trié.
        // Example: milestones = [2, 5, 8], last_epoch = 1 -> count = 0
        //          milestones = [2, 5, 8], last_epoch = 2 -> count = 1 (m=2)
        //          milestones = [2, 5, 8], last_epoch = 3 -> count = 1
        //          milestones = [2, 5, 8], last_epoch = 5 -> count = 2 (m=2, m=5)
        let num_milestones_passed = self.milestones.iter().filter(|&&m| m <= self.last_epoch).count();
        // Alternative plus efficace avec partition_point (Rust 1.52+)
        // let num_milestones_passed = self.milestones.partition_point(|&m| m < self.last_epoch);
        // Si un milestone est EXACTEMENT self.last_epoch, il doit être compté.
        // Donc, `m <= self.last_epoch` est correct. `partition_point` trouve l'index du premier élément > X (ou == X si X existe).
        // `self.milestones.partition_point(|&m| m <= self.last_epoch)` donnerait l'index *après* tous les milestones <= last_epoch, qui est le compte.
        // Let's use partition_point for efficiency if available, otherwise filter().count().
        // As of current Rust stable, partition_point is available.
        
        // Correction: PyTorch's MultiStepLR applies gamma when current epoch *is in* milestones. 
        // Not a cumulative power. It means last_lr * gamma if self.last_epoch is in milestones.
        // No, their get_lr is: `gamma ** bisect_right(self.milestones, self.last_epoch)`
        // `bisect_right` est équivalent à `partition_point` (compte les éléments <= X).
        // Donc, c'est bien une puissance du nombre de milestones passés.

        let scale_factor = self.gamma.powi(num_milestones_passed as i32);

        let param_groups = self.optimizer.param_groups_mut();
        if self._base_lrs.len() != param_groups.len() {
            return Err(NeuraRustError::ConfigurationError(
                "MultiStepLR: Number of base_lrs and param_groups mismatch. Optimizer structure changed?".to_string()
            ));
        }

        for (pg, base_lr) in param_groups.iter_mut().zip(&self._base_lrs) {
            let new_lr = base_lr * scale_factor;
            pg.set_lr(new_lr);
        }
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.optimizer.param_groups().iter().map(|pg| pg.lr()).collect()
    }

    fn optimizer(&self) -> &O {
        &self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

#[cfg(test)]
#[path = "lr_scheduler_tests.rs"]
mod tests; 