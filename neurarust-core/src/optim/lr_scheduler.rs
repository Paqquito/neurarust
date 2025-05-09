use crate::error::NeuraRustError;
use crate::optim::Optimizer;
// use crate::optim::param_group::ParamGroup; // Tentative de suppression
use std::fmt::Debug;

/// Base trait for all learning rate schedulers.
pub trait LRScheduler<'a, O: Optimizer + ?Sized> {
    /// Returns the last computed learning rate values for each parameter group.
    fn get_last_lr(&self) -> Vec<f32>;

    /// Performs a scheduler step.
    ///
    /// # Arguments
    ///
    /// * `epoch`: The current epoch number (optional, used by some schedulers).
    /// * `metrics`: A metric value (optional, used by schedulers like ReduceLROnPlateau).
    ///
    /// # Returns
    ///
    /// `Ok(())` if the step was successful, or `NeuraRustError` on failure.
    fn step(&mut self, epoch: Option<u64>, metrics: Option<f32>) -> Result<(), NeuraRustError>;

    // Nouvelles méthodes
    fn optimizer(&self) -> &O;
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
pub struct ReduceLROnPlateau<'a, O: Optimizer + ?Sized> {
    optimizer: &'a mut O,
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
}

impl<'a, O: Optimizer + ?Sized> ReduceLROnPlateau<'a, O> {
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
        optimizer: &'a mut O,
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
    fn reduce_lr(&mut self, _epoch: Option<usize>) {
        for (i, pg) in self.optimizer.param_groups_mut().iter_mut().enumerate() {
            let old_lr = pg.options.lr.expect(&format!("Learning rate not set for param group {}", i));
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
                if let Some(ep) = _epoch {
                    println!(
                        "Epoch {}: reducing learning rate of group {} to {:.4e}.",
                        ep,
                        i,
                        new_lr
                    );
                } else {
                    println!(
                        "Reducing learning rate of group {} to {:.4e}.",
                        i,
                        new_lr
                    );
                }
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }
    }

    fn get_current_lrs(&self) -> Vec<f32> {
        self.optimizer
            .param_groups()
            .iter()
            .map(|pg| pg.options.lr.unwrap_or_else(|| {
                eprintln!("Warning: Learning rate not set for a param group, defaulting to 0.0");
                0.0
            }))
            .collect()
    }
}

impl<'a, O: Optimizer + ?Sized> LRScheduler<'a, O> for ReduceLROnPlateau<'a, O> {
    fn step(&mut self, _epoch: Option<u64>, metrics: Option<f32>) -> Result<(), NeuraRustError> {
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
        self.get_current_lrs()
    }

    fn optimizer(&self) -> &O {
        self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        self.optimizer
    }
}

/// Implements the StepLR learning rate scheduler.
///
/// Decays the learning rate of each parameter group by gamma every step_size epochs.
#[derive(Debug)]
pub struct StepLR<'a, O: Optimizer + ?Sized> {
    optimizer: &'a mut O,
    step_size: u64,
    gamma: f32,
    last_epoch: i64,
    base_lrs: Vec<f32>,
}

impl<'a, O: Optimizer + ?Sized> StepLR<'a, O> {
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
    pub fn new(optimizer: &'a mut O, step_size: u64, gamma: f32) -> Self {
        if step_size == 0 {
            panic!("StepLR: step_size must be greater than 0.");
        }
        if gamma <= 0.0 {
            panic!("Factor gamma must be positive.");
        }
        let base_lrs: Vec<f32> = optimizer
            .param_groups()
            .iter()
            .map(|pg| pg.options.lr.expect("Optimizer param group missing LR at StepLR init"))
            .collect();

        StepLR {
            optimizer,
            step_size,
            gamma,
            last_epoch: -1,
            base_lrs,
        }
    }
}

impl<'a, O: Optimizer + ?Sized> LRScheduler<'a, O> for StepLR<'a, O> {
    fn step(&mut self, epoch: Option<u64>, _metrics: Option<f32>) -> Result<(), NeuraRustError> {
        let current_epoch_val = match epoch {
            Some(e) => e as i64,
            None => self.last_epoch + 1, // Avance l'epoch si non fournie
        };
        self.last_epoch = current_epoch_val;

        if self.last_epoch >= 0 { // Ne rien faire si l'epoch est < 0 (avant le premier step)
            // get_last_lr() calcule déjà les LRs corrects pour self.last_epoch
            let lrs_to_apply = self.get_last_lr(); 
            
            let mut lrs_changed_in_optimizer = false;
            for (i, pg) in self.optimizer.param_groups_mut().iter_mut().enumerate() {
                if let Some(new_lr_for_group) = lrs_to_apply.get(i) {
                    if pg.options.lr != Some(*new_lr_for_group) { 
                        pg.options.lr = Some(*new_lr_for_group);
                        lrs_changed_in_optimizer = true;
                    }
                }
            }
            if lrs_changed_in_optimizer {
                 println!("[StepLR] Epoch {}: Applied learning rates to optimizer: {:?}", self.last_epoch, lrs_to_apply);
            }
        }
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        if self.last_epoch == -1 {
            return self.base_lrs.clone();
        }
        let exponent = (self.last_epoch as u64 / self.step_size) as i32;
        self.base_lrs
            .iter()
            .map(|&base_lr| base_lr * self.gamma.powi(exponent))
            .collect()
    }

    fn optimizer(&self) -> &O {
        self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        self.optimizer
    }
}

/// Implements the MultiStepLR learning rate scheduler.
///
/// Decays the learning rate of each parameter group by gamma once the epoch reaches one of the milestones.
#[derive(Debug)]
pub struct MultiStepLR<'a, O: Optimizer + ?Sized> {
    optimizer: &'a mut O,
    milestones: Vec<u64>, // Sorted list of epoch indices at which to decay LR.
    gamma: f32,
    last_epoch: i64,      // Modifié en i64, nombre de fois step() a été appelé.
    base_lrs: Vec<f32>,
}

impl<'a, O: Optimizer + ?Sized> MultiStepLR<'a, O> {
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
    pub fn new(optimizer: &'a mut O, mut milestones: Vec<u64>, gamma: f32) -> Self {
        if gamma <= 0.0 {
            panic!("Factor gamma must be positive.");
        }
        milestones.sort_unstable();
        milestones.dedup(); // Ajout de dedup
        
        let base_lrs: Vec<f32> = optimizer // Initialisation correcte de base_lrs
            .param_groups()
            .iter()
            .map(|pg| pg.options.lr.expect("Optimizer param group missing LR at MultiStepLR init"))
            .collect();

        MultiStepLR {
            optimizer,
            milestones,
            gamma,
            last_epoch: -1, // Initialisé à -1
            base_lrs,
        }
    }
}

impl<'a, O: Optimizer + ?Sized> LRScheduler<'a, O> for MultiStepLR<'a, O> {
    fn get_last_lr(&self) -> Vec<f32> {
        if self.last_epoch == -1 { // Gestion de l'état initial
            return self.base_lrs.clone();
        }
        // self.last_epoch est maintenant 0, 1, 2...
        let current_epoch_as_u64 = self.last_epoch as u64;
        let num_decays = self.milestones.iter().filter(|&&m| m <= current_epoch_as_u64).count();
        let scale_factor = self.gamma.powi(num_decays as i32);
        self.base_lrs.iter().map(|&base_lr| base_lr * scale_factor).collect()
    }

    fn step(&mut self, epoch: Option<u64>, _metrics: Option<f32>) -> Result<(), NeuraRustError> {
        let current_epoch_val = match epoch {
            Some(e) => e as i64,
            None => self.last_epoch + 1,
        };
        self.last_epoch = current_epoch_val;

        if self.last_epoch >= 0 { // Appliquer seulement après le premier pas effectif
            let lrs_to_apply = self.get_last_lr();
            
            let mut lrs_changed_in_optimizer = false;
            for (i, pg) in self.optimizer.param_groups_mut().iter_mut().enumerate() {
                if let Some(new_lr_for_group) = lrs_to_apply.get(i) {
                    if pg.options.lr != Some(*new_lr_for_group) {
                        pg.options.lr = Some(*new_lr_for_group);
                        lrs_changed_in_optimizer = true;
                    }
                }
            }
            if lrs_changed_in_optimizer {
                 println!("[MultiStepLR] Epoch {}: Applied learning rates to optimizer: {:?}", self.last_epoch, lrs_to_apply);
            }
        }
        Ok(())
    }

    fn optimizer(&self) -> &O {
        self.optimizer
    }

    fn optimizer_mut(&mut self) -> &mut O {
        self.optimizer
    }
}

#[cfg(test)]
#[path = "lr_scheduler_tests.rs"]
mod tests; 