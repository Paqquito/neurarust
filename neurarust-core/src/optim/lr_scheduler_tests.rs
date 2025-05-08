use super::{OptimizerInterface, ParamGroupInterface, NeuraRustError, StepLR, ReduceLROnPlateau, ReduceLROnPlateauMode, LRScheduler, MultiStepLR};

#[derive(Debug, Clone)]
pub(super) struct MockParamGroup {
    lr: f32,
    name: String, 
}

impl MockParamGroup {
    fn new(lr: f32) -> Self {
        MockParamGroup { lr, name: "mock_param".to_string() }
    }
}

impl ParamGroupInterface for MockParamGroup {
    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
    fn param_names(&self) -> Vec<String> { vec![self.name.clone()] }
}

#[derive(Debug, Clone)]
pub(super) struct MockOptimizer {
    pub(super) param_groups: Vec<MockParamGroup>,
}

impl OptimizerInterface for MockOptimizer {
    type ParamGroup = MockParamGroup;
    fn param_groups_mut(&mut self) -> &mut [Self::ParamGroup] { self.param_groups.as_mut_slice() }
    fn param_groups(&self) -> &[Self::ParamGroup] { self.param_groups.as_slice() }
}

impl MockOptimizer {
    pub(super) fn new(lrs: Vec<f32>) -> Self {
        Self { param_groups: lrs.into_iter().map(MockParamGroup::new).collect() }
    }
}

mod reduce_lr_plateau_tests {
    use super::*;

    #[test]
    fn test_reduce_lr_plateau_new_defaults() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = ReduceLROnPlateau::new(
            optimizer,
            ReduceLROnPlateauMode::Min,
            None, None, None, None, None, None, None,
        );
        assert_eq!(scheduler.mode, ReduceLROnPlateauMode::Min);
        assert!((scheduler.factor - 0.1).abs() < f32::EPSILON);
        assert_eq!(scheduler.patience, 10);
    }

    #[test]
    #[should_panic(expected = "ReduceLROnPlateau: factor must be between 0.0 and 1.0.")]
    fn test_reduce_lr_plateau_new_invalid_factor() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        ReduceLROnPlateau::new(
            optimizer,
            ReduceLROnPlateauMode::Min,
            Some(1.5), None, None, None, None, None, None,
        );
    }

    #[test]
    fn test_reduce_lr_plateau_step_requires_metric() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = ReduceLROnPlateau::new(
            optimizer,
            ReduceLROnPlateauMode::Min,
            None, Some(0), None, None, None, None, None,
        );
        let result = scheduler.step(None, None);
        assert!(matches!(result, Err(NeuraRustError::ConfigurationError(_))));
        Ok(())
    }

    #[test]
    fn test_reduce_lr_plateau_min_lr_clamp() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![0.0125]);
        let mut scheduler = ReduceLROnPlateau::new(
            optimizer, 
            ReduceLROnPlateauMode::Min,
            Some(0.5),    
            Some(0),      
            Some(1e-4),   
            None,         
            Some(0),      
            Some(0.01),   
            Some(1e-9),   
        );
        scheduler.step(None, Some(10.0))?;
        assert_eq!(scheduler.best, 10.0);
        assert_eq!(scheduler.get_last_lr(), vec![0.0125]);
        scheduler.step(None, Some(11.0))?;
        let current_lr = scheduler.get_last_lr()[0];
        assert!((current_lr - 0.01).abs() < 1e-7, "LR should clamp to 0.01");
        scheduler.step(None, Some(12.0))?;
        let current_lr_after_clamp = scheduler.get_last_lr()[0];
        assert!((current_lr_after_clamp - 0.01).abs() < 1e-7, "LR should stay clamped to 0.01");
        Ok(())
    }
}

mod step_lr_tests {
    use super::*;

    #[test]
    fn test_step_lr_new() {
        let optimizer = MockOptimizer::new(vec![0.1, 0.01]);
        let scheduler = StepLR::new(optimizer, 3, 0.1);
        assert_eq!(scheduler.step_size, 3);
        assert_eq!(scheduler.gamma, 0.1);
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler._base_lrs, vec![0.1, 0.01]);
    }

    #[test]
    #[should_panic(expected = "StepLR: step_size must be greater than 0.")]
    fn test_step_lr_new_panic_on_zero_step_size() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        StepLR::new(optimizer, 0, 0.1); 
    }

    #[test]
    fn test_step_lr_basic_decay() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = StepLR::new(optimizer, 2, 0.5);

        // Initial state: last_epoch = 0, LR = 0.1
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);

        // Step 1: last_epoch becomes 1. num_steps = 1/2 = 0. factor = 0.5^0 = 1. LR = 0.1 * 1 = 0.1
        scheduler.step(Some(1), None)?;
        assert_eq!(scheduler.last_epoch, 1);
        assert_eq!(scheduler.get_last_lr(), vec![0.1], "LR should not change at epoch 1 for step_size 2");

        // Step 2: last_epoch becomes 2. num_steps = 2/2 = 1. factor = 0.5^1 = 0.5. LR = 0.1 * 0.5 = 0.05
        scheduler.step(Some(2), None)?;
        assert_eq!(scheduler.last_epoch, 2);
        assert_eq!(scheduler.get_last_lr(), vec![0.05], "LR should decay at epoch 2");

        // Step 3: last_epoch becomes 3. num_steps = 3/2 = 1. factor = 0.5^1 = 0.5. LR = 0.1 * 0.5 = 0.05
        scheduler.step(Some(3), None)?;
        assert_eq!(scheduler.last_epoch, 3);
        assert_eq!(scheduler.get_last_lr(), vec![0.05], "LR should not change at epoch 3");

        // Step 4: last_epoch becomes 4. num_steps = 4/2 = 2. factor = 0.5^2 = 0.25. LR = 0.1 * 0.25 = 0.025
        scheduler.step(Some(4), None)?;
        assert_eq!(scheduler.last_epoch, 4);
        assert_eq!(scheduler.get_last_lr(), vec![0.025], "LR should decay again at epoch 4");

        Ok(())
    }

    #[test]
    fn test_step_lr_multiple_param_groups() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0, 0.5]);
        let mut scheduler = StepLR::new(optimizer, 1, 0.1); // step_size = 1, decays every epoch

        // Base LRs: [1.0, 0.5]
        assert_eq!(scheduler._base_lrs, vec![1.0, 0.5]);
        assert_eq!(scheduler.get_last_lr(), vec![1.0, 0.5]);

        // Step 1: last_epoch = 1. num_steps = 1/1 = 1. factor = 0.1^1 = 0.1
        // LR1 = 1.0 * 0.1 = 0.1
        // LR2 = 0.5 * 0.1 = 0.05
        scheduler.step(Some(1), None)?;
        let lrs_e1 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 1);
        assert!((lrs_e1[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs_e1[1] - 0.05).abs() < f32::EPSILON);

        // Step 2: last_epoch = 2. num_steps = 2/1 = 2. factor = 0.1^2 = 0.01
        // LR1 = 1.0 * 0.01 = 0.01
        // LR2 = 0.5 * 0.01 = 0.005
        scheduler.step(Some(2), None)?;
        let lrs_e2 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 2);
        assert!((lrs_e2[0] - 0.01).abs() < f32::EPSILON);
        assert!((lrs_e2[1] - 0.005).abs() < f32::EPSILON);
        
        Ok(())
    }

    #[test]
    fn test_step_lr_get_optimizer_refs() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        // Need to bind to a mutable variable to call new, even if scheduler itself isn't mutated for this test directly.
        let mut scheduler = StepLR::new(optimizer, 2, 0.5);

        // Test immutable access
        let _opt_ref_immutable = scheduler.optimizer();
        // We can't easily verify much without more complex OptimizerInterface methods, 
        // but we can check it doesn't panic.

        // Test mutable access and its effect via get_last_lr
        scheduler.optimizer_mut().param_groups[0].set_lr(0.99);
        assert_eq!(scheduler.get_last_lr(), vec![0.99], "get_last_lr should reflect direct optimizer changes");
    }
}

mod multistep_lr_tests {
    use super::*;

    #[test]
    fn test_multistep_lr_new() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let milestones = vec![2, 5, 8];
        let scheduler = MultiStepLR::new(optimizer, milestones.clone(), 0.1);
        assert_eq!(scheduler.milestones, milestones);
        assert_eq!(scheduler.gamma, 0.1);
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler._base_lrs, vec![0.1]);
    }

    #[test]
    fn test_multistep_lr_new_sorts_and_deduplicates_milestones() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        // Milestones non triés et avec doublons
        let scheduler = MultiStepLR::new(optimizer, vec![5, 2, 8, 2, 5, 5], 0.1);
        assert_eq!(scheduler.milestones, vec![2, 5, 8], "Milestones should be sorted and deduplicated");
    }

    #[test]
    fn test_multistep_lr_new_empty_milestones() {
        let optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = MultiStepLR::new(optimizer, vec![], 0.1);
        assert_eq!(scheduler.milestones, Vec::<usize>::new());
        assert_eq!(scheduler.last_epoch, 0);
    }

    #[test]
    fn test_multistep_lr_basic_decay() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0]);
        let mut scheduler = MultiStepLR::new(optimizer, vec![2, 4], 0.1);

        // Initial: last_epoch = 0. num_milestones_passed = 0. factor = 0.1^0 = 1. LR = 1.0
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);

        // Epoch 1: last_epoch = 1. Milestones passed [2,4] for last_epoch=1 -> 0. factor=1. LR = 1.0
        scheduler.step(None, None)?;
        assert_eq!(scheduler.last_epoch, 1);
        assert_eq!(scheduler.get_last_lr(), vec![1.0], "LR shouldn't change at epoch 1");

        // Epoch 2: last_epoch = 2. Milestones passed [2,4] for last_epoch=2 -> 1 (milestone 2). factor=0.1^1=0.1. LR = 1.0 * 0.1 = 0.1
        scheduler.step(None, None)?;
        assert_eq!(scheduler.last_epoch, 2);
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "LR should decay at epoch 2");

        // Epoch 3: last_epoch = 3. Milestones passed [2,4] for last_epoch=3 -> 1. factor=0.1. LR = 1.0 * 0.1 = 0.1
        scheduler.step(None, None)?;
        assert_eq!(scheduler.last_epoch, 3);
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "LR shouldn't change at epoch 3");

        // Epoch 4: last_epoch = 4. Milestones passed [2,4] for last_epoch=4 -> 2 (m 2,4). factor=0.1^2=0.01. LR = 1.0 * 0.01 = 0.01
        scheduler.step(None, None)?;
        assert_eq!(scheduler.last_epoch, 4);
        assert!((scheduler.get_last_lr()[0] - 0.01).abs() < f32::EPSILON, "LR should decay at epoch 4");

        // Epoch 5: last_epoch = 5. Milestones passed [2,4] for last_epoch=5 -> 2. factor=0.01. LR = 1.0 * 0.01 = 0.01
        scheduler.step(None, None)?;
        assert_eq!(scheduler.last_epoch, 5);
        assert!((scheduler.get_last_lr()[0] - 0.01).abs() < f32::EPSILON, "LR shouldn't change at epoch 5");

        Ok(())
    }

    #[test]
    fn test_multistep_lr_multiple_param_groups() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0, 0.4]);
        let mut scheduler = MultiStepLR::new(optimizer, vec![1, 3], 0.1);

        // Base LRs: [1.0, 0.4]
        assert_eq!(scheduler.get_last_lr(), vec![1.0, 0.4]);

        // Step 1 (epoch 1): last_epoch = 1. Milestones passed for 1 -> 1 (m 1). factor = 0.1^1 = 0.1
        // LR1 = 1.0 * 0.1 = 0.1
        // LR2 = 0.4 * 0.1 = 0.04
        scheduler.step(None, None)?;
        let lrs1 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 1);
        assert!((lrs1[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs1[1] - 0.04).abs() < f32::EPSILON);

        // Step 2 (epoch 2): last_epoch = 2. Milestones passed for 2 -> 1. factor = 0.1
        // LR1 = 1.0 * 0.1 = 0.1
        // LR2 = 0.4 * 0.1 = 0.04
        scheduler.step(None, None)?;
        let lrs2 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 2);
        assert!((lrs2[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs2[1] - 0.04).abs() < f32::EPSILON);

        // Step 3 (epoch 3): last_epoch = 3. Milestones passed for 3 -> 2 (m 1,3). factor = 0.1^2 = 0.01
        // LR1 = 1.0 * 0.01 = 0.01
        // LR2 = 0.4 * 0.01 = 0.004
        scheduler.step(None, None)?;
        let lrs3 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 3);
        assert!((lrs3[0] - 0.01).abs() < f32::EPSILON);
        assert!((lrs3[1] - 0.004).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_multistep_lr_no_milestones() -> Result<(), NeuraRustError> {
        let optimizer = MockOptimizer::new(vec![1.0]);
        let mut scheduler = MultiStepLR::new(optimizer, vec![], 0.1); // No milestones

        assert_eq!(scheduler.get_last_lr(), vec![1.0]);
        scheduler.step(None, None)?; // last_epoch = 1. num_milestones = 0. factor = 1.
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);
        scheduler.step(None, None)?; // last_epoch = 2. num_milestones = 0. factor = 1.
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);
        Ok(())
    }

    #[test]
    fn test_multistep_lr_milestone_at_zero_or_one_behavior() -> Result<(), NeuraRustError> {
        // Milestones are epoch counts. last_epoch is number of steps taken.
        // If a milestone is 1, it means decay after the 1st step.
        let optimizer1 = MockOptimizer::new(vec![1.0]);
        let mut scheduler1 = MultiStepLR::new(optimizer1, vec![1], 0.1);
        
        // Initial LR = 1.0
        scheduler1.step(None, None)?; // last_epoch becomes 1. Milestone 1 is met. num_milestones_passed = 1. factor=0.1
        assert!((scheduler1.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "Decay should happen at epoch 1");

        // Test milestone at 0 (should be effectively ignored by `m <= last_epoch` if last_epoch starts at 0 and increments)
        // Or, if sorted, it's the first milestone. If last_epoch=0 (initial), num_passed=0.
        // If last_epoch=1 (after 1st step), and milestone 0 exists, it *is* <= 1.
        // PyTorch bisect_right([0], 0) is 1. So gamma^1 is applied. (Incorrect understanding previously)
        // PyTorch bisect_right([0], 1) is 1. gamma^1.
        // bisect_right(a, x) counts elements in a strictly less than x, then adds 1 if x is in a. No, it's number of elements <= x.
        // My `filter(|&&m| m <= self.last_epoch).count()` is `bisect_right`.
        // So, if milestones = [0, 2] and last_epoch = 0 (before any step), num_milestones_passed = 0. factor = 1.
        // After 1st step, last_epoch = 1. num_milestones_passed (for m=0) = 1. factor = gamma.
        let optimizer0 = MockOptimizer::new(vec![1.0]);
        let mut scheduler0 = MultiStepLR::new(optimizer0, vec![0, 2], 0.1);
        assert_eq!(scheduler0.get_last_lr(), vec![1.0]); // Before any step

        scheduler0.step(None, None)?; // last_epoch = 1. Milestones <= 1 is [0]. count = 1. factor = 0.1
        assert!((scheduler0.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "Decay due to milestone 0 at epoch 1");

        scheduler0.step(None, None)?; // last_epoch = 2. Milestones <= 2 is [0, 2]. count = 2. factor = 0.01
        assert!((scheduler0.get_last_lr()[0] - 0.01).abs() < f32::EPSILON, "Decay due to milestone 0 & 2 at epoch 2");

        Ok(())
    }
} 