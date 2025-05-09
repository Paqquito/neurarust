use crate::optim::Optimizer;
use crate::optim::param_group::{ParamGroup, ParamGroupOptions};
use super::{NeuraRustError, StepLR, ReduceLROnPlateau, ReduceLROnPlateauMode, LRScheduler, MultiStepLR};
use std::sync::{Arc, RwLock};
use crate::nn::parameter::Parameter;
use crate::optim::OptimizerState;

#[derive(Debug, Clone)]
pub(super) struct MockOptimizer {
    pub(super) param_groups: Vec<ParamGroup>,
}

impl Optimizer for MockOptimizer {
    fn param_groups_mut(&mut self) -> &mut [ParamGroup] { 
        self.param_groups.as_mut_slice() 
    }
    fn param_groups(&self) -> &[ParamGroup] { 
        self.param_groups.as_slice() 
    }

    fn step(&mut self) -> Result<(), NeuraRustError> { Ok(()) } 
    fn zero_grad(&mut self) {} 
    fn add_param_group(&mut self, param_group: ParamGroup) { self.param_groups.push(param_group); }
    fn load_state_dict(&mut self, _state_dict: &crate::optim::OptimizerState) -> Result<(), NeuraRustError> { Ok(()) }
    fn state_dict(&self) -> Result<OptimizerState, NeuraRustError> { 
        Err(NeuraRustError::UnsupportedOperation("state_dict not implemented for MockOptimizer".to_string()))
    }
}

impl MockOptimizer {
    pub(super) fn new(lrs: Vec<f32>) -> Self {
        let param_groups = lrs.into_iter().map(|lr_val| {
            let mut options = ParamGroupOptions::default();
            options.lr = Some(lr_val);
            let mock_param_tensor = crate::tensor::Tensor::new(vec![1.0f32], vec![1]).expect("Failed to create mock tensor");
            let mock_parameter = Parameter::new(mock_param_tensor, Some("mock_param".to_string()));
            let mut pg = ParamGroup::new(vec![Arc::new(RwLock::new(mock_parameter))]);
            pg.options = options;
            pg
        }).collect();
        Self { param_groups }
    }
}

mod reduce_lr_plateau_tests {
    use super::*;

    #[test]
    fn test_reduce_lr_plateau_new_defaults() {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = ReduceLROnPlateau::new(
            &mut optimizer,
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
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        ReduceLROnPlateau::new(
            &mut optimizer,
            ReduceLROnPlateauMode::Min,
            Some(1.5), None, None, None, None, None, None,
        );
    }

    #[test]
    fn test_reduce_lr_plateau_step_requires_metric() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = ReduceLROnPlateau::new(
            &mut optimizer,
            ReduceLROnPlateauMode::Min,
            None, Some(0), None, None, None, None, None,
        );
        let result = scheduler.step(None, None);
        assert!(matches!(result, Err(NeuraRustError::ConfigurationError(_))));
        Ok(())
    }

    #[test]
    fn test_reduce_lr_plateau_min_lr_clamp() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![0.0125]);
        let mut scheduler = ReduceLROnPlateau::new(
            &mut optimizer, 
            ReduceLROnPlateauMode::Min,
            Some(0.5),    
            Some(0),      
            Some(1e-4),   
            None,         
            Some(0),      
            Some(0.01),   
            Some(1e-9),   
        );
        scheduler.step(Some(0), Some(10.0))?;
        assert_eq!(scheduler.best, 10.0);
        assert_eq!(scheduler.get_last_lr(), vec![0.0125]);
        scheduler.step(Some(1), Some(11.0))?;
        let current_lr = scheduler.get_last_lr()[0];
        assert!((current_lr - 0.01).abs() < 1e-7, "LR should clamp to 0.01");
        scheduler.step(Some(2), Some(12.0))?;
        let current_lr_after_clamp = scheduler.get_last_lr()[0];
        assert!((current_lr_after_clamp - 0.01).abs() < 1e-7, "LR should stay clamped to 0.01");
        Ok(())
    }
}

mod step_lr_tests {
    use super::*;

    #[test]
    fn test_step_lr_new() {
        let mut optimizer = MockOptimizer::new(vec![0.1, 0.01]);
        let scheduler = StepLR::new(&mut optimizer, 3, 0.1);
        assert_eq!(scheduler.step_size, 3);
        assert_eq!(scheduler.gamma, 0.1);
        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![0.1, 0.01]);
    }

    #[test]
    #[should_panic(expected = "StepLR: step_size must be greater than 0.")]
    fn test_step_lr_new_panic_on_zero_step_size() {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        StepLR::new(&mut optimizer, 0, 0.1); 
    }

    #[test]
    fn test_step_lr_basic_decay() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = StepLR::new(&mut optimizer, 2, 0.5);

        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![0.1], "LR initial avant le premier step");

        scheduler.step(Some(0), None)?;
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler.get_last_lr(), vec![0.1], "LR ne doit pas changer à l'epoch 0");

        scheduler.step(Some(1), None)?;
        assert_eq!(scheduler.last_epoch, 1);
        assert_eq!(scheduler.get_last_lr(), vec![0.1], "LR should not change at epoch 1 for step_size 2");

        scheduler.step(Some(2), None)?;
        assert_eq!(scheduler.last_epoch, 2);
        assert_eq!(scheduler.get_last_lr(), vec![0.05], "LR should decay at epoch 2");

        scheduler.step(Some(3), None)?;
        assert_eq!(scheduler.last_epoch, 3);
        assert_eq!(scheduler.get_last_lr(), vec![0.05], "LR should not change at epoch 3");

        scheduler.step(Some(4), None)?;
        assert_eq!(scheduler.last_epoch, 4);
        assert_eq!(scheduler.get_last_lr(), vec![0.025], "LR should decay again at epoch 4");

        Ok(())
    }

    #[test]
    fn test_step_lr_multiple_param_groups() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![1.0, 0.5]);
        let mut scheduler = StepLR::new(&mut optimizer, 1, 0.1);

        assert_eq!(scheduler.get_last_lr(), vec![1.0, 0.5]);

        scheduler.step(Some(0), None)?;
        let lrs_e0 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 0);
        assert!((lrs_e0[0] - 1.0).abs() < f32::EPSILON);
        assert!((lrs_e0[1] - 0.5).abs() < f32::EPSILON);

        scheduler.step(Some(1), None)?;
        let lrs_e1 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 1);
        assert!((lrs_e1[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs_e1[1] - 0.05).abs() < f32::EPSILON);

        scheduler.step(Some(2), None)?;
        let lrs_e2 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 2);
        assert!((lrs_e2[0] - 0.01).abs() < f32::EPSILON);
        assert!((lrs_e2[1] - 0.005).abs() < f32::EPSILON);
        
        Ok(())
    }

    #[test]
    fn test_step_lr_get_optimizer_refs() {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let mut scheduler = StepLR::new(&mut optimizer, 2, 0.5);

        let _opt_ref_immutable = scheduler.optimizer();

        scheduler.optimizer_mut().param_groups_mut()[0].options.lr = Some(0.99);
        assert_eq!(scheduler.get_last_lr(), vec![0.1], "get_last_lr should return LR based on initial base_lrs and current epoch, not reflect later direct optimizer changes to LR for base_lrs calculation");
    
        scheduler.step(Some(0), None).unwrap();
        assert_eq!(scheduler.optimizer.param_groups[0].options.lr, Some(0.1), "Step should apply the calculated LR, overriding external changes");
    }
}

mod multistep_lr_tests {
    use super::*;

    #[test]
    fn test_multistep_lr_new() {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let milestones = vec![2, 5, 8];
        let scheduler = MultiStepLR::new(&mut optimizer, milestones.clone(), 0.1);
        assert_eq!(scheduler.milestones, milestones.iter().map(|&x| x as u64).collect::<Vec<u64>>());
        assert_eq!(scheduler.gamma, 0.1);
        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);
    }

    #[test]
    fn test_multistep_lr_new_sorts_and_deduplicates_milestones() {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = MultiStepLR::new(&mut optimizer, vec![5, 2, 8, 2, 5, 5], 0.1);
        assert_eq!(scheduler.milestones, vec![2, 5, 8].iter().map(|&x| x as u64).collect::<Vec<u64>>(), "Milestones should be sorted and deduplicated");
    }

    #[test]
    fn test_multistep_lr_new_empty_milestones() {
        let mut optimizer = MockOptimizer::new(vec![0.1]);
        let scheduler = MultiStepLR::new(&mut optimizer, vec![], 0.1);
        assert_eq!(scheduler.milestones, Vec::<u64>::new());
        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);
    }

    #[test]
    fn test_multistep_lr_basic_decay() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![1.0]);
        let mut scheduler = MultiStepLR::new(&mut optimizer, vec![2, 4], 0.1);

        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![1.0], "Initial LR");

        scheduler.step(Some(0), None)?;
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler.get_last_lr(), vec![1.0], "LR shouldn't change at epoch 0");

        scheduler.step(Some(1), None)?;
        assert_eq!(scheduler.last_epoch, 1);
        assert_eq!(scheduler.get_last_lr(), vec![1.0], "LR shouldn't change at epoch 1");

        scheduler.step(Some(2), None)?;
        assert_eq!(scheduler.last_epoch, 2);
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "LR should decay at epoch 2");

        scheduler.step(Some(3), None)?;
        assert_eq!(scheduler.last_epoch, 3);
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "LR should be 0.1 at epoch 3");

        scheduler.step(Some(4), None)?;
        assert_eq!(scheduler.last_epoch, 4);
        assert!((scheduler.get_last_lr()[0] - 0.01).abs() < f32::EPSILON, "LR should decay at epoch 4");
        Ok(())
    }

    #[test]
    fn test_multistep_lr_multiple_param_groups() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![1.0, 0.4]);
        let mut scheduler = MultiStepLR::new(&mut optimizer, vec![1, 3], 0.1);

        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![1.0, 0.4], "Initial LRs");

        scheduler.step(Some(0), None)?;
        assert_eq!(scheduler.last_epoch, 0);
        let lrs_e0 = scheduler.get_last_lr();
        assert!((lrs_e0[0] - 1.0).abs() < f32::EPSILON);
        assert!((lrs_e0[1] - 0.4).abs() < f32::EPSILON);

        scheduler.step(Some(1), None)?;
        let lrs_e1 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 1);
        assert!((lrs_e1[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs_e1[1] - 0.04).abs() < f32::EPSILON);

        scheduler.step(Some(2), None)?;
        let lrs_e2 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 2);
        assert!((lrs_e2[0] - 0.1).abs() < f32::EPSILON);
        assert!((lrs_e2[1] - 0.04).abs() < f32::EPSILON);

        scheduler.step(Some(3), None)?;
        let lrs_e3 = scheduler.get_last_lr();
        assert_eq!(scheduler.last_epoch, 3);
        assert!((lrs_e3[0] - 0.01).abs() < f32::EPSILON);
        assert!((lrs_e3[1] - 0.004).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_multistep_lr_no_milestones() -> Result<(), NeuraRustError> {
        let mut optimizer = MockOptimizer::new(vec![1.0]);
        let mut scheduler = MultiStepLR::new(&mut optimizer, vec![], 0.1);

        assert_eq!(scheduler.last_epoch, -1);
        assert_eq!(scheduler.get_last_lr(), vec![1.0]);
        scheduler.step(Some(0), None)?;
        assert_eq!(scheduler.last_epoch, 0);
        assert_eq!(scheduler.get_last_lr(), vec![1.0], "LR shouldn't change with no milestones");
        scheduler.step(Some(1), None)?;
        assert_eq!(scheduler.last_epoch, 1);
        assert_eq!(scheduler.get_last_lr(), vec![1.0], "LR shouldn't change with no milestones");
        Ok(())
    }

    #[test]
    fn test_multistep_lr_milestone_at_zero_or_one_behavior() -> Result<(), NeuraRustError> {
        let mut optimizer_m0 = MockOptimizer::new(vec![1.0]);
        let mut scheduler_m0 = MultiStepLR::new(&mut optimizer_m0, vec![0, 2], 0.1);
        assert_eq!(scheduler_m0.last_epoch, -1);
        assert_eq!(scheduler_m0.get_last_lr(), vec![1.0], "Initial LR for m0");

        scheduler_m0.step(Some(0), None)?;
        assert_eq!(scheduler_m0.last_epoch, 0);
        assert!((scheduler_m0.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "Decay at epoch 0 due to milestone 0");

        scheduler_m0.step(Some(1), None)?;
        assert_eq!(scheduler_m0.last_epoch, 1);
        assert!((scheduler_m0.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "LR is 0.1 at epoch 1");

        scheduler_m0.step(Some(2), None)?;
        assert_eq!(scheduler_m0.last_epoch, 2);
        assert!((scheduler_m0.get_last_lr()[0] - 0.01).abs() < f32::EPSILON, "Decay at epoch 2 due to milestones 0, 2");

        let mut optimizer_m1 = MockOptimizer::new(vec![1.0]);
        let mut scheduler_m1 = MultiStepLR::new(&mut optimizer_m1, vec![1, 3], 0.1);
        assert_eq!(scheduler_m1.last_epoch, -1);
        assert_eq!(scheduler_m1.get_last_lr(), vec![1.0], "Initial LR for m1");

        scheduler_m1.step(Some(0), None)?;
        assert_eq!(scheduler_m1.last_epoch, 0);
        assert!((scheduler_m1.get_last_lr()[0] - 1.0).abs() < f32::EPSILON, "No decay at epoch 0 for m1");

        scheduler_m1.step(Some(1), None)?;
        assert_eq!(scheduler_m1.last_epoch, 1);
        assert!((scheduler_m1.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "Decay at epoch 1 due to milestone 1");

        scheduler_m1.step(Some(2), None)?;
        assert_eq!(scheduler_m1.last_epoch, 2);
        assert!((scheduler_m1.get_last_lr()[0] - 0.1).abs() < f32::EPSILON, "LR is 0.1 at epoch 2 for m1");

        scheduler_m1.step(Some(3), None)?;
        assert_eq!(scheduler_m1.last_epoch, 3);
        assert!((scheduler_m1.get_last_lr()[0] - 0.01).abs() < f32::EPSILON, "Decay at epoch 3 due to milestones 1, 3 for m1");
        Ok(())
    }
} 