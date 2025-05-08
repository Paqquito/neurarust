#[cfg(test)]
mod tests {
    use crate::nn::parameter::Parameter;
    use crate::optim::{adagrad::AdagradOptimizer, Optimizer};
    use crate::optim::optimizer_state::OptimizerState;
    
    use crate::tensor::create::from_vec_f32;
    use std::sync::{Arc, Mutex};

    fn create_param(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Arc<Mutex<Parameter>> {
        let tensor = from_vec_f32(data, shape.to_vec()).unwrap();
        if requires_grad {
            tensor.set_requires_grad(true).unwrap();
        }
        Arc::new(Mutex::new(Parameter::new(tensor, Some("param_test".to_string()))))
    }

    #[test]
    fn adagrad_optimizer_creation() {
        let param = create_param(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let params_vec = vec![param];
        let optimizer = AdagradOptimizer::new(
            params_vec.into_iter(),
            0.1, // lr
            0.0, // lr_decay
            0.0, // weight_decay
            0.0, // initial_accumulator_value
            1e-8, // eps
        );
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_adagrad_basic_step() {
        let initial_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let param = create_param(initial_data.clone(), &[2, 2], true);
        
        let grad_data = vec![0.1f32, 0.2, 0.3, 0.4];
        let grad_tensor = from_vec_f32(grad_data.clone(), vec![2, 2]).unwrap();
        
        param.lock().unwrap().acc_grad(grad_tensor.clone()).unwrap();

        let lr = 0.1;
        let eps = 1e-8;
        let initial_accumulator_value = 0.0;

        let mut optimizer = AdagradOptimizer::new(
            vec![param.clone()].into_iter(),
            lr,
            0.0, // lr_decay
            0.0, // weight_decay
            initial_accumulator_value,
            eps,
        ).unwrap();

        optimizer.step().unwrap();
        
        let p_guard = param.lock().unwrap();
        let updated_data = p_guard.tensor.get_f32_data().unwrap();

        for i in 0..initial_data.len() {
            let p_old = initial_data[i];
            let g = grad_data[i];
            let state_sum_sq_grad = initial_accumulator_value + g * g;
            let effective_lr_denom = (state_sum_sq_grad as f64).sqrt() + eps as f64;
            let effective_lr = lr as f64 / effective_lr_denom;
            let p_new_expected = p_old as f64 - effective_lr * g as f64;
            assert!((updated_data[i] as f64 - p_new_expected).abs() < 1e-6, "Mismatch for element {}: expected {}, got {}", i, p_new_expected, updated_data[i]);
        }
        
        drop(p_guard);

        let state_group = &optimizer.state_sum_gradient_squares[0];
        let param_state_sum = &state_group[0];
        let state_data = param_state_sum.get_f32_data().unwrap();

        for i in 0..grad_data.len() {
            let g = grad_data[i];
            let expected_state_val = initial_accumulator_value + g * g;
            assert!((state_data[i] - expected_state_val).abs() < 1e-6, "Mismatch for state element {}: expected {}, got {}", i, expected_state_val, state_data[i]);
        }
    }

    #[test]
    fn test_adagrad_weight_decay() {
        let initial_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let param = create_param(initial_data.clone(), &[2, 2], true);
        
        let grad_data = vec![0.0f32, 0.0, 0.0, 0.0]; 
        let grad_tensor = from_vec_f32(grad_data.clone(), vec![2, 2]).unwrap();
        
        param.lock().unwrap().acc_grad(grad_tensor.clone()).unwrap();

        let lr = 0.1;
        let weight_decay = 0.01;
        let eps = 1e-8;
        let initial_accumulator_value = 0.0;

        let mut optimizer = AdagradOptimizer::new(
            vec![param.clone()].into_iter(),
            lr,
            0.0,
            weight_decay,
            initial_accumulator_value,
            eps,
        ).unwrap();

        optimizer.step().unwrap();
        
        let p_guard = param.lock().unwrap();
        let updated_data = p_guard.tensor.get_f32_data().unwrap();

        for i in 0..initial_data.len() {
            let p_old = initial_data[i];
            let g = grad_data[i];
            
            let grad_effective = g + weight_decay * p_old;
            let state_sum_sq_grad = initial_accumulator_value + grad_effective * grad_effective;
            let effective_lr_denom = (state_sum_sq_grad as f64).sqrt() + eps as f64;
            let effective_lr = lr as f64 / effective_lr_denom;
            let p_new_expected = p_old as f64 - effective_lr * grad_effective as f64;
            
            assert!((updated_data[i] as f64 - p_new_expected).abs() < 1e-6, 
                    "Weight decay test mismatch for element {}: expected {}, got {}. Effective grad: {}, State sum: {}, Eff LR: {}", 
                    i, p_new_expected, updated_data[i], grad_effective, state_sum_sq_grad, effective_lr);
        }
    }

    #[test]
    fn test_adagrad_lr_decay() {
        let initial_param_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let param = create_param(initial_param_data.clone(), &[2, 2], true);
        
        let grad_values = vec![0.1f32, 0.2, 0.3, 0.4];
        let grad_tensor = from_vec_f32(grad_values.clone(), vec![2, 2]).unwrap();
        
        // Appliquer le gradient une seule fois au début
        param.lock().unwrap().acc_grad(grad_tensor.clone()).unwrap();

        let base_lr = 1.0;
        let decay_rate = 0.1;
        let eps_val = 1e-8;
        let initial_acc_val = 0.0;
        let wd = 0.0;
        let steps_to_run = 3;

        let mut optimizer = AdagradOptimizer::new(
            vec![param.clone()].into_iter(),
            base_lr,
            decay_rate,
            wd,
            initial_acc_val,
            eps_val,
        ).unwrap();

        let mut current_param_values_manual = initial_param_data.clone();
        // State pour le calcul manuel, un par élément du paramètre
        let mut manual_sum_sq_grads: Vec<f32> = vec![initial_acc_val; initial_param_data.len()]; 

        for step_num in 1..=steps_to_run {
            // L'optimizer effectue son pas
            optimizer.step().unwrap();
            
            // Récupérer les valeurs mises à jour par l'optimizer
            let param_guard_after_step = param.lock().unwrap();
            let updated_param_values_optimizer = param_guard_after_step.tensor.get_f32_data().unwrap();
            drop(param_guard_after_step);

            // Calcul manuel pour ce pas
            let decayed_lr_for_this_step = base_lr / (1.0 + (step_num as f32 - 1.0) * decay_rate);
            println!(
                "Step {}: Decayed LR: {}, Optimizer updated values: {:?}",
                step_num,
                decayed_lr_for_this_step,
                updated_param_values_optimizer
            );

            let mut next_param_values_manual = vec![0.0f32; current_param_values_manual.len()];

            for i in 0..current_param_values_manual.len() {
                let p_old = current_param_values_manual[i];
                let g = grad_values[i]; // Gradient constant pour ce test

                // Adagrad met à jour son état AVANT de calculer la mise à jour du paramètre
                manual_sum_sq_grads[i] += g * g;
                
                let denom = (manual_sum_sq_grads[i] as f64).sqrt() + eps_val as f64;
                let update_amount = (decayed_lr_for_this_step as f64 * g as f64) / denom;
                let p_new_manual = p_old as f64 - update_amount;
                next_param_values_manual[i] = p_new_manual as f32;

                assert!(
                    (updated_param_values_optimizer[i] as f64 - p_new_manual).abs() < 1e-5,
                    "LR Decay mismatch: Step {}, Elem {}. Optimizer: {}, Manual: {}. DecayedLR: {}, ManualSumSqGrad: {}, Denom: {}",
                    step_num, i, updated_param_values_optimizer[i], p_new_manual, decayed_lr_for_this_step, manual_sum_sq_grads[i], denom
                );
            }
            // Mettre à jour les valeurs manuelles pour le prochain pas
            current_param_values_manual = updated_param_values_optimizer.to_vec(); 
        }
    }

    #[test]
    fn test_adagrad_state_dict() {
        // -- Setup --
        let initial_data1 = vec![1.0f32, 2.0];
        let param1 = create_param(initial_data1.clone(), &[2], true);
        let initial_data2 = vec![3.0f32, 4.0];
        let param2 = create_param(initial_data2.clone(), &[2], true);

        let grad_data1 = vec![0.1f32, 0.2];
        let grad_tensor1 = from_vec_f32(grad_data1.clone(), vec![2]).unwrap();
        let grad_data2 = vec![0.3f32, 0.4];
        let grad_tensor2 = from_vec_f32(grad_data2.clone(), vec![2]).unwrap();

        let lr = 0.1;
        let eps = 1e-8;
        let initial_acc = 0.0;
        let steps_before_save = 2;

        let mut optimizer1 = AdagradOptimizer::new(
            vec![param1.clone(), param2.clone()].into_iter(),
            lr, 0.0, 0.0, initial_acc, eps
        ).unwrap();

        // -- Run optimizer 1 for some steps --
        for _ in 0..steps_before_save {
            // Simuler zero_grad + backward avant chaque step
            param1.lock().unwrap().tensor.clear_grad(); // Effacer ancien grad si présent
            param2.lock().unwrap().tensor.clear_grad();
            param1.lock().unwrap().acc_grad(grad_tensor1.clone()).unwrap();
            param2.lock().unwrap().acc_grad(grad_tensor2.clone()).unwrap();
            optimizer1.step().unwrap();
        }

        // -- Sauvegarder état et valeurs des paramètres --
        let state_dict_saved = optimizer1.state_dict().unwrap();
        let param1_val_after_n_steps = param1.lock().unwrap().tensor.get_f32_data().unwrap();
        let param2_val_after_n_steps = param2.lock().unwrap().tensor.get_f32_data().unwrap();

        // -- Exécuter un pas de plus avec optimizer1 pour référence --
        param1.lock().unwrap().tensor.clear_grad();
        param2.lock().unwrap().tensor.clear_grad();
        param1.lock().unwrap().acc_grad(grad_tensor1.clone()).unwrap();
        param2.lock().unwrap().acc_grad(grad_tensor2.clone()).unwrap();
        optimizer1.step().unwrap(); // Step N+1
        let param1_val_ref = param1.lock().unwrap().tensor.get_f32_data().unwrap();
        let param2_val_ref = param2.lock().unwrap().tensor.get_f32_data().unwrap();

        // -- Créer un nouvel optimiseur et de nouveaux paramètres --
        // Les valeurs initiales n'importent pas, on va les écraser.
        let param1_new = create_param(vec![10.0, 20.0], &[2], true);
        let param2_new = create_param(vec![30.0, 40.0], &[2], true);
        let mut optimizer2 = AdagradOptimizer::new(
            vec![param1_new.clone(), param2_new.clone()].into_iter(),
            lr, 0.0, 0.0, initial_acc, eps
        ).unwrap();

        // -- Charger l'état et restaurer les valeurs des paramètres --
        optimizer2.load_state_dict(&state_dict_saved).unwrap();
        // Restaurer la valeur des paramètres pour simuler la reprise
        param1_new.lock().unwrap().tensor = from_vec_f32(param1_val_after_n_steps.clone(), vec![2]).unwrap();
        param2_new.lock().unwrap().tensor = from_vec_f32(param2_val_after_n_steps.clone(), vec![2]).unwrap();
        // Il faut aussi remettre requires_grad sur les nouveaux tenseurs 
        param1_new.lock().unwrap().tensor.set_requires_grad(true).unwrap();
        param2_new.lock().unwrap().tensor.set_requires_grad(true).unwrap();

        // -- Exécuter un pas avec optimizer2 --
        // Simuler zero_grad + backward
        param1_new.lock().unwrap().tensor.clear_grad();
        param2_new.lock().unwrap().tensor.clear_grad();
        param1_new.lock().unwrap().acc_grad(grad_tensor1.clone()).unwrap();
        param2_new.lock().unwrap().acc_grad(grad_tensor2.clone()).unwrap();
        optimizer2.step().unwrap(); // Step N+1 simulé

        // -- Comparer les résultats --
        let param1_val_new = param1_new.lock().unwrap().tensor.get_f32_data().unwrap();
        let param2_val_new = param2_new.lock().unwrap().tensor.get_f32_data().unwrap();

        assert_eq!(param1_val_new, param1_val_ref, "Mismatch in param1 after loading state");
        assert_eq!(param2_val_new, param2_val_ref, "Mismatch in param2 after loading state");

        // Vérifier aussi l'état interne de l'optimiseur après le pas N+1
        let final_state_dict1 = optimizer1.state_dict().unwrap();
        let final_state_dict2 = optimizer2.state_dict().unwrap();
        // Note: Comparer directement les OptimizerState peut nécessiter PartialEq
        // Comparons les champs manuellement pour l'instant
        if let (OptimizerState::Adagrad { state_sum_gradient_squares: s1, steps: st1 }, 
                OptimizerState::Adagrad { state_sum_gradient_squares: s2, steps: st2 }) = (&final_state_dict1, &final_state_dict2) {
            assert_eq!(st1, st2, "Steps mismatch after N+1 steps");
            assert_eq!(s1.len(), s2.len(), "Number of state groups mismatch");
            for g in 0..s1.len() {
                assert_eq!(s1[g].len(), s2[g].len(), "Number of states in group {} mismatch", g);
                for p in 0..s1[g].len() {
                    assert_eq!(s1[g][p].get_f32_data().unwrap(), s2[g][p].get_f32_data().unwrap(), 
                               "State sum mismatch for group {}, param {}", g, p);
                }
            }
        } else {
            panic!("Unexpected OptimizerState variant");
        }
    }

    // TODO: Add more tests for step, lr_decay, weight_decay, state_dict, etc.
} 