use neurarust_core::tensor::Tensor;
use crate::Optimizer;
use num_traits::{Float, FromPrimitive, Zero, One, Pow};
use std::{collections::HashMap, fmt::Debug};
use std::ops::{AddAssign, Sub, Neg, Mul, Add, Div};
use std::iter::Sum as IterSum;


/// Implements the Adam optimization algorithm.
/// Reference: https://arxiv.org/abs/1412.6980
///
/// Stores the hyperparameters and the moving averages (moments) for each parameter.
/// Parameters themselves are passed during the `step` call.
#[derive(Debug)]
pub struct Adam<T: Float> {
    lr: T,           // Learning rate
    beta1: T,
    beta2: T,
    epsilon: T,
    t: u64, // Use u64 for timestep to avoid overflow and match common practice
    // State for each parameter (keyed by the stable ID obtained from Tensor::id())
    moments1: HashMap<*const (), Tensor<T>>, // 1st moment vector (exponential moving average of grad)
    moments2: HashMap<*const (), Tensor<T>>, // 2nd moment vector (exponential moving average of squared grad)
}

impl<T> Adam<T>
where
    T: Float + FromPrimitive + Debug + Zero + One + Copy + 'static,
{
    /// Creates a new Adam optimizer instance.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (e.g., 1e-3).
    /// * `betas` - Coefficients used for computing running averages of gradient and its square (e.g., (0.9, 0.999)).
    /// * `eps` - Term added to the denominator to improve numerical stability (e.g., 1e-8).
    pub fn new(lr: T, betas: (T, T), eps: T) -> Self {
        assert!(lr >= T::zero(), "Learning rate must be non-negative");
        assert!(betas.0 >= T::zero() && betas.0 < T::one(), "Beta1 must be in [0, 1)");
        assert!(betas.1 >= T::zero() && betas.1 < T::one(), "Beta2 must be in [0, 1)");
        assert!(eps >= T::zero(), "Epsilon must be non-negative");

        Adam {
            lr,
            beta1: betas.0,
            beta2: betas.1,
            epsilon: eps,
            moments1: HashMap::new(),
            moments2: HashMap::new(),
            t: 0, // Initialize timestep to 0
        }
    }
}

impl<T> Optimizer<T> for Adam<T>
where
    T: Float + FromPrimitive + Debug + Zero + One 
       + Pow<T, Output = T>    // For sqrt
       + Pow<i32, Output = T>  // For powi
       + AddAssign             // For element-wise ops
       + Sub<Output = T>       // For sqrt & element-wise ops
       + Neg<Output = T>       // For sqrt
       + Mul<Output = T>       // For element-wise ops
       + Add<Output = T>       // For element-wise ops
       + Div<Output = T>       // For element-wise ops
       + IterSum               // For sqrt
       + Default               // For sqrt
       + Copy + 'static,
    // Assume Tensor<T>: Clone exists implicitly
{
    /// Performs a single optimization step.
    ///
    /// # Arguments
    /// * `params` - A mutable slice of Tensors representing the model parameters to be updated.
    fn step(&mut self, params: &mut [&mut Tensor<T>]) {
        self.t += 1;
        let t_usize = self.t as usize;
        // Avoid potential division by zero if t=0 (though unlikely with u64)
        let bc1_pow = if t_usize == 0 { T::one() } else { self.beta1.powi(t_usize as i32) };
        let bc2_pow = if t_usize == 0 { T::one() } else { self.beta2.powi(t_usize as i32) };
        let bias_correction1 = T::one() - bc1_pow;
        let bias_correction2 = T::one() - bc2_pow;
        // Avoid division by zero if bias_correction is zero (can happen if beta=1 or t is huge)
        let bc1 = if bias_correction1.is_zero() { T::epsilon() } else { bias_correction1 };
        let bc2 = if bias_correction2.is_zero() { T::epsilon() } else { bias_correction2 };

        for param in params.iter_mut() {
            // Clone the gradient tensor FIRST to avoid borrow conflicts
            if let Some(grad_tensor) = param.grad().map(|g_ref| g_ref.clone()) { 
                let grad_data = grad_tensor.data(); // Immutable borrow on grad clone
                let key = param.id(); // Use param ID after cloning grad
                let numel = param.numel();

                assert_eq!(grad_data.len(), numel, "Gradient numel mismatch");

                // Get or insert moments (these are separate tensors)
                let m_t = self.moments1.entry(key).or_insert_with(|| Tensor::zeros_like(param));
                let v_t = self.moments2.entry(key).or_insert_with(|| Tensor::zeros_like(param));

                // --- Perform calculations using cloned grad_data and moment data --- 
                // Borrow moment data mutably
                let mut m_t_data = m_t.data_mut();
                let mut v_t_data = v_t.data_mut();

                let one_minus_beta1 = T::one() - self.beta1;
                let one_minus_beta2 = T::one() - self.beta2;

                // Calculate updates in a separate buffer to avoid borrowing param_data yet
                let mut update_values = Vec::with_capacity(numel);

                for i in 0..numel {
                    // Update moments in-place
                    m_t_data[i] = self.beta1 * m_t_data[i] + one_minus_beta1 * grad_data[i];
                    let grad_i_sq = grad_data[i] * grad_data[i];
                    v_t_data[i] = self.beta2 * v_t_data[i] + one_minus_beta2 * grad_i_sq;

                    // Calculate bias-corrected moments
                    let m_hat_i = m_t_data[i] / bc1;
                    let v_hat_i = v_t_data[i] / bc2;

                    // Calculate the update value
                    let v_hat_sqrt_i = v_hat_i.sqrt();
                    let denom_i = v_hat_sqrt_i + self.epsilon;
                    let update_i = self.lr * m_hat_i / denom_i;
                    update_values.push(update_i);
                }
                // Drop borrows on moment data explicitly before borrowing param data
                drop(m_t_data);
                drop(v_t_data);

                // --- NOW, borrow param data mutably and apply updates --- 
                let mut param_data = param.data_mut();
                assert_eq!(param_data.len(), update_values.len(), "Update values length mismatch");
                for i in 0..numel {
                    // Apply update: param = param - update
                    // Requires T: Sub<Output=T> which is covered by Float trait generally
                    // If T doesn't implement Sub, this would need SubAssign or manual subtract
                    param_data[i] = param_data[i] - update_values[i]; 
                }
            }
        }
    }

    /// Clears the gradients of the specified parameters.
    /// This should be called before the backward pass.
    ///
    /// # Arguments
    /// * `params` - A mutable slice of Tensors representing the model parameters whose gradients should be cleared.
    fn zero_grad(&self, params: &mut [&mut Tensor<T>]) where T: Zero {
        for param in params.iter_mut() {
            param.zero_grad();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use neurarust_core::tensor::Tensor;
    use num_traits::{Zero, One, Float, FromPrimitive};
    use std::fmt::Debug;
    use std::ops::{AddAssign, Mul, Neg};
    use std::iter::Sum;

    // Helper to check approximate data equality - Copied from sgd.rs tests
    fn check_data_approx<T>(tensor: &Tensor<T>, expected_data: &[T])
    where
        T: Float + Debug + FromPrimitive,
    {
        let tensor_data = tensor.data();
        assert_eq!(tensor_data.len(), expected_data.len(), "Tensor length mismatch");
        let tol = T::epsilon() * T::from_u32(100).unwrap_or_else(T::one);
        for (a, b) in tensor_data.iter().zip(expected_data.iter()) {
            assert!( (*a - *b).abs() < tol, "Data mismatch: expected {:?}, got {:?}. Diff: {:?}",
                    expected_data, &*tensor_data, (*a - *b).abs());
        }
    }

    // Helper function to generate a gradient on a tensor using backward()
    // Copied from sgd.rs tests
    fn generate_gradient<T>(
        tensor: &Tensor<T>, 
        grad_values: Vec<T>
    )
    where 
        // Added PartialEq to support Tensor::backward bound
        T: Mul<Output = T> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + Neg<Output=T> + PartialEq,
    {
        assert_eq!(tensor.numel(), grad_values.len(), "Tensor numel must match grad_values length");
        tensor.set_requires_grad(true);
        let constant = Tensor::new(grad_values, tensor.shape());
        let mul_result = tensor * &constant;
        let loss = mul_result.sum(); 
        assert!(loss.requires_grad());
        loss.backward(None); 
        assert!(tensor.grad().is_some(), "Gradient was not generated by backward pass");
    }

    // Type alias for tests
    type TestFloat = f64;

    // Helper to create a tensor with *generated* gradients using backward()
    fn create_grad_tensor(data: Vec<TestFloat>, shape: Vec<usize>, grad_data: Vec<TestFloat>) -> Tensor<TestFloat> 
    where
        // Bounds needed by generate_gradient
        TestFloat: Mul<Output = TestFloat> + AddAssign + Copy + Clone + Debug + Default + Zero + One + Sum + 'static + Neg<Output=TestFloat> + PartialEq,
    {
        // Create tensor 
        let t = Tensor::new(data, shape);
        // Use generate_gradient to populate the .grad field via backward()
        generate_gradient(&t, grad_data);
        t
    }

    #[test]
    fn test_adam_zero_grad() {
        type TestFloat = f64;
        let mut p1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        // Créer p2 sans gradient initial
        let mut p2 = Tensor::new(vec![3.0, 4.0], vec![2]); 
        p2.set_requires_grad(false); // p2 n'a pas besoin de gradient
        
        // Générer un gradient uniquement pour p1
        let initial_grad_p1 = vec![0.1, 0.2];
        generate_gradient(&p1, initial_grad_p1.clone());
        
        assert!(p1.grad().is_some(), "p1 should have gradient after generate_gradient");
        check_data_approx(&p1.grad().unwrap(), &initial_grad_p1);
        // Vérifier que p2 n'a pas de gradient
        assert!(p2.grad().is_none(), "p2 should not have gradient initially");

        let optim: Adam<TestFloat> = Adam::new(0.1, (0.9, 0.999), 1e-8);
        let mut params_slice = [&mut p1, &mut p2];
        optim.zero_grad(&mut params_slice);

        assert!(p1.grad().is_some(), "p1 gradient should still exist after zero_grad");
        check_data_approx(&p1.grad().unwrap(), &[0.0, 0.0]); 
        assert!(p2.grad().is_none(), "p2 gradient should remain None after zero_grad");
    }

    #[test]
    fn test_adam_step_basic() {
        let mut p1 = create_grad_tensor(vec![1.0, 2.0], vec![2], vec![0.1, 0.2]);
        let mut p2 = create_grad_tensor(vec![3.0, 4.0], vec![2], vec![0.3, 0.4]);
        let mut p3 = Tensor::new(vec![5.0], vec![1]);

        let lr = 0.001;
        let betas = (0.9, 0.999);
        let eps = 1e-8;
        let mut optimizer: Adam<TestFloat> = Adam::new(lr, betas, eps);

        // --- Step 1 --- 
        let grad1_s1 = vec![0.1, 0.2];
        let grad2_s1 = vec![0.3, 0.4];
        generate_gradient(&p1, grad1_s1.clone());
        generate_gradient(&p2, grad2_s1.clone());
        
        {
            let mut params_slice: Vec<&mut Tensor<TestFloat>> = vec![&mut p1, &mut p2, &mut p3];
            optimizer.step(&mut params_slice);
        } 
        
        // Calculate expected values after step 1 MANUALLY
        let t1 = 1_i32;
        let beta1 = betas.0 as TestFloat;
        let beta2 = betas.1 as TestFloat;
        let bc1_s1 = 1.0 - beta1.powi(t1);
        let bc2_s1 = 1.0 - beta2.powi(t1);

        // Param 1 update
        let m1_1 = (1.0 - beta1) * grad1_s1[0]; // m = (1-b1)*g
        let v1_1 = (1.0 - beta2) * grad1_s1[0].powi(2); // v = (1-b2)*g^2
        let m1_hat_1 = m1_1 / bc1_s1;
        let v1_hat_1 = v1_1 / bc2_s1;
        let update1_1 = lr * m1_hat_1 / (v1_hat_1.sqrt() + eps);

        let m1_2 = (1.0 - beta1) * grad1_s1[1];
        let v1_2 = (1.0 - beta2) * grad1_s1[1].powi(2);
        let m1_hat_2 = m1_2 / bc1_s1;
        let v1_hat_2 = v1_2 / bc2_s1;
        let update1_2 = lr * m1_hat_2 / (v1_hat_2.sqrt() + eps);
        let expected_p1_s1 = vec![1.0 - update1_1, 2.0 - update1_2];

        // Param 2 update
        let m2_1 = (1.0 - beta1) * grad2_s1[0];
        let v2_1 = (1.0 - beta2) * grad2_s1[0].powi(2);
        let m2_hat_1 = m2_1 / bc1_s1;
        let v2_hat_1 = v2_1 / bc2_s1;
        let update2_1 = lr * m2_hat_1 / (v2_hat_1.sqrt() + eps);

        let m2_2 = (1.0 - beta1) * grad2_s1[1];
        let v2_2 = (1.0 - beta2) * grad2_s1[1].powi(2);
        let m2_hat_2 = m2_2 / bc1_s1;
        let v2_hat_2 = v2_2 / bc2_s1;
        let update2_2 = lr * m2_hat_2 / (v2_hat_2.sqrt() + eps);
        let expected_p2_s1 = vec![3.0 - update2_1, 4.0 - update2_2];

        // Check results of step 1
        check_data_approx(&p1, &expected_p1_s1);
        check_data_approx(&p2, &expected_p2_s1);

        // --- Step 2 --- 
        // Need previous moments m1_1, v1_1 etc. for calculation
        let m1_s1 = vec![m1_1, m1_2];
        let v1_s1 = vec![v1_1, v1_2];
        let m2_s1 = vec![m2_1, m2_2];
        let v2_s1 = vec![v2_1, v2_2];
        
        // Generate new gradients
        // Need to zero previous grad first before generating new one
        optimizer.zero_grad(&mut [&mut p1, &mut p2, &mut p3]);
        let grad1_s2 = vec![0.5, 0.6];
        let grad2_s2 = vec![0.7, 0.8];
        generate_gradient(&p1, grad1_s2.clone());
        generate_gradient(&p2, grad2_s2.clone());

        {
            let mut params_slice: Vec<&mut Tensor<TestFloat>> = vec![&mut p1, &mut p2, &mut p3];
            optimizer.step(&mut params_slice);
        }
        
        // Calculate expected values after step 2
        let t2 = 2_i32;
        let bc1_s2 = 1.0 - beta1.powi(t2);
        let bc2_s2 = 1.0 - beta2.powi(t2);

        // Param 1 update (step 2)
        let m1_1_s2 = beta1 * m1_s1[0] + (1.0 - beta1) * grad1_s2[0];
        let v1_1_s2 = beta2 * v1_s1[0] + (1.0 - beta2) * grad1_s2[0].powi(2);
        let m1_hat_1_s2 = m1_1_s2 / bc1_s2;
        let v1_hat_1_s2 = v1_1_s2 / bc2_s2;
        let update1_1_s2 = lr * m1_hat_1_s2 / (v1_hat_1_s2.sqrt() + eps);

        let m1_2_s2 = beta1 * m1_s1[1] + (1.0 - beta1) * grad1_s2[1];
        let v1_2_s2 = beta2 * v1_s1[1] + (1.0 - beta2) * grad1_s2[1].powi(2);
        let m1_hat_2_s2 = m1_2_s2 / bc1_s2;
        let v1_hat_2_s2 = v1_2_s2 / bc2_s2;
        let update1_2_s2 = lr * m1_hat_2_s2 / (v1_hat_2_s2.sqrt() + eps);
        let expected_p1_s2 = vec![expected_p1_s1[0] - update1_1_s2, expected_p1_s1[1] - update1_2_s2];

        // Param 2 update (step 2)
        let m2_1_s2 = beta1 * m2_s1[0] + (1.0 - beta1) * grad2_s2[0];
        let v2_1_s2 = beta2 * v2_s1[0] + (1.0 - beta2) * grad2_s2[0].powi(2);
        let m2_hat_1_s2 = m2_1_s2 / bc1_s2;
        let v2_hat_1_s2 = v2_1_s2 / bc2_s2;
        let update2_1_s2 = lr * m2_hat_1_s2 / (v2_hat_1_s2.sqrt() + eps);

        let m2_2_s2 = beta1 * m2_s1[1] + (1.0 - beta1) * grad2_s2[1];
        let v2_2_s2 = beta2 * v2_s1[1] + (1.0 - beta2) * grad2_s2[1].powi(2);
        let m2_hat_2_s2 = m2_2_s2 / bc1_s2;
        let v2_hat_2_s2 = v2_2_s2 / bc2_s2;
        let update2_2_s2 = lr * m2_hat_2_s2 / (v2_hat_2_s2.sqrt() + eps);
        let expected_p2_s2 = vec![expected_p2_s1[0] - update2_1_s2, expected_p2_s1[1] - update2_2_s2];

        // Check results of step 2
        check_data_approx(&p1, &expected_p1_s2);
        check_data_approx(&p2, &expected_p2_s2);
    }
}