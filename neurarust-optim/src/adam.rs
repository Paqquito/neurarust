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
        let bias_correction1 = T::one() - self.beta1.powi(t_usize as i32);
        let bias_correction2 = T::one() - self.beta2.powi(t_usize as i32);
        let bc1 = if bias_correction1.is_zero() { T::epsilon() } else { bias_correction1 };
        let bc2 = if bias_correction2.is_zero() { T::epsilon() } else { bias_correction2 };

        for param in params.iter_mut() {
            if let Some(grad_ref) = param.grad() {
                let grad = grad_ref.clone();
                let key = param.id();

                let numel = param.numel();
                let grad_data = grad.data();

                let m_t = self.moments1.entry(key).or_insert_with(|| Tensor::zeros_like(param));
                let v_t = self.moments2.entry(key).or_insert_with(|| Tensor::zeros_like(param));

                let mut m_t_data = m_t.data_mut();
                let mut v_t_data = v_t.data_mut();
                let mut param_data = param.data_mut();

                assert_eq!(m_t_data.len(), numel);
                assert_eq!(v_t_data.len(), numel);
                assert_eq!(param_data.len(), numel);
                assert_eq!(grad_data.len(), numel);

                let one_minus_beta1 = T::one() - self.beta1;
                let one_minus_beta2 = T::one() - self.beta2;

                for i in 0..numel {
                    m_t_data[i] = self.beta1 * m_t_data[i] + one_minus_beta1 * grad_data[i];
                    let grad_i_sq = grad_data[i] * grad_data[i];
                    v_t_data[i] = self.beta2 * v_t_data[i] + one_minus_beta2 * grad_i_sq;

                    let m_hat_i = m_t_data[i] / bc1;
                    let v_hat_i = v_t_data[i] / bc2;

                    let v_hat_sqrt_i = v_hat_i.sqrt();
                    let denom_i = v_hat_sqrt_i + self.epsilon;
                    let update_i = self.lr * m_hat_i / denom_i;

                    param_data[i] = param_data[i] - update_i;
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
    use crate::Optimizer;

    fn check_data_approx<T>(tensor: &Tensor<T>, expected_data: &[T])
    where
        T: Float + Debug + FromPrimitive,
    {
        let tensor_data = tensor.data();
        assert_eq!(tensor_data.len(), expected_data.len());
        let tol = T::epsilon() * T::from_u32(100).unwrap_or_else(T::one);
        for (a, b) in tensor_data.iter().zip(expected_data.iter()) {
            assert!( (*a - *b).abs() < tol, "Data mismatch: expected {:?} ({:?}), got {:?} ({:?}). Diff: {:?}",
                    expected_data, b, tensor_data.as_ref(), a, (*a - *b).abs());
        }
    }

    // --- Tests need rework to set gradients without private access --- 
    // TODO: Rework these tests after deciding on a public gradient setting method.

    /*
    #[test]
    fn test_adam_step_basic() {
        type TestFloat = f32;

        let param1_data = vec![1.0 as TestFloat, 2.0];
        let param2_data = vec![3.0 as TestFloat, 4.0];
        let mut param1 = Tensor::new(param1_data.clone(), vec![2]);
        let mut param2 = Tensor::new(param2_data.clone(), vec![2]);

        let lr = 0.001 as TestFloat;
        let betas = (0.9 as TestFloat, 0.999 as TestFloat);
        let eps = 1e-8 as TestFloat;
        let mut optimizer: Adam<TestFloat> = Adam::new(lr, betas, eps);

        // Cannot set gradient easily for now
        // set_test_gradient(&param1, vec![0.1 as TestFloat, 0.2]);
        // set_test_gradient(&param2, vec![0.3 as TestFloat, 0.4]);

        // Placeholder: Check step runs without grads
        let initial_p1_data = param1.data().to_vec();
        let initial_p2_data = param2.data().to_vec();
        {
            let mut params_slice: Vec<&mut Tensor<TestFloat>> = vec![&mut param1, &mut param2];
            optimizer.step(&mut params_slice);
        } 

        check_data_approx(&param1, &initial_p1_data);
        check_data_approx(&param2, &initial_p2_data);

        // --- Test Step 2 - Invalid without gradients ---
        // set_test_gradient(&param1, vec![0.5 as TestFloat, 0.5]);
        // set_test_gradient(&param2, vec![0.6 as TestFloat, 0.6]);
        // {
        //     let mut params_slice: Vec<&mut Tensor<TestFloat>> = vec![&mut param1, &mut param2];
        //     optimizer.step(&mut params_slice);
        // }
    }

    #[test]
    fn test_adam_zero_grad() {
        type TestFloat = f32;
        let mut p1 = Tensor::new(vec![1.0 as TestFloat, 2.0], vec![2]);
        let mut p2 = Tensor::new(vec![3.0 as TestFloat, 4.0], vec![2]);

        // Cannot set gradient easily
        // set_test_gradient(&p1, vec![0.1, 0.2]);
        p1.zero_grad(); // Ensure starts clean
        p2.zero_grad();

        let optim: Adam<TestFloat> = Adam::new(0.001, (0.9, 0.999), 1e-8);

        assert!(p1.grad().is_none());
        assert!(p2.grad().is_none());

        let mut params_slice = [&mut p1, &mut p2];
        optim.zero_grad(&mut params_slice);

        assert!(p1.grad().is_none());
        assert!(p2.grad().is_none());
    }
    */
}