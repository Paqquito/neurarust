#[cfg(test)]
mod tests {
    use crate::nn::init::{zeros_, ones_, kaiming_uniform_, kaiming_normal_, xavier_uniform_};
    use crate::tensor::{zeros, ones, zeros_f64, ones_f64};
    use crate::DType;

    #[test]
    fn test_zeros_() {
        let mut t = ones(&[2, 3]).unwrap();
        zeros_(&mut t).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        let data = t.get_f32_data().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));

        let mut t_f64 = ones_f64(&[1, 5]).unwrap();
        zeros_(&mut t_f64).unwrap();
        assert_eq!(t_f64.dtype(), DType::F64);
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_() {
        let mut t = zeros(&[4]).unwrap();
        ones_(&mut t).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        let data = t.get_f32_data().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));

        let mut t_f64 = zeros_f64(&[2, 1, 2]).unwrap();
        ones_(&mut t_f64).unwrap();
        assert_eq!(t_f64.dtype(), DType::F64);
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_kaiming_uniform_() {
        // Standard case: Linear layer weights
        let mut t_f32 = zeros(&[10, 5]).unwrap();
        kaiming_uniform_(&mut t_f32).unwrap();
        let fan_in = 5;
        let bound = (6.0 / fan_in as f64).sqrt() as f32;
        let data_f32 = t_f32.get_f32_data().unwrap();
        assert!(data_f32.iter().all(|&x| x >= -bound && x <= bound));
        // Check that not all values are zero (highly unlikely for uniform)
        assert!(data_f32.iter().any(|&x| x != 0.0));

        // F64 case
        let mut t_f64 = zeros_f64(&[4, 8]).unwrap();
        kaiming_uniform_(&mut t_f64).unwrap();
        let fan_in_f64 = 8;
        let bound_f64 = (6.0 / fan_in_f64 as f64).sqrt();
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().all(|&x| x >= -bound_f64 && x <= bound_f64));
        assert!(data_f64.iter().any(|&x| x != 0.0));

        // Error case: 1D tensor
        let mut t_1d = zeros(&[10]).unwrap();
        let result = kaiming_uniform_(&mut t_1d);
        assert!(result.is_err());
    }

    #[test]
    fn test_kaiming_normal_() {
        // F32 case
        let mut t_f32 = zeros(&[200, 100]).unwrap();
        kaiming_normal_(&mut t_f32).unwrap();
        let fan_in = 100;
        let expected_std = (2.0 / fan_in as f64).sqrt() as f32;
        let data_f32 = t_f32.get_f32_data().unwrap();

        // Basic check: not all zeros
        assert!(data_f32.iter().any(|&x| x != 0.0));

        // Check statistics (mean should be close to 0, stddev close to expected_std)
        // These checks can be flaky, especially for smaller tensors.
        let n = data_f32.len() as f32;
        let mean = data_f32.iter().sum::<f32>() / n;
        let variance = data_f32.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let stddev = variance.sqrt();

        println!("Kaiming Normal F32: Mean: {:.4}, StdDev: {:.4}, Expected StdDev: {:.4}", mean, stddev, expected_std);
        assert!(mean.abs() < 0.1, "Mean deviates too much from 0"); // Loose check for mean
        assert!((stddev - expected_std).abs() / expected_std < 0.2, "StdDev deviates too much from expected"); // Loose check for stddev (20% tolerance)

        // F64 case
        let mut t_f64 = zeros_f64(&[50, 200]).unwrap();
        kaiming_normal_(&mut t_f64).unwrap();
        let fan_in_f64 = 200;
        let expected_std_f64 = (2.0 / fan_in_f64 as f64).sqrt();
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().any(|&x| x != 0.0));

        let n_f64 = data_f64.len() as f64;
        let mean_f64 = data_f64.iter().sum::<f64>() / n_f64;
        let variance_f64 = data_f64.iter().map(|&x| (x - mean_f64).powi(2)).sum::<f64>() / n_f64;
        let stddev_f64 = variance_f64.sqrt();

        println!("Kaiming Normal F64: Mean: {:.4}, StdDev: {:.4}, Expected StdDev: {:.4}", mean_f64, stddev_f64, expected_std_f64);
        assert!(mean_f64.abs() < 0.1, "Mean (F64) deviates too much from 0"); 
        assert!((stddev_f64 - expected_std_f64).abs() / expected_std_f64 < 0.2, "StdDev (F64) deviates too much from expected");

        // Error case: 1D tensor
        let mut t_1d = zeros(&[10]).unwrap();
        let result = kaiming_normal_(&mut t_1d);
        assert!(result.is_err());
    }

    #[test]
    fn test_xavier_uniform_() {
        // F32 case
        let mut t_f32 = zeros(&[50, 20]).unwrap();
        xavier_uniform_(&mut t_f32).unwrap();
        let fan_in = 20;
        let fan_out = 50;
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;
        let data_f32 = t_f32.get_f32_data().unwrap();
        assert!(data_f32.iter().all(|&x| x >= -bound && x <= bound));
        assert!(data_f32.iter().any(|&x| x != 0.0)); // Check not all zeros

        // F64 case
        let mut t_f64 = zeros_f64(&[10, 30]).unwrap();
        xavier_uniform_(&mut t_f64).unwrap();
        let fan_in_f64 = 30;
        let fan_out_f64 = 10;
        let bound_f64 = (6.0 / (fan_in_f64 + fan_out_f64) as f64).sqrt();
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().all(|&x| x >= -bound_f64 && x <= bound_f64));
        assert!(data_f64.iter().any(|&x| x != 0.0));

        // Error case: 1D tensor
        let mut t_1d = zeros(&[10]).unwrap();
        let result = xavier_uniform_(&mut t_1d);
        assert!(result.is_err());
    }

    #[test]
    fn test_fill_inplace_autograd_error() {
        let mut t = zeros(&[2]).unwrap();
        let _ = t.set_requires_grad(true);
        let result = zeros_(&mut t);
        assert!(result.is_err());
        // Optionally check the specific error type
        // match result.unwrap_err() {
        //     NeuraRustError::InplaceModificationError { .. } => (), 
        //     _ => panic!("Expected InplaceModificationError"),
        // }
    }
    
    // TODO: Add tests for kaiming, xavier etc. when implemented
} 