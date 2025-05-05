#[cfg(test)]
mod tests {
    use crate::nn::init::{zeros_, ones_};
    use crate::tensor::Tensor;
    use crate::DType;

    #[test]
    fn test_zeros_() {
        let mut t = Tensor::ones(&[2, 3]).unwrap(); // Start with ones
        zeros_(&mut t).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        let data = t.get_f32_data().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));

        let mut t_f64 = Tensor::ones_f64(&[1, 5]).unwrap();
        zeros_(&mut t_f64).unwrap();
        assert_eq!(t_f64.dtype(), DType::F64);
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_() {
        let mut t = Tensor::zeros(&[4]).unwrap(); // Start with zeros
        ones_(&mut t).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        let data = t.get_f32_data().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));

        let mut t_f64 = Tensor::zeros_f64(&[2, 1, 2]).unwrap();
        ones_(&mut t_f64).unwrap();
        assert_eq!(t_f64.dtype(), DType::F64);
        let data_f64 = t_f64.get_f64_data().unwrap();
        assert!(data_f64.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_fill_inplace_autograd_error() {
        let mut t = Tensor::zeros(&[2]).unwrap();
        t.set_requires_grad(true);
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