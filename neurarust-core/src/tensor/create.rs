// src/tensor/create.rs

use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;
use rand::Rng;
use rand_distr::StandardNormal;
use crate::device::StorageDevice;
 // Import necessary types

/// Creates a new CPU tensor filled with zeros with the specified shape.
///
/// The tensor will have `DType::F32`.
///
/// # Arguments
/// * `shape`: A slice defining the dimensions of the tensor.
///
/// # Returns
/// A `Result` containing the new tensor or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::zeros;
///
/// let t = zeros(&[2, 3]).unwrap();
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.get_f32_data().unwrap(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// ```
pub fn zeros(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f32> = vec![0.0; numel]; // Create f32 data
    Tensor::new(data_vec, shape.to_vec())
}

/// Creates a new CPU tensor filled with zeros with the specified shape and `DType::F64`.
///
/// # Arguments
/// * `shape`: A slice defining the dimensions of the tensor.
///
/// # Returns
/// A `Result` containing the new F64 tensor or a `NeuraRustError`.
pub fn zeros_f64(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f64> = vec![0.0; numel]; // Create f64 data
    Tensor::new_f64(data_vec, shape.to_vec())
}

/// Creates a new CPU tensor filled with ones with the specified shape.
///
/// The tensor will have `DType::F32`.
///
/// # Arguments
/// * `shape`: A slice defining the dimensions of the tensor.
///
/// # Returns
/// A `Result` containing the new tensor or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::ones;
///
/// let t = ones(&[1, 4]).unwrap();
/// assert_eq!(t.shape(), &[1, 4]);
/// assert_eq!(t.get_f32_data().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
/// ```
pub fn ones(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f32> = vec![1.0; numel]; // Create f32 data
    Tensor::new(data_vec, shape.to_vec())
}

/// Creates a new CPU tensor filled with ones with the specified shape and `DType::F64`.
///
/// # Arguments
/// * `shape`: A slice defining the dimensions of the tensor.
///
/// # Returns
/// A `Result` containing the new F64 tensor or a `NeuraRustError`.
pub fn ones_f64(shape: &[usize]) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f64> = vec![1.0; numel]; // Create f64 data
    Tensor::new_f64(data_vec, shape.to_vec())
}

/// Creates a new CPU tensor filled with a specific value with the specified shape.
///
/// The tensor will have `DType::F32`.
///
/// # Arguments
/// * `shape`: A slice defining the dimensions of the tensor.
/// * `value`: The `f32` value to fill the tensor with.
///
/// # Returns
/// A `Result` containing the new tensor or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::full;
///
/// let t = full(&[2, 2], 3.14f32).unwrap();
/// assert_eq!(t.shape(), &[2, 2]);
/// assert_eq!(t.get_f32_data().unwrap(), vec![3.14, 3.14, 3.14, 3.14]);
/// ```
pub fn full(shape: &[usize], value: f32) -> Result<Tensor, NeuraRustError> { // value is now f32
    let numel = shape.iter().product();
    let data_vec: Vec<f32> = vec![value; numel]; // Create f32 data
    Tensor::new(data_vec, shape.to_vec())
}

/// Creates a new CPU tensor filled with a specific value with the specified shape and `DType::F64`.
///
/// # Arguments
/// * `shape`: A slice defining the dimensions of the tensor.
/// * `value`: The `f64` value to fill the tensor with.
///
/// # Returns
/// A `Result` containing the new F64 tensor or a `NeuraRustError`.
pub fn full_f64(shape: &[usize], value: f64) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let data_vec: Vec<f64> = vec![value; numel]; // Create f64 data
    Tensor::new_f64(data_vec, shape.to_vec())
}

/// Creates a new CPU tensor with `DType::F32` from a `Vec<f32>` and shape.
///
/// This function takes ownership of the data vector.
///
/// # Arguments
/// * `data_vec`: The `Vec<f32>` containing the tensor data in row-major order.
/// * `shape`: The desired shape of the tensor as a `Vec<usize>`.
///
/// # Errors
/// Returns `NeuraRustError::TensorCreationError` if the number of elements in `data_vec`
/// does not match the total number of elements specified by `shape`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::from_vec_f32;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let shape = vec![2, 3];
/// let t = from_vec_f32(data, shape).unwrap();
/// assert_eq!(t.shape(), &[2, 3]);
/// ```
pub fn from_vec_f32(data_vec: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    Tensor::new(data_vec, shape)
}

/// Creates a new CPU tensor with `DType::F64` from a `Vec<f64>` and shape.
///
/// This function takes ownership of the data vector.
///
/// # Arguments
/// * `data_vec`: The `Vec<f64>` containing the tensor data in row-major order.
/// * `shape`: The desired shape of the tensor as a `Vec<usize>`.
///
/// # Errors
/// Returns `NeuraRustError::TensorCreationError` if the number of elements in `data_vec`
/// does not match the total number of elements specified by `shape`.
pub fn from_vec_f64(data_vec: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    Tensor::new_f64(data_vec, shape)
}

/// Creates a new CPU tensor filled with zeros, having the same shape, device, and `DType` as the input tensor.
///
/// # Arguments
/// * `tensor`: A reference to the tensor whose properties (shape, dtype, device) should be matched.
///
/// # Returns
/// A `Result` containing the new tensor of zeros or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::{from_vec_f64, zeros_like};
/// use neurarust_core::DType;
///
/// let t1 = from_vec_f64(vec![1.0, 2.0], vec![2]).unwrap();
/// let z1 = zeros_like(&t1).unwrap();
/// assert_eq!(z1.shape(), t1.shape());
/// assert_eq!(z1.dtype(), DType::F64);
/// assert_eq!(z1.get_f64_data().unwrap(), vec![0.0, 0.0]);
/// ```
pub fn zeros_like(tensor: &Tensor) -> Result<Tensor, NeuraRustError> {
    // TODO: Later, use tensor.device() to create on the same device.
    let shape = tensor.shape();
    match tensor.dtype() {
        DType::F32 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f32> = vec![0.0; numel];
            Tensor::new(data_vec, shape)
        }
        DType::F64 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f64> = vec![0.0; numel];
            Tensor::new_f64(data_vec, shape)
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    }
}

/// Creates a new CPU tensor filled with ones, having the same shape, device, and `DType` as the input tensor.
///
/// # Arguments
/// * `tensor`: A reference to the tensor whose properties (shape, dtype, device) should be matched.
///
/// # Returns
/// A `Result` containing the new tensor of ones or a `NeuraRustError`.
pub fn ones_like(tensor: &Tensor) -> Result<Tensor, NeuraRustError> {
    // TODO: Later, use tensor.device() to create on the same device.
    let shape = tensor.shape();
    match tensor.dtype() {
        DType::F32 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f32> = vec![1.0; numel];
            Tensor::new(data_vec, shape)
        }
        DType::F64 => {
            let numel = shape.iter().product();
            let data_vec: Vec<f64> = vec![1.0; numel];
            Tensor::new_f64(data_vec, shape)
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    }
}

// --- Keep other creation functions like arange, linspace, eye, rand, randn --- 
// They might need adaptation later, especially regarding DType and Device.
// For now, let's assume they primarily work with f32 or can be adapted easily later.

/// Creates a 1-D CPU tensor containing a range of values with `DType::F32`.
///
/// Generates values from `start` up to (but not including) `end` with a step size of `step`.
/// Similar to Python's `numpy.arange`.
///
/// # Arguments
/// * `start`: The starting value of the sequence.
/// * `end`: The end value of the sequence (exclusive).
/// * `step`: The step size between values.
///
/// # Errors
/// Returns `NeuraRustError::UnsupportedOperation` if the step size is zero or has the wrong sign
/// (e.g., positive step for `end < start`).
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::arange;
///
/// let t = arange(1.0, 5.0, 1.5).unwrap(); // 1.0, 2.5, 4.0
/// assert_eq!(t.get_f32_data().unwrap(), vec![1.0, 2.5, 4.0]);
/// ```
pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor, NeuraRustError> {
    if (end > start && step <= 0.0) || (end < start && step >= 0.0) || step == 0.0 {
        return Err(NeuraRustError::UnsupportedOperation(
            format!("Invalid step {} for arange({}, {})", step, start, end)
        ));
    }
    let numel = ((end - start) / step).ceil() as usize;
    let data_vec: Vec<f32> = (0..numel).map(|i| start + i as f32 * step).collect();
    Tensor::new(data_vec, vec![numel])
}

/// Creates a 1-D CPU tensor containing evenly spaced values over a specified interval with `DType::F32`.
///
/// Generates `steps` values starting from `start` and ending at `end` (inclusive).
/// Similar to Python's `numpy.linspace`.
///
/// # Arguments
/// * `start`: The starting value of the sequence.
/// * `end`: The ending value of the sequence.
/// * `steps`: The total number of steps (samples) to generate.
///
/// # Errors
/// Returns `NeuraRustError::UnsupportedOperation` if `steps` is less than 2.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::linspace;
///
/// let t = linspace(0.0, 10.0, 5).unwrap(); // 0.0, 2.5, 5.0, 7.5, 10.0
/// assert_eq!(t.get_f32_data().unwrap(), vec![0.0, 2.5, 5.0, 7.5, 10.0]);
/// ```
pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Tensor, NeuraRustError> {
    if steps < 2 {
        return Err(NeuraRustError::UnsupportedOperation(
            "Linspace requires at least 2 steps".to_string()
        ));
    }
    let mut data_vec = Vec::with_capacity(steps);
    let step_size = (end - start) / (steps - 1) as f32;
    for i in 0..steps {
        data_vec.push(start + i as f32 * step_size);
    }
    Tensor::new(data_vec, vec![steps])
}

/// Creates a 2-D CPU identity matrix (tensor) with `DType::F32`.
///
/// Generates an `n x n` tensor with ones on the diagonal and zeros elsewhere.
///
/// # Arguments
/// * `n`: The size of the square matrix (number of rows and columns).
///
/// # Returns
/// A `Result` containing the identity tensor or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::eye;
///
/// let t = eye(3).unwrap();
/// assert_eq!(t.shape(), &[3, 3]);
/// assert_eq!(t.get_f32_data().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
/// ```
pub fn eye(n: usize) -> Result<Tensor, NeuraRustError> {
    let mut data_vec = vec![0.0f32; n * n];
    for i in 0..n {
        data_vec[i * n + i] = 1.0;
    }
    Tensor::new(data_vec, vec![n, n])
}

/// Creates a tensor with the given shape, filled with random numbers from a uniform
/// distribution between 0.0 (inclusive) and 1.0 (exclusive).
///
/// The tensor will have `DType::F32` and be located on the `CPU`.
///
/// # Arguments
/// * `shape`: A vector representing the desired shape of the tensor.
///
/// # Returns
/// A `Result` containing the new `Tensor` or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::rand;
///
/// let tensor = rand(vec![2, 3]).unwrap();
/// assert_eq!(tensor.shape(), vec![2, 3]);
/// // Values should be between 0.0 and 1.0
/// let data = tensor.get_f32_data().unwrap();
/// assert!(data.iter().all(|&x| x >= 0.0 && x < 1.0));
/// ```
pub fn rand(shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..numel).map(|_| rng.gen::<f32>()).collect();
    Tensor::new(data, shape)
}

/// Creates a tensor with the given shape, filled with random numbers from a standard
/// normal distribution (mean 0, standard deviation 1).
///
/// The tensor will have `DType::F32` and be located on the `CPU`.
///
/// # Arguments
/// * `shape`: A vector representing the desired shape of the tensor.
///
/// # Returns
/// A `Result` containing the new `Tensor` or a `NeuraRustError`.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::randn;
///
/// let tensor = randn(vec![2, 3]).unwrap();
/// assert_eq!(tensor.shape(), vec![2, 3]);
/// // Values should be normally distributed (difficult to test precisely)
/// ```
pub fn randn(shape: Vec<usize>) -> Result<Tensor, NeuraRustError> {
    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();
    // Use StandardNormal distribution from rand_distr
    let data: Vec<f32> = (0..numel)
        .map(|_| rng.sample(StandardNormal))
        .collect();
    Tensor::new(data, shape)
}

/// Creates a CPU tensor with the given shape, filled with random integers
/// from `low` (inclusive) to `high` (exclusive).
///
/// Currently, this function supports `DType::F32` and `DType::F64`,
/// where the generated integers are cast to floating-point numbers.
/// True integer DType support for tensors is planned for a later phase.
/// The `device` parameter is currently ignored; tensors are always created on the CPU.
///
/// # Arguments
/// * `low`: The lower bound of the random integers (inclusive).
/// * `high`: The upper bound of the random integers (exclusive).
/// * `shape`: A vector defining the dimensions of the tensor.
/// * `dtype`: The data type of the tensor (currently `DType::F32` or `DType::F64`).
/// * `device`: The device to create the tensor on (currently ignored, always CPU).
///
/// # Returns
/// A `Result` containing the new tensor or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError::ArithmeticError` if `low >= high`.
/// Returns `NeuraRustError::UnsupportedOperation` if a DType other than F32 or F64 is specified.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::randint;
/// use neurarust_core::types::DType;
/// use neurarust_core::device::StorageDevice;
///
/// let t_f32 = randint(0, 10, vec![2, 2], DType::F32, StorageDevice::CPU).unwrap();
/// assert_eq!(t_f32.shape(), &[2, 2]);
/// assert_eq!(t_f32.dtype(), DType::F32);
/// // Values should be between 0.0 and 9.0 inclusive
/// for &val in t_f32.get_f32_data().unwrap().iter() {
///     assert!(val >= 0.0 && val < 10.0);
///     assert_eq!(val, val.trunc()); // Check if it's an integer value
/// }
///
/// let t_f64 = randint(-5, 5, vec![3], DType::F64, StorageDevice::CPU).unwrap();
/// assert_eq!(t_f64.shape(), &[3]);
/// assert_eq!(t_f64.dtype(), DType::F64);
/// ```
pub fn randint(
    low: i64,
    high: i64,
    shape: Vec<usize>,
    dtype: DType,
    _device: StorageDevice, // Device is ignored for now
) -> Result<Tensor, NeuraRustError> {
    if low >= high {
        return Err(NeuraRustError::ArithmeticError(format!(
            "low bound {} must be less than high bound {} for randint",
            low, high
        )));
    }

    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();

    match dtype {
        DType::F32 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(rng.gen_range(low..high) as f32);
            }
            Tensor::new(data_vec, shape)
        }
        DType::F64 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(rng.gen_range(low..high) as f64);
            }
            Tensor::new_f64(data_vec, shape)
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    }
}

/// Creates a CPU tensor with the given shape, where each element is 0 or 1,
/// drawn from a Bernoulli distribution with the given probability `p` of being 1.
///
/// Currently, this function supports `DType::F32` and `DType::F64`,
/// returning `0.0` or `1.0`.
/// True Boolean DType support for tensors is planned for a later phase.
/// The `device` parameter is currently ignored; tensors are always created on the CPU.
///
/// # Arguments
/// * `p`: The probability of an element being 1. Must be between 0.0 and 1.0 inclusive.
/// * `shape`: A vector defining the dimensions of the tensor.
/// * `dtype`: The data type of the tensor (currently `DType::F32` or `DType::F64`).
/// * `device`: The device to create the tensor on (currently ignored, always CPU).
///
/// # Returns
/// A `Result` containing the new tensor or a `NeuraRustError`.
///
/// # Errors
/// Returns `NeuraRustError::ArithmeticError` if `p` is not between 0.0 and 1.0.
/// Returns `NeuraRustError::UnsupportedOperation` if a DType other than F32 or F64 is specified.
///
/// # Example
/// ```
/// use neurarust_core::tensor::create::bernoulli_scalar;
/// use neurarust_core::types::DType;
/// use neurarust_core::device::StorageDevice;
///
/// let t_f32 = bernoulli_scalar(0.7, vec![2, 2], DType::F32, StorageDevice::CPU).unwrap();
/// assert_eq!(t_f32.shape(), &[2, 2]);
/// assert_eq!(t_f32.dtype(), DType::F32);
/// // Values should be 0.0 or 1.0
/// for &val in t_f32.get_f32_data().unwrap().iter() {
///     assert!(val == 0.0 || val == 1.0, "Value not 0.0 or 1.0: {}", val);
/// }
/// ```
pub fn bernoulli_scalar(
    p: f64,
    shape: Vec<usize>,
    dtype: DType,
    _device: StorageDevice, // Device is ignored for now
) -> Result<Tensor, NeuraRustError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(NeuraRustError::ArithmeticError(format!(
            "probability p ({}) must be between 0.0 and 1.0 for bernoulli_scalar",
            p
        )));
    }

    let numel = shape.iter().product();
    let mut rng = rand::thread_rng();

    match dtype {
        DType::F32 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(if rng.gen_bool(p) { 1.0f32 } else { 0.0f32 });
            }
            Tensor::new(data_vec, shape)
        }
        DType::F64 => {
            let mut data_vec = Vec::with_capacity(numel);
            for _ in 0..numel {
                data_vec.push(if rng.gen_bool(p) { 1.0f64 } else { 0.0f64 });
            }
            Tensor::new_f64(data_vec, shape)
        }
        DType::I32 | DType::I64 | DType::Bool => todo!(),
    }
}

// Link the external tests file
#[cfg(test)]
#[path = "create_test.rs"] mod tests;