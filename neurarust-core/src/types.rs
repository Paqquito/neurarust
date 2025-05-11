/// Defines the possible data types for Tensor elements.
///
/// This enum allows the framework to handle tensors with different
/// numerical types dynamically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating-point type.
    F32,
    /// 64-bit floating-point type.
    F64,
    /// 32-bit integer type.
    I32,
    /// 64-bit integer type.
    I64,
    /// Boolean type (true/false values).
    Bool,
    // TODO: Add other types like U8 etc. here later.
}

// Optional: Add helper methods later if needed, e.g.,
// impl DType {
//     pub fn size_of(&self) -> usize {
//         match self {
//             DType::F32 => std::mem::size_of::<f32>(),
//             // ... other types
//         }
//     }
// } 