use std::sync::Arc;
use crate::error::NeuraRustError;

// --- Helper function for physical offset calculation ---
#[inline]
fn calculate_physical_offset(coords: &[usize], strides: &[usize]) -> Result<usize, NeuraRustError> {
    if coords.len() != strides.len() {
         return Err(NeuraRustError::ShapeMismatch {
            expected: "Matching coords and strides length".to_string(),
            actual: format!("Coords len: {}, Strides len: {}", coords.len(), strides.len()),
            operation: "calculate_physical_offset".to_string(),
        });
    }
    let mut offset = 0;
    for i in 0..coords.len() {
        offset += coords[i] * strides[i];
    }
    Ok(offset)
}

// NdArrayBroadcastingIter for f32
#[derive(Debug)] // Add Debug derive
pub struct NdArrayBroadcastingIter<'a> { // Make pub
    buffer: &'a Arc<Vec<f32>>,
    original_shape: &'a [usize],
    original_strides: &'a [usize],
    original_offset: usize,
    target_shape: &'a [usize],
    current_index: usize,
    total_elements: usize,
}

impl<'a> NdArrayBroadcastingIter<'a> {
    pub fn new( // Make pub
        buffer: &'a Arc<Vec<f32>>,
        original_shape: &'a [usize],
        original_strides: &'a [usize],
        original_offset: usize,
        target_shape: &'a [usize],
    ) -> Result<Self, NeuraRustError> {
        if !original_shape.is_empty() && original_shape.len() > target_shape.len() {
            // Basic check
        }
        
        let total_elements = target_shape.iter().product();
        Ok(Self {
            buffer,
            original_shape,
            original_strides,
            original_offset,
            target_shape,
            current_index: 0,
            total_elements,
        })
    }
}

impl<'a> Iterator for NdArrayBroadcastingIter<'a> {
    type Item = f32; // Iterates over f32 values

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_elements {
            return None;
        }

        let target_rank = self.target_shape.len();
        let original_rank = self.original_shape.len();
        
        // Calculate multi-dimensional index in the target shape
        let mut target_multi_index = vec![0; target_rank];
        let mut current_linear = self.current_index;
        for dim in (0..target_rank).rev() {
            let shape_val = self.target_shape[dim];
             if shape_val > 0 { // Avoid division by zero for empty dimensions
                target_multi_index[dim] = current_linear % shape_val;
                current_linear /= shape_val;
            } else {
                target_multi_index[dim] = 0;
            }
        }

        // Calculate corresponding multi-dimensional index in the original shape
        let mut original_multi_index = vec![0; original_rank];
        let rank_diff = target_rank as isize - original_rank as isize;

        for original_dim in 0..original_rank {
             let target_dim = (original_dim as isize + rank_diff) as usize;
             // If the original dimension size is 1, use index 0 (broadcast)
             // Otherwise, use the corresponding index from the target shape
             if self.original_shape[original_dim] == 1 {
                 original_multi_index[original_dim] = 0;
             } else {
                 // Boundary check for target_multi_index access
                 if target_dim < target_multi_index.len() {
                    original_multi_index[original_dim] = target_multi_index[target_dim];
                 } else {
                     // This case might indicate an issue with shape validation logic upstream.
                     // For robustness, maybe default to 0 or handle error? Assuming 0 for now.
                     original_multi_index[original_dim] = 0;
                     // Consider logging a warning or returning an error if this happens unexpectedly.
                     // eprintln!("Warning: target_dim out of bounds in NdArrayBroadcastingIter::next");
                 }
             }
        }

        // Calculate physical offset using original strides
        let physical_offset = self.original_offset
            + original_multi_index
                .iter()
                .zip(self.original_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();

        // Boundary check before accessing buffer
        if physical_offset >= self.buffer.len() {
             // This indicates an internal error, potentially incorrect strides or offset calculation.
             // Return None or handle error appropriately. Returning None signals end of iteration.
             // eprintln!("Warning: physical_offset out of bounds in NdArrayBroadcastingIter::next");
             self.current_index = self.total_elements; // Force iterator end
             return None;
        }


        let value = self.buffer[physical_offset];
        self.current_index += 1;
        Some(value)
    }
}

// NdArrayBroadcastingIter for f64
#[derive(Debug)] // Add Debug derive
pub struct NdArrayBroadcastingIterF64<'a> { // Make pub
    buffer: &'a Arc<Vec<f64>>,
    original_shape: &'a [usize],
    original_strides: &'a [usize],
    original_offset: usize,
    target_shape: &'a [usize],
    current_index: usize,
    total_elements: usize,
}

impl<'a> NdArrayBroadcastingIterF64<'a> {
    pub fn new( // Make pub
        buffer: &'a Arc<Vec<f64>>,
        original_shape: &'a [usize],
        original_strides: &'a [usize],
        original_offset: usize,
        target_shape: &'a [usize],
    ) -> Result<Self, NeuraRustError> {
        if !original_shape.is_empty() && original_shape.len() > target_shape.len() {
           // Basic check
        }
        let total_elements = target_shape.iter().product();
        Ok(Self {
            buffer,
            original_shape,
            original_strides,
            original_offset,
            target_shape,
            current_index: 0,
            total_elements,
        })
    }
}

impl<'a> Iterator for NdArrayBroadcastingIterF64<'a> {
    type Item = f64; // Iterates over f64 values

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_elements {
            return None;
        }

        let target_rank = self.target_shape.len();
        let original_rank = self.original_shape.len();
        
        let mut target_multi_index = vec![0; target_rank];
        let mut current_linear = self.current_index;
        for dim in (0..target_rank).rev() {
            let shape_val = self.target_shape[dim];
             if shape_val > 0 {
                target_multi_index[dim] = current_linear % shape_val;
                current_linear /= shape_val;
            } else {
                target_multi_index[dim] = 0;
            }
        }

        let mut original_multi_index = vec![0; original_rank];
        let rank_diff = target_rank as isize - original_rank as isize;
        for original_dim in 0..original_rank {
             let target_dim = (original_dim as isize + rank_diff) as usize;
             if self.original_shape[original_dim] == 1 {
                 original_multi_index[original_dim] = 0;
             } else {
                  // Boundary check for target_multi_index access
                 if target_dim < target_multi_index.len() {
                    original_multi_index[original_dim] = target_multi_index[target_dim];
                 } else {
                     // eprintln!("Warning: target_dim out of bounds in NdArrayBroadcastingIterF64::next");
                     original_multi_index[original_dim] = 0;
                 }
             }
        }

        let physical_offset = self.original_offset
            + original_multi_index
                .iter()
                .zip(self.original_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();
        
        // Boundary check before accessing buffer
        if physical_offset >= self.buffer.len() {
            // eprintln!("Warning: physical_offset out of bounds in NdArrayBroadcastingIterF64::next");
             self.current_index = self.total_elements; // Force iterator end
             return None;
        }

        let value = self.buffer[physical_offset]; // Access f64 buffer
        self.current_index += 1;
        Some(value)
    }
}

// --- Single NdArray Iterator (Handles strides/offsets) ---

pub struct NdArraySimpleIter<'a> {
    pub(crate) buffer: &'a [f32],
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) offset: usize,
    pub(crate) current_logical_index: Vec<usize>,
    pub(crate) current_physical_offset: usize,
    pub(crate) is_done: bool,
    pub(crate) numel: usize,
    pub(crate) counter: usize, // To limit iterations to numel
}

impl<'a> NdArraySimpleIter<'a> {
    pub fn new(
        buffer: &'a [f32],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, NeuraRustError> {
        if shape.len() != strides.len() {
            return Err(NeuraRustError::ShapeMismatch {
                expected: "Matching shape and strides length".to_string(),
                actual: format!("Shape len: {}, Strides len: {}", shape.len(), strides.len()),
                operation: "NdArraySimpleIter::new".to_string(),
            });
        }
        let numel = shape.iter().product();
        let initial_physical_offset = offset + calculate_physical_offset(&vec![0; shape.len()], strides)?;

        Ok(NdArraySimpleIter {
            buffer,
            shape,
            strides,
            offset,
            current_logical_index: vec![0; shape.len()],
            current_physical_offset: initial_physical_offset,
            is_done: numel == 0, // Done immediately if empty
            numel,
            counter: 0,
        })
    }

    #[inline]
    fn advance(&mut self) {
        if self.is_done || self.shape.is_empty() { return; }
        
        for i in (0..self.shape.len()).rev() {
            // Subtract old stride contribution for this dimension
            self.current_physical_offset -= self.current_logical_index[i] * self.strides[i];

            self.current_logical_index[i] += 1;
            if self.current_logical_index[i] < self.shape[i] {
                 // Add new stride contribution
                self.current_physical_offset += self.current_logical_index[i] * self.strides[i];
                return; // Found the dimension to increment, finished advancing
            }
            // Reset this dimension's index and continue to the next
            self.current_logical_index[i] = 0;
            // Physical offset for index 0 is added implicitly when the loop continues/ends
        }
        // If we fall through, we have rolled over all dimensions
        self.is_done = true;
        // Reset physical offset calculation basis for next potential (but unused) advance
        self.current_physical_offset = self.offset; 
    }
}

impl<'a> Iterator for NdArraySimpleIter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done || self.counter >= self.numel {
            return None;
        }

        let value = self.buffer[self.current_physical_offset];
        self.counter += 1;
        self.advance();
        Some(value)
    }
}

// --- F64 Version ---
pub struct NdArraySimpleIterF64<'a> {
    pub(crate) buffer: &'a [f64],
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) offset: usize,
    pub(crate) current_logical_index: Vec<usize>,
    pub(crate) current_physical_offset: usize,
    pub(crate) is_done: bool,
    pub(crate) numel: usize,
    pub(crate) counter: usize,
}

impl<'a> NdArraySimpleIterF64<'a> {
     pub fn new(
        buffer: &'a [f64],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, NeuraRustError> {
        if shape.len() != strides.len() {
            return Err(NeuraRustError::ShapeMismatch {
                expected: "Matching shape and strides length".to_string(),
                actual: format!("Shape len: {}, Strides len: {}", shape.len(), strides.len()),
                operation: "NdArraySimpleIterF64::new".to_string(),
            });
        }
        let numel = shape.iter().product();
        let initial_physical_offset = offset + calculate_physical_offset(&vec![0; shape.len()], strides)?;

        Ok(NdArraySimpleIterF64 {
            buffer,
            shape,
            strides,
            offset,
            current_logical_index: vec![0; shape.len()],
            current_physical_offset: initial_physical_offset,
            is_done: numel == 0,
            numel,
            counter: 0,
        })
    }

    #[inline]
    fn advance(&mut self) {
        if self.is_done || self.shape.is_empty() { return; }
        
        for i in (0..self.shape.len()).rev() {
            self.current_physical_offset -= self.current_logical_index[i] * self.strides[i];
            self.current_logical_index[i] += 1;
            if self.current_logical_index[i] < self.shape[i] {
                self.current_physical_offset += self.current_logical_index[i] * self.strides[i];
                return;
            }
            self.current_logical_index[i] = 0;
        }
        self.is_done = true;
         self.current_physical_offset = self.offset; 
    }
}

impl<'a> Iterator for NdArraySimpleIterF64<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done || self.counter >= self.numel {
            return None;
        }
        let value = self.buffer[self.current_physical_offset];
        self.counter += 1;
        self.advance();
        Some(value)
    }
}

// NdArrayBroadcastingIter for i32
#[derive(Debug)]
pub struct NdArrayBroadcastingIterI32<'a> {
    buffer: &'a Arc<Vec<i32>>,
    original_shape: &'a [usize],
    original_strides: &'a [usize],
    original_offset: usize,
    target_shape: &'a [usize],
    current_index: usize,
    total_elements: usize,
}

impl<'a> NdArrayBroadcastingIterI32<'a> {
    pub fn new(
        buffer: &'a Arc<Vec<i32>>,
        original_shape: &'a [usize],
        original_strides: &'a [usize],
        original_offset: usize,
        target_shape: &'a [usize],
    ) -> Result<Self, NeuraRustError> {
        let total_elements = target_shape.iter().product();
        Ok(Self {
            buffer,
            original_shape,
            original_strides,
            original_offset,
            target_shape,
            current_index: 0,
            total_elements,
        })
    }
}

impl<'a> Iterator for NdArrayBroadcastingIterI32<'a> {
    type Item = i32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_elements {
            return None;
        }
        let target_rank = self.target_shape.len();
        let original_rank = self.original_shape.len();
        let mut target_multi_index = vec![0; target_rank];
        let mut current_linear = self.current_index;
        for dim in (0..target_rank).rev() {
            let shape_val = self.target_shape[dim];
            if shape_val > 0 {
                target_multi_index[dim] = current_linear % shape_val;
                current_linear /= shape_val;
            } else {
                target_multi_index[dim] = 0;
            }
        }
        let mut original_multi_index = vec![0; original_rank];
        let rank_diff = target_rank as isize - original_rank as isize;
        for original_dim in 0..original_rank {
            let target_dim = (original_dim as isize + rank_diff) as usize;
            if self.original_shape[original_dim] == 1 {
                original_multi_index[original_dim] = 0;
            } else if target_dim < target_multi_index.len() {
                original_multi_index[original_dim] = target_multi_index[target_dim];
            } else {
                original_multi_index[original_dim] = 0;
            }
        }
        let physical_offset = self.original_offset
            + original_multi_index
                .iter()
                .zip(self.original_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();
        if physical_offset >= self.buffer.len() {
            self.current_index = self.total_elements;
            return None;
        }
        let value = self.buffer[physical_offset];
        self.current_index += 1;
        Some(value)
    }
}

// NdArrayBroadcastingIter for i64
#[derive(Debug)]
pub struct NdArrayBroadcastingIterI64<'a> {
    buffer: &'a Arc<Vec<i64>>,
    original_shape: &'a [usize],
    original_strides: &'a [usize],
    original_offset: usize,
    target_shape: &'a [usize],
    current_index: usize,
    total_elements: usize,
}

impl<'a> NdArrayBroadcastingIterI64<'a> {
    pub fn new(
        buffer: &'a Arc<Vec<i64>>,
        original_shape: &'a [usize],
        original_strides: &'a [usize],
        original_offset: usize,
        target_shape: &'a [usize],
    ) -> Result<Self, NeuraRustError> {
        let total_elements = target_shape.iter().product();
        Ok(Self {
            buffer,
            original_shape,
            original_strides,
            original_offset,
            target_shape,
            current_index: 0,
            total_elements,
        })
    }
}

impl<'a> Iterator for NdArrayBroadcastingIterI64<'a> {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_elements {
            return None;
        }
        let target_rank = self.target_shape.len();
        let original_rank = self.original_shape.len();
        let mut target_multi_index = vec![0; target_rank];
        let mut current_linear = self.current_index;
        for dim in (0..target_rank).rev() {
            let shape_val = self.target_shape[dim];
            if shape_val > 0 {
                target_multi_index[dim] = current_linear % shape_val;
                current_linear /= shape_val;
            } else {
                target_multi_index[dim] = 0;
            }
        }
        let mut original_multi_index = vec![0; original_rank];
        let rank_diff = target_rank as isize - original_rank as isize;
        for original_dim in 0..original_rank {
            let target_dim = (original_dim as isize + rank_diff) as usize;
            if self.original_shape[original_dim] == 1 {
                original_multi_index[original_dim] = 0;
            } else if target_dim < target_multi_index.len() {
                original_multi_index[original_dim] = target_multi_index[target_dim];
            } else {
                original_multi_index[original_dim] = 0;
            }
        }
        let physical_offset = self.original_offset
            + original_multi_index
                .iter()
                .zip(self.original_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();
        if physical_offset >= self.buffer.len() {
            self.current_index = self.total_elements;
            return None;
        }
        let value = self.buffer[physical_offset];
        self.current_index += 1;
        Some(value)
    }
}

#[derive(Debug)]
pub struct NdArrayBroadcastingIterBool<'a> {
    buffer: &'a Arc<Vec<bool>>,
    original_shape: &'a [usize],
    original_strides: &'a [usize],
    original_offset: usize,
    target_shape: &'a [usize],
    current_index: usize,
    total_elements: usize,
}

impl<'a> NdArrayBroadcastingIterBool<'a> {
    pub fn new(
        buffer: &'a Arc<Vec<bool>>,
        original_shape: &'a [usize],
        original_strides: &'a [usize],
        original_offset: usize,
        target_shape: &'a [usize],
    ) -> Result<Self, NeuraRustError> {
        let total_elements = target_shape.iter().product();
        Ok(Self {
            buffer,
            original_shape,
            original_strides,
            original_offset,
            target_shape,
            current_index: 0,
            total_elements,
        })
    }
}

impl<'a> Iterator for NdArrayBroadcastingIterBool<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_elements {
            return None;
        }
        let target_rank = self.target_shape.len();
        let original_rank = self.original_shape.len();
        let mut target_multi_index = vec![0; target_rank];
        let mut current_linear = self.current_index;
        for dim in (0..target_rank).rev() {
            let shape_val = self.target_shape[dim];
            if shape_val > 0 {
                target_multi_index[dim] = current_linear % shape_val;
                current_linear /= shape_val;
            } else {
                target_multi_index[dim] = 0;
            }
        }
        let mut original_multi_index = vec![0; original_rank];
        let rank_diff = target_rank as isize - original_rank as isize;
        for original_dim in 0..original_rank {
            let target_dim = (original_dim as isize + rank_diff) as usize;
            if self.original_shape[original_dim] == 1 {
                original_multi_index[original_dim] = 0;
            } else if target_dim < target_multi_index.len() {
                original_multi_index[original_dim] = target_multi_index[target_dim];
            } else {
                original_multi_index[original_dim] = 0;
            }
        }
        let physical_offset = self.original_offset
            + original_multi_index
                .iter()
                .zip(self.original_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();
        if physical_offset >= self.buffer.len() {
            self.current_index = self.total_elements;
            return None;
        }
        let value = self.buffer[physical_offset];
        self.current_index += 1;
        Some(value)
    }
} 