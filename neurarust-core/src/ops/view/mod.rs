// neurarust-core/src/ops/view/mod.rs

mod slice;
mod transpose;
mod permute;
mod reshape;

// Utiliser pub(crate) use pour rendre accessible à l'intérieur du crate
// tout en résolvant l'erreur de visibilité.
pub(crate) use slice::slice_op;
pub(crate) use transpose::transpose_op;
pub(crate) use permute::permute_op;
pub(crate) use reshape::reshape_op;

// Conserver la définition de SliceArg ici car elle est utilisée par slice_op
// et potentiellement par l'API publique via Tensor::slice plus tard.
#[derive(Debug, Clone)]
pub struct SliceArg {
    pub start: usize,
    pub end: usize,
    // Optional: Add step later if needed
    // pub step: usize,
}

impl SliceArg {
    // Helper constructor if needed
    pub fn new(start: usize, end: usize) -> Self {
        SliceArg { start, end }
    }
}

// Le reste du code (implémentations des ops, backward structs, tests)
// a été déplacé dans les modules slice.rs, transpose.rs, etc.
