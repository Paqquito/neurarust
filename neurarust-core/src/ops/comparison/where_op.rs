//! Opération conditionnelle where (ternaire) : where_op

use crate::tensor::Tensor;
use crate::error::NeuraRustError;
use crate::types::DType;
use crate::tensor::utils::broadcast_shapes;
use crate::tensor::create;
use std::sync::{Arc, RwLock};
use crate::tensor_data::TensorData;
use crate::autograd::backward_op::BackwardOp;
use crate::ops::dtype::cast_op;
use crate::tensor::broadcast_utils;
use crate::ops::arithmetic::mul_op;
use crate::ops::comparison::logical_not_op;

/// Opération conditionnelle élément par élément :
/// Retourne un tenseur avec les éléments de `x` là où `condition` est vrai, sinon ceux de `y`.
///
/// # Arguments
/// * `condition` - Tenseur booléen (DType::Bool)
/// * `x` - Tenseur de même DType que `y`, broadcastable avec `condition`
/// * `y` - Tenseur de même DType que `x`, broadcastable avec `condition`
///
/// # Retour
/// Tenseur résultant du même DType que `x`/`y`.
pub fn where_op(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, NeuraRustError> {
    // Vérification du DType de la condition
    if condition.dtype() != DType::Bool {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: DType::Bool,
            actual: condition.dtype(),
            operation: "where_op (condition)".to_string(),
        });
    }
    // Vérification du DType de x et y
    if x.dtype() != y.dtype() {
        return Err(NeuraRustError::DataTypeMismatch {
            expected: x.dtype(),
            actual: y.dtype(),
            operation: "where_op (x/y)".to_string(),
        });
    }
    // Vérification du device (on suppose que tous doivent être sur le même device)
    if x.device() != y.device() || x.device() != condition.device() {
        return Err(NeuraRustError::DeviceMismatch {
            expected: x.device(),
            actual: y.device(),
            operation: "where_op (device)".to_string(),
        });
    }
    // Calcul du broadcast shape
    let shape = broadcast_shapes(&condition.shape()[..], &x.shape()[..])
        .map_err(|_| NeuraRustError::BroadcastError {
            shape1: condition.shape().to_vec(),
            shape2: x.shape().to_vec(),
        })?;
    let shape = broadcast_shapes(&shape[..], &y.shape()[..])
        .map_err(|_| NeuraRustError::BroadcastError {
            shape1: shape.clone(),
            shape2: y.shape().to_vec(),
        })?;

    // Récupération des données des tenseurs
    let cond_data = condition.get_bool_data()?;

    // Application de la sélection selon la condition
    let out = match x.dtype() {
        DType::F32 => {
            let x_data = x.get_f32_data()?;
            let y_data = y.get_f32_data()?;
            let mut out_data = vec![0.0f32; shape.iter().product()];
            for i in 0..out_data.len() {
                let coords = crate::tensor::utils::index_to_coord(i, &shape);
                let cond_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - condition.shape().len() {
                        let cond_dim = dim - (shape.len() - condition.shape().len());
                        if condition.shape()[cond_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let x_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - x.shape().len() {
                        let x_dim = dim - (shape.len() - x.shape().len());
                        if x.shape()[x_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let y_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - y.shape().len() {
                        let y_dim = dim - (shape.len() - y.shape().len());
                        if y.shape()[y_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let cond_idx = crate::tensor::utils::coord_to_index(&cond_coords, &condition.shape());
                let x_idx = crate::tensor::utils::coord_to_index(&x_coords, &x.shape());
                let y_idx = crate::tensor::utils::coord_to_index(&y_coords, &y.shape());
                out_data[i] = if cond_data[cond_idx] { x_data[x_idx] } else { y_data[y_idx] };
            }
            create::from_vec_f32(out_data, shape)?
        }
        DType::F64 => {
            let x_data = x.get_f64_data()?;
            let y_data = y.get_f64_data()?;
            let mut out_data = vec![0.0f64; shape.iter().product()];
            for i in 0..out_data.len() {
                let coords = crate::tensor::utils::index_to_coord(i, &shape);
                let cond_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - condition.shape().len() {
                        let cond_dim = dim - (shape.len() - condition.shape().len());
                        if condition.shape()[cond_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let x_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - x.shape().len() {
                        let x_dim = dim - (shape.len() - x.shape().len());
                        if x.shape()[x_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let y_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - y.shape().len() {
                        let y_dim = dim - (shape.len() - y.shape().len());
                        if y.shape()[y_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let cond_idx = crate::tensor::utils::coord_to_index(&cond_coords, &condition.shape());
                let x_idx = crate::tensor::utils::coord_to_index(&x_coords, &x.shape());
                let y_idx = crate::tensor::utils::coord_to_index(&y_coords, &y.shape());
                out_data[i] = if cond_data[cond_idx] { x_data[x_idx] } else { y_data[y_idx] };
            }
            create::from_vec_f64(out_data, shape)?
        }
        DType::I32 => {
            let x_data = x.get_i32_data()?;
            let y_data = y.get_i32_data()?;
            let mut out_data = vec![0i32; shape.iter().product()];
            for i in 0..out_data.len() {
                let coords = crate::tensor::utils::index_to_coord(i, &shape);
                let cond_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - condition.shape().len() {
                        let cond_dim = dim - (shape.len() - condition.shape().len());
                        if condition.shape()[cond_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let x_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - x.shape().len() {
                        let x_dim = dim - (shape.len() - x.shape().len());
                        if x.shape()[x_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let y_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - y.shape().len() {
                        let y_dim = dim - (shape.len() - y.shape().len());
                        if y.shape()[y_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let cond_idx = crate::tensor::utils::coord_to_index(&cond_coords, &condition.shape());
                let x_idx = crate::tensor::utils::coord_to_index(&x_coords, &x.shape());
                let y_idx = crate::tensor::utils::coord_to_index(&y_coords, &y.shape());
                out_data[i] = if cond_data[cond_idx] { x_data[x_idx] } else { y_data[y_idx] };
            }
            create::from_vec_i32(out_data, shape)?
        }
        DType::I64 => {
            let x_data = x.get_i64_data()?;
            let y_data = y.get_i64_data()?;
            let mut out_data = vec![0i64; shape.iter().product()];
            for i in 0..out_data.len() {
                let coords = crate::tensor::utils::index_to_coord(i, &shape);
                let cond_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - condition.shape().len() {
                        let cond_dim = dim - (shape.len() - condition.shape().len());
                        if condition.shape()[cond_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let x_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - x.shape().len() {
                        let x_dim = dim - (shape.len() - x.shape().len());
                        if x.shape()[x_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let y_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - y.shape().len() {
                        let y_dim = dim - (shape.len() - y.shape().len());
                        if y.shape()[y_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let cond_idx = crate::tensor::utils::coord_to_index(&cond_coords, &condition.shape());
                let x_idx = crate::tensor::utils::coord_to_index(&x_coords, &x.shape());
                let y_idx = crate::tensor::utils::coord_to_index(&y_coords, &y.shape());
                out_data[i] = if cond_data[cond_idx] { x_data[x_idx] } else { y_data[y_idx] };
            }
            create::from_vec_i64(out_data, shape)?
        }
        DType::Bool => {
            let x_data = x.get_bool_data()?;
            let y_data = y.get_bool_data()?;
            let mut out_data = vec![false; shape.iter().product()];
            for i in 0..out_data.len() {
                let coords = crate::tensor::utils::index_to_coord(i, &shape);
                let cond_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - condition.shape().len() {
                        let cond_dim = dim - (shape.len() - condition.shape().len());
                        if condition.shape()[cond_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let x_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - x.shape().len() {
                        let x_dim = dim - (shape.len() - x.shape().len());
                        if x.shape()[x_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let y_coords: Vec<usize> = coords.iter().enumerate().map(|(dim, &c)| {
                    if dim >= shape.len() - y.shape().len() {
                        let y_dim = dim - (shape.len() - y.shape().len());
                        if y.shape()[y_dim] == 1 { 0 } else { c }
                    } else { 0 }
                }).collect();
                let cond_idx = crate::tensor::utils::coord_to_index(&cond_coords, &condition.shape());
                let x_idx = crate::tensor::utils::coord_to_index(&x_coords, &x.shape());
                let y_idx = crate::tensor::utils::coord_to_index(&y_coords, &y.shape());
                out_data[i] = if cond_data[cond_idx] { x_data[x_idx] } else { y_data[y_idx] };
            }
            create::from_vec_bool(out_data, shape)?
        }
//        _ => return Err(NeuraRustError::UnsupportedOperation(format!("where_op: type non supporté {:?}", x.dtype()))),
    };

    // --- Autograd ---
    let x_requires_grad = x.requires_grad();
    let y_requires_grad = y.requires_grad();
    if x_requires_grad || y_requires_grad {
        let grad_fn = WhereBackward {
            cond_node: condition.data.clone(),
            x_node: x.data.clone(),
            y_node: y.data.clone(),
            x_shape: x.shape().to_vec(),
            y_shape: y.shape().to_vec(),
            x_requires_grad,
            y_requires_grad,
        };
        let mut out_guard = out.write_data();
        out_guard.grad_fn = Some(Arc::new(grad_fn));
        out_guard.requires_grad = true;
    }
    Ok(out)
}

#[derive(Debug)]
struct WhereBackward {
    cond_node: Arc<RwLock<TensorData>>,
    x_node: Arc<RwLock<TensorData>>,
    y_node: Arc<RwLock<TensorData>>,
    x_shape: Vec<usize>,
    y_shape: Vec<usize>,
    x_requires_grad: bool,
    y_requires_grad: bool,
}

impl BackwardOp for WhereBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        let mut grads = Vec::new();
        let cond_tensor = Tensor { data: self.cond_node.clone() };
        let cond_b = cond_tensor.expand_to_match_nd(&grad_output.shape())?;
        let not_cond_b = logical_not_op(&cond_b)?;
        let cond_b_contig = cond_b.contiguous()?;
        let not_cond_b_contig = not_cond_b.contiguous()?;
        let cond_b_cast = cast_op(&cond_b_contig, grad_output.dtype())?;
        let not_cond_b_cast = cast_op(&not_cond_b_contig, grad_output.dtype())?;

        if self.x_requires_grad {
            let grad_x = mul_op(grad_output, &cond_b_cast)?;
            let grad_x_reduced = broadcast_utils::reduce_broadcasted_gradient(&grad_x, &self.x_shape)?;
            grads.push(grad_x_reduced);
        }
        if self.y_requires_grad {
            let grad_y = mul_op(grad_output, &not_cond_b_cast)?;
            let grad_y_reduced = broadcast_utils::reduce_broadcasted_gradient(&grad_y, &self.y_shape)?;
            grads.push(grad_y_reduced);
        }
        Ok(grads)
    }
    fn inputs(&self) -> Vec<*const RwLock<TensorData>> {
        let mut ids = Vec::new();
        if self.x_requires_grad {
            ids.push(Arc::as_ptr(&self.x_node));
        }
        if self.y_requires_grad {
            ids.push(Arc::as_ptr(&self.y_node));
        }
        ids
    }
} 