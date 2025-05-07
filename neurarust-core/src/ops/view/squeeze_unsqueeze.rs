use crate::{
    autograd::{graph::NodeId, BackwardOp},
    error::NeuraRustError,
    tensor::Tensor,
    tensor_data::TensorData,
};
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

#[derive(Debug)]
struct UnsqueezeBackward {
    input_node_id: NodeId, // Used as an opaque ID
    original_shape: Vec<usize>,
    dim: usize,
}

// Unsafe impl Send and Sync because NodeId is a raw pointer but used as an ID.
// The actual data access is guarded by Tensor's Arc<RwLock<TensorData>>.
unsafe impl Send for UnsqueezeBackward {}
unsafe impl Sync for UnsqueezeBackward {}

impl BackwardOp for UnsqueezeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // Le backward de unsqueeze est squeeze sur la dimension où l'unsqueeze a eu lieu.
        // La shape de grad_output est la shape après unsqueeze.
        // La shape attendue pour grad_input est self.original_shape.
        
        // Squeezer la dimension `self.dim` de `grad_output`.
        // Note: `self.dim` est l'index dans la *nouvelle* shape (après unsqueeze).
        let grad_input = squeeze_op(grad_output, Some(self.dim))?;

        // Normalement, après avoir squeezé la dimension ajoutée par unsqueeze,
        // la shape de grad_input devrait correspondre à self.original_shape.
        if grad_input.shape() != self.original_shape {
            // Si ce n'est pas le cas, cela peut indiquer un problème. 
            // Par exemple, si unsqueeze(0) sur [2,3] -> [1,2,3], puis squeeze(0) sur grad [1,2,3] -> [2,3]. C'est correct.
            // Mais si original_shape était [] (scalaire) et unsqueeze(0) -> [1]. grad_output est [1]. squeeze(0) sur [1] -> []. Correct.
            // Si original_shape était [1] et unsqueeze(0) -> [1,1]. grad_output est [1,1]. squeeze(0) sur [1,1] -> [1]. Correct.
            // Si original_shape était [2,1,3] et unsqueeze(1) -> [2,1,1,3]. grad_output est [2,1,1,3].
            // squeeze(1) sur [2,1,1,3] -> [2,1,3]. Correct.
            // La vérification est une sécurité, mais squeeze_op avec une dim spécifique devrait juste enlever cette dim si elle est 1.
            // Un reshape pourrait être nécessaire seulement si squeeze(None) était utilisé et qu'il y avait d'autres dimensions de taille 1
            // qui ne correspondaient pas à `self.dim` mais qui ont été squeezées quand même, ce qui ne devrait pas arriver ici.
            return Err(NeuraRustError::InternalError(format!(
                "UnsqueezeBackward: shape mismatch after squeeze. Expected {:?}, got {:?}. grad_output shape: {:?}, dim: {}",
                self.original_shape,
                grad_input.shape(),
                grad_output.shape(),
                self.dim
            )));
        }
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input_node_id]
    }
}

pub fn unsqueeze_op(tensor: &Tensor, dim: usize) -> Result<Tensor, NeuraRustError> {
    let input_guard = tensor.read_data();
    let input_shape = &input_guard.shape;
    let input_strides = &input_guard.strides;
    let rank = input_shape.len();

    if dim > rank {
        return Err(NeuraRustError::InvalidAxis {
            axis: dim,
            rank, // Correct rank for error message (max insert position is rank)
        });
    }

    let mut new_shape = input_shape.clone();
    new_shape.insert(dim, 1);

    let mut new_strides = vec![0; new_shape.len()];

    if new_shape.is_empty() { // Should not happen if unsqueezing from scalar (shape []) which becomes [1]
        // This case is more for if new_shape became empty, which unsqueeze doesn't do.
        // However, if input was scalar (rank 0), new_shape is [1].
    } else if new_shape.len() == 1 { // Input was scalar (shape [], rank 0), new_shape is [1]
        new_strides[0] = 1;
    } else {
        // Copy strides for dimensions before the inserted dimension
        for i in 0..dim {
            new_strides[i] = input_strides[i];
        }
        
        // Set stride for the new dimension
        if dim == rank { // Inserted at the end, new dimension is the most minor, stride 1
            new_strides[dim] = 1;
        } else { // Inserted at the beginning or in the middle
                 // The stride of the new dimension is the same as the stride of the
                 // dimension that *was* at this position (input_strides[dim]).
            new_strides[dim] = input_strides[dim];
        }

        // Copy strides for dimensions after the inserted dimension (shifted by 1 in new_strides)
        for i in dim..input_shape.len() { // Iterate through remaining original dimensions
            new_strides[i + 1] = input_strides[i];
        }
    }

    let requires_grad_input = input_guard.requires_grad;
    let input_node_id = tensor.node_id();
    
    let buffer_arc = input_guard.buffer.clone();
    let device = input_guard.device;
    let offset = input_guard.offset;
    // DType is part of the buffer_arc / TensorData internal structure
    let original_shape_clone = input_shape.clone();

    drop(input_guard);

    let view_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape,
        new_strides,
    )?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    if requires_grad_input {
        let grad_fn_arc = Arc::new(UnsqueezeBackward {
            input_node_id,
            original_shape: original_shape_clone,
            dim,
        }) as Arc<dyn BackwardOp>; 
        output_tensor.set_grad_fn(Some(grad_fn_arc))?;
        output_tensor.set_requires_grad(true)?;
    }

    Ok(output_tensor)
}

#[derive(Debug)]
struct SqueezeBackward {
    input_node_id: NodeId,
    original_shape: Vec<usize>, // Pour reconstruire lors de l'unsqueeze dans le backward
                                // Alternative: stocker les axes qui ont été squeezés.
                                // Pour l'instant, original_shape est plus simple à gérer.
}

// Unsafe impl Send and Sync pour la même raison que UnsqueezeBackward
unsafe impl Send for SqueezeBackward {}
unsafe impl Sync for SqueezeBackward {}

impl BackwardOp for SqueezeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, NeuraRustError> {
        // Le backward de squeeze est un unsqueeze. 
        // Il faut unsqueezer grad_output pour qu'il corresponde à self.original_shape.
        // Cela peut nécessiter plusieurs unsqueeze si plusieurs dims ont été squeezées.
        
        let mut current_grad = grad_output.clone();
        let current_shape = grad_output.shape();
        let target_shape = &self.original_shape;
        let target_rank = target_shape.len();
        let mut current_rank = current_shape.len();
        
        // Si current_shape est scalaire [] et target_shape ne l'est pas (ex: [1,1,5])
        // nous devons insérer les dimensions 1.
        if current_rank == 0 && target_rank > 0 {
            // Un tensor scalaire squeezé venait d'un tensor avec que des 1s, ex: [1,1,1]
            // On le reshape à la forme cible (qui ne devrait contenir que des 1s si c'est le cas)
            // Ceci est un cas spécial. Normalement, on unsqueeze les dims manquantes.
            if target_shape.iter().all(|&d| d == 1) && target_shape.iter().product::<usize>() == 1 {
                // current_grad = current_grad.reshape(target_shape.clone())?; // Assignation commentée car current_grad n'est pas lu ensuite dans cette branche
                // current_shape = current_grad.shape(); 
                // current_rank = current_shape.len(); 
            } else {
                // Plus complexe: reconstruire par unsqueezes successifs
                // En partant de la fin pour que les indices de dim soient stables
                for i in (0..target_rank).rev() {
                    if current_rank < target_rank && target_shape[i] == 1 {
                         // On assume que si le rang est inférieur, les dims manquantes sont celles de taille 1
                         // et on les ajoute à la position correspondante de la target_shape
                         current_grad = unsqueeze_op(&current_grad, i)?;
                         current_rank += 1;
                         if current_rank == target_rank { break; }
                    } else if current_rank == target_rank && current_shape[i] != target_shape[i] && target_shape[i] == 1 {
                        // Ce cas ne devrait pas arriver si current_shape[i] != target_shape[i] ET target_shape[i] == 1
                        // car la dim aurait dû être squeezée ou être présente.
                        // Sauf si current_shape[i] était > 1 et target_shape[i] == 1 (ne devrait pas être le cas pour SqueezeBackward)
                    }
                }
            }
        }

        // Itérer pour insérer les dimensions qui étaient 1 dans target_shape mais sont absentes ou différentes dans current_shape
        // On doit reconstruire la forme originale self.original_shape à partir de grad_output.shape()
        // en insérant des dimensions de taille 1 là où elles ont été squeezées.
        let mut grad_input = grad_output.clone();
        let mut rebuilt_shape = grad_output.shape();
        let mut original_dim_idx = 0;
        let mut grad_dim_idx = 0;

        while original_dim_idx < self.original_shape.len() {
            if grad_dim_idx < rebuilt_shape.len() && self.original_shape[original_dim_idx] == rebuilt_shape[grad_dim_idx] {
                // Les dimensions correspondent, on avance
                original_dim_idx += 1;
                grad_dim_idx += 1;
            } else if self.original_shape[original_dim_idx] == 1 {
                // Cette dimension de taille 1 a été squeezée, il faut la réinsérer
                grad_input = unsqueeze_op(&grad_input, original_dim_idx)?;
                rebuilt_shape = grad_input.shape(); // Mettre à jour la shape reconstruite
                original_dim_idx += 1;
                // grad_dim_idx ne change pas car on a inséré avant
            } else {
                // Incohérence, les shapes ne peuvent pas être réconciliées
                return Err(NeuraRustError::ShapeMismatch {
                    operation: "SqueezeBackward".to_string(),
                    expected: format!("A shape compatible with {:?}", self.original_shape),
                    actual: format!("Got {:?}", grad_output.shape()),
                });
            }
        }
        
        // Vérification finale
        if grad_input.shape() != self.original_shape {
            return Err(NeuraRustError::InternalError(format!(
                "SqueezeBackward failed to reconstruct original shape. Expected {:?}, got {:?}",
                self.original_shape,
                grad_input.shape()
            )));
        }

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input_node_id]
    }
}

pub fn squeeze_op(tensor: &Tensor, dim_opt: Option<usize>) -> Result<Tensor, NeuraRustError> {
    let input_guard = tensor.read_data();
    let input_shape = &input_guard.shape;
    let input_strides = &input_guard.strides;
    let rank = input_shape.len();

    let mut new_shape: Vec<usize> = Vec::new();
    let mut new_strides: Vec<usize> = Vec::new();
    let mut squeezed_dims_indices: Vec<usize> = Vec::new(); // Pour l'autograd

    match dim_opt {
        Some(d) => {
            if d >= rank {
                return Err(NeuraRustError::InvalidAxis { axis: d, rank });
            }
            for i in 0..rank {
                if i == d && input_shape[i] == 1 {
                    squeezed_dims_indices.push(i);
                    // On ne push pas cette dimension dans new_shape/new_strides
                } else {
                    new_shape.push(input_shape[i]);
                    new_strides.push(input_strides[i]);
                }
            }
        }
        None => {
            if rank == 0 { // Si c'est un scalaire, shape [], on ne peut rien squeezer de plus.
                new_shape = input_shape.clone();
                new_strides = input_strides.clone();
            } else {
                for i in 0..rank {
                    if input_shape[i] == 1 {
                        squeezed_dims_indices.push(i);
                        // On ne push pas cette dimension
                    } else {
                        new_shape.push(input_shape[i]);
                        new_strides.push(input_strides[i]);
                    }
                }
                // Si toutes les dimensions étaient 1 (ex: [1,1,1]), new_shape est vide.
                // Un tenseur de shape vide est un scalaire.
                // Strides pour un scalaire est aussi vide.
            }
        }
    }
    
    // Si new_shape est vide et que le tenseur original n'était pas un scalaire,
    // cela signifie que toutes ses dimensions étaient 1. Le résultat est un scalaire (shape []).
    // new_strides devrait aussi être vide dans ce cas.
    // La logique ci-dessus devrait déjà gérer cela en ne poussant rien si toutes les dims sont 1.

    let requires_grad_input = input_guard.requires_grad;
    let input_node_id = tensor.node_id();
    let buffer_arc = input_guard.buffer.clone();
    let device = input_guard.device;
    let offset = input_guard.offset;
    let original_shape_clone = input_shape.clone(); // Pour SqueezeBackward

    drop(input_guard);

    let view_td = TensorData::new_view(
        buffer_arc,
        device,
        offset,
        new_shape, // new_shape calculée
        new_strides, // new_strides correspondants
    )?;

    let output_tensor = Tensor { data: Arc::new(RwLock::new(view_td)) };

    if requires_grad_input {
        // Seulement créer un grad_fn si la shape a effectivement changé.
        // Si squeeze n'a rien fait, output_tensor est un clone de l'input en termes de shape/strides.
        // Cependant, pour la chaîne d'autograd, il est peut-être plus simple de toujours ajouter le noeud.
        // Si input_shape == output_tensor.shape(), on pourrait techniquement retourner tensor.clone().
        // Mais pour être cohérent avec d'autres ops de vue, on crée une nouvelle vue.
        let grad_fn_arc = Arc::new(SqueezeBackward {
            input_node_id,
            original_shape: original_shape_clone, 
        }) as Arc<dyn BackwardOp>;
        output_tensor.set_grad_fn(Some(grad_fn_arc))?;
        output_tensor.set_requires_grad(true)?;
    }

    Ok(output_tensor)
} 