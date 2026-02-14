use crate::tensor::Tensor;
use std::collections::HashSet;

/// Runs the backward pass starting from `root`.
/// `root` is usually the loss value (scalar).
pub fn backward(root: &Tensor) {
    // 1. Topological Sort
    let mut sorted_nodes = Vec::new();
    let mut visited = HashSet::new();


    // The root might not have a ctx if it's a leaf, but usually backward is called on a result of an op.
    // If it's a leaf (created by user), ctx is None.
    
    // We traverse TENSORS, not just Nodes, because we need to accumulate gradients into Tensors.
    // Actually, we traverse the graph of Tensors connected by Nodes.
    
    // Standard DFS for Topo Sort
    fn dfs(tensor: &Tensor, visited: &mut HashSet<usize>, sorted: &mut Vec<Tensor>) {
        if visited.contains(&tensor.id) {
            return;
        }
        visited.insert(tensor.id);

        if let Some(ctx) = &tensor.ctx {
            for parent in ctx.parents() {
                dfs(&parent, visited, sorted);
            }
        }
        sorted.push(tensor.clone());
    }

    dfs(root, &mut visited, &mut sorted_nodes);

    // 2. Initialize Gradient of Root to 1.0 (if not set)
    // We assume root is a scalar F32 for now.
    {
        // TODO: check if root is scalar
        // let one = Tensor::ones(root.shape(), root.dtype());
        // root.accumulate_grad(one);
        // For now, let's assume the user seeds the gradient or we do it here.
        // If we implement `ones` later, we can use it.
        // For now, let's just make the user manually seed logic or placeholder.
        // Wait, I should implement `ones` capability or `accumulate_grad`.
    }

    // 3. Reverse Iterate and Propagate
    for tensor in sorted_nodes.iter().rev() {
        let grad = {
            let lock = tensor.grad.read().unwrap();
            if lock.is_none() { 
                continue; 
            }
            lock.as_ref().unwrap().clone()
        };

        if let Some(ctx) = &tensor.ctx {
            let grads = ctx.backward(&grad);
            let parents = ctx.parents();
            
            if grads.len() != parents.len() {
                panic!("Gradient count mismatch for node {:?}", ctx);
            }

            for (parent, parent_grad) in parents.iter().zip(grads.into_iter()) {
                if parent.requires_grad {
                   parent.add_grad(parent_grad);
                }
            }
        }
    }
}

