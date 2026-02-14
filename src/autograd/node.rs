use crate::tensor::Tensor;
use std::fmt::Debug;

/// A Node in the computational graph.
/// Knows how to compute gradients for its inputs given its own gradient.
pub trait Node: Debug + Send + Sync {
    /// Returns the input tensors (parents) of this node.
    fn parents(&self) -> Vec<Tensor>;

    /// Computes gradients for parents given the gradient of the output.
    /// Returns a vector of gradients, one for each parent, in the same order.
    fn backward(&self, grad: &Tensor) -> Vec<Tensor>;
}
