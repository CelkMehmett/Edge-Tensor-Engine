use crate::tensor::{Tensor, DType};
use crate::autograd::node::Node;


#[derive(Debug)]
pub struct AddNode {
    lhs: Tensor,
    rhs: Tensor,
}

impl Node for AddNode {
    fn parents(&self) -> Vec<Tensor> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        // z = x + y -> dz/dx = 1 * grad, dz/dy = 1 * grad
        // Need to handle broadcasting reduction!
        // For now assuming same shape.
        vec![grad.clone(), grad.clone()]
    }
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    // Basic elementwise add (assuming same shape/strides for now)
    assert_eq!(lhs.shape, rhs.shape, "Broadcasting not implemented yet");
    
    let output = Tensor::zeros(lhs.shape.clone(), lhs.dtype);
    
    // Unsafe fast path
    if lhs.dtype == DType::F32 {
         unsafe {
             let a_ptr = lhs.storage.as_ptr().add(lhs.offset) as *const f32;
             let b_ptr = rhs.storage.as_ptr().add(rhs.offset) as *const f32;
             let c_ptr = output.storage.as_ptr() as *mut f32;
             
             for i in 0..lhs.numel() {
                 *c_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
             }
         }
    }
    
    if lhs.requires_grad || rhs.requires_grad {
        // output.requires_grad = true; // Mutating output inside? Tensor::zeros returns new struct.
        // But `output` is owned mut here? No it's not.
        // Wait, Tensor::zeros returns Self.
        let mut out = output;
        out.requires_grad = true;
        out.ctx = Some(Box::new(AddNode { lhs: lhs.clone(), rhs: rhs.clone() }));
        return out;
    }

    output
}
