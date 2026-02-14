use crate::tensor::{Tensor, DType};
use crate::autograd::node::Node;

#[derive(Debug)]
pub struct ReluNode {
    input: Tensor,
    output_cache: Tensor, // Need output for backward: grad = grad_output * (output > 0)
}

impl Node for ReluNode {
    fn parents(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        // grad_input = grad * (input > 0)
        // We can use output > 0 too (since relu(x) = x if x > 0 else 0)
        
        let grad_input = Tensor::zeros(grad.shape.clone(), grad.dtype);
        
        if grad.dtype == DType::F32 {
            unsafe {
                 let g_ptr = grad.storage.as_ptr().add(grad.offset) as *const f32;
                 let out_ptr = self.output_cache.storage.as_ptr().add(self.output_cache.offset) as *const f32;
                 let gi_ptr = grad_input.storage.as_ptr() as *mut f32;
                 
                 for i in 0..grad.numel() {
                     let mask = if *out_ptr.add(i) > 0.0 { 1.0 } else { 0.0 };
                     *gi_ptr.add(i) = *g_ptr.add(i) * mask;
                 }
            }
        }
        
        vec![grad_input]
    }
}

pub fn relu(input: &Tensor) -> Tensor {
    let output = Tensor::zeros(input.shape.clone(), input.dtype);
     
    if input.dtype == DType::F32 {
        unsafe {
             let in_ptr = input.storage.as_ptr().add(input.offset) as *const f32;
             let out_ptr = output.storage.as_ptr() as *mut f32;
             
             for i in 0..input.numel() {
                 let val = *in_ptr.add(i);
                 *out_ptr.add(i) = if val > 0.0 { val } else { 0.0 };
             }
        }
    }
    
    if input.requires_grad {
        let mut out = output.clone();
        out.requires_grad = true;
        out.ctx = Some(Box::new(ReluNode { input: input.clone(), output_cache: output }));
        return out;
    }
    
    output
}
