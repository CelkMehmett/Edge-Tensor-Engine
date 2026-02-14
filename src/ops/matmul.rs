use crate::tensor::{Tensor, DType};
use crate::autograd::node::Node;


#[derive(Debug)]
pub struct MatmulNode {
    lhs: Tensor,
    rhs: Tensor,
}

impl Node for MatmulNode {
    fn parents(&self) -> Vec<Tensor> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        // C = A @ B
        // dA = grad @ B.T
        // dB = A.T @ grad
        
        let dlhs = matmul(grad, &self.rhs.t());
        let drhs = matmul(&self.lhs.t(), grad);
        
        vec![dlhs, drhs]
    }
}

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    assert_eq!(lhs.dtype, DType::F32, "Only F32 supported for now");
    assert_eq!(rhs.dtype, DType::F32);
    
    // Shapes: [..., M, K] x [..., K, N] -> [..., M, N]
    // Simplify: 2D only for now
    assert_eq!(lhs.shape.len(), 2);
    assert_eq!(rhs.shape.len(), 2);
    
    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let k2 = rhs.shape[0];
    let n = rhs.shape[1];
    
    assert_eq!(k, k2, "Dimension mismatch");
    
    let _blocks_m = (m + 31) / 32;
    let _blocks_n = (n + 31) / 32;
    let _blocks_k = (k + 31) / 32;
    // 32 is a small block size, usually 64-256 for L2 cache.
    // L1 cache blocking is typically smaller (register tile).
    
    let mut output = Tensor::zeros(vec![m, n], DType::F32);
    
    // Pointers
    let a_ptr = lhs.storage.as_slice().as_ptr(); // TODO: Add offset support
    let b_ptr = rhs.storage.as_slice().as_ptr();
    let c_ptr = output.storage.as_ptr() as *mut f32; // Use raw pointer from shared storage
    
    // Naive Tiled Implementation
    // Optimizations to add: SIMD, Register Blocking, Cache Blocking
    // This is essentially just a triple loop but structured for adding tiling later.
    
    // Current: Simple Triple Loop
    // Use raw pointers for speed (unsafe)
    unsafe {
        let a_data = std::slice::from_raw_parts(a_ptr as *const f32, lhs.numel());
        let b_data = std::slice::from_raw_parts(b_ptr as *const f32, rhs.numel());
        let c_data = std::slice::from_raw_parts_mut(c_ptr as *mut f32, output.numel());
        
        let a_stride_m = lhs.strides[0];
        let a_stride_k = lhs.strides[1];
        let b_stride_k = rhs.strides[0];
        let b_stride_n = rhs.strides[1];
        let c_stride_m = output.strides[0];
        let c_stride_n = output.strides[1];

        // Parallelize m dimension with Rayon if available? 
        // For now single threaded.
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k { // dot product
                     let a_val = *a_data.get_unchecked(i * a_stride_m + p * a_stride_k);
                     let b_val = *b_data.get_unchecked(p * b_stride_k + j * b_stride_n);
                     sum += a_val * b_val;
                }
                *c_data.get_unchecked_mut(i * c_stride_m + j * c_stride_n) = sum;
            }
        }
    }
    
    // Attach graph
    if lhs.requires_grad || rhs.requires_grad {
        output.requires_grad = true;
        output.ctx = Some(Box::new(MatmulNode {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }
    
    output
}

pub fn matmul_int4(input: &Tensor, weight_packed: &Tensor, scales: &Tensor, bias: &Option<Tensor>) -> Tensor {
    // specialized forward pass
    // input: [M, K]
    // weight: [N, K/2]
    // output: [M, N]
    
    let m = input.shape[0];
    let k = input.shape[1];
    let n = weight_packed.shape[0];
    
    // assert k % 2 == 0
    
    let output = Tensor::zeros(vec![m, n], DType::F32);
    
     unsafe {
          let a_ptr = input.storage.as_ptr() as *const f32;
          let w_ptr = weight_packed.storage.as_ptr() as *const u8;
          let s_ptr = scales.storage.as_ptr() as *const f32;
          let c_ptr = output.storage.as_ptr() as *mut f32;

         
         let a_stride = input.strides[0]; // M stride
         let w_stride = weight_packed.strides[0]; // N stride in packed
         
         for i in 0..m {
             for j in 0..n {
                 let scale = *s_ptr.add(j);
                 let mut sum = 0.0;
                 
                 for p in 0..(k/2) {
                     let packed = *w_ptr.add(j * w_stride + p);
                     let low = (packed & 0x0F) as i8;
                     let high = ((packed >> 4) & 0x0F) as i8;
                     // Convert raw 4-bit uint to signed int if needed (usually offset binary or 2s comp)
                     // Let's assume offset binary for simplicity or just raw value * scale.
                     // A standard scheme: val = (nibble - 8) * scale
                     
                     let val_low = (low as f32 - 8.0) * scale;
                     let val_high = (high as f32 - 8.0) * scale;
                     
                     let a_val1 = *a_ptr.add(i * a_stride + (p * 2));
                     let a_val2 = *a_ptr.add(i * a_stride + (p * 2 + 1));
                     
                     sum += a_val1 * val_low + a_val2 * val_high;
                 }
                 
                 *c_ptr.add(i * output.strides[0] + j) = sum;
             }
         }
    }
    
    if let Some(_b) = bias {
        // Add bias
        // crate::ops::binary::add(&output, b)
    }
    
    output
}

