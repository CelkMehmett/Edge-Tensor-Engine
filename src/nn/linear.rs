use crate::tensor::{Tensor, DType};
use crate::ops::matmul::matmul;
use crate::ops::binary::add;


pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Random init later. For now Zeros/Ones.
        let weight = Tensor::ones(vec![out_features, in_features], DType::F32);
        let bias = if bias {
            Some(Tensor::zeros(vec![out_features], DType::F32))
        } else {
            None
        };
        
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // y = x @ W.T + b
        // x: [Batch, In]
        // W: [Out, In]
        // W.T: [In, Out]
        // y: [Batch, Out]
        
        // Matmul expects explicit Tensors. 
        // Our matmul is naive, expects (M, K) x (K, N)
        // input: (Batch, In), weight.t(): (In, Out)
        
        // Note: Generic `t()` creates a view.
        let out = matmul(input, &self.weight.t());
        
        if let Some(b) = &self.bias {
            // Broadcasting add needed.
            // b is [Out]. out is [Batch, Out].
            // Current add implementation is naive elementwise same shape.
            // Need broadcast support. Assuming Batch=1 for now or implementing naive broadcast in Add later.
            // For now, let's just assert Batch=1 or ignore bias if shapes mismatch... 
            // Better: implement basic broadcast in `add` logic?
            // Or just loop here.
            add(&out, b) // This will panic if shapes mismatch
        } else {
            out
        }
    }
}

// Q8 or Q4 Linear Layer
pub struct LinearInt4 {
    pub weight_packed: Tensor, // Type I8/U8, each byte has 2 nibbles
    pub scales: Tensor,        // F32 scales per channel (output channel)
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearInt4 {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Packed size: [Out, In / 2]
        let packed_shape = vec![out_features, in_features / 2];
        let weight_packed = Tensor::zeros(packed_shape, DType::I8); // pretend I8 is byte
        let scales = Tensor::ones(vec![out_features], DType::F32);
        
        Self {
            weight_packed,
            scales,
            bias: None,
            in_features,
            out_features,
        }
    }
    
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // x: [Batch, In]
        // W: [Out, In] (Packed)
        // y: x @ W.T
        
        // We need a specialized kernel `matmul_f32_int4`.
        // Unpacking generic Tensor operations is hard.
        // We implement a custom Op for this.
        
        // Placeholder for the kernel call
        crate::ops::matmul::matmul_int4(input, &self.weight_packed, &self.scales, &self.bias)
    }
}
