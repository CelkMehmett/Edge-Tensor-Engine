use crate::tensor::{Tensor, DType};

pub struct KVCache {
    pub k: Tensor, // [MaxSeq, Head, Dim]
    pub v: Tensor, // [MaxSeq, Head, Dim]
    pub max_seq_len: usize,
    pub current_pos: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, head: usize, dim: usize) -> Self {
        let k = Tensor::zeros(vec![max_seq_len, head, dim], DType::F32);
        let v = Tensor::zeros(vec![max_seq_len, head, dim], DType::F32);
        Self {
            k,
            v,
            max_seq_len,
            current_pos: 0,
        }
    }
    
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor, pos: usize) {
        // new_k: [1, Head, Dim] (single token update) or [Len, Head, Dim]
        // Copy new_k into self.k at slice [pos..pos+len]
        
        // This requires `copy_from` or slice update method in Tensor.
        // Using unsafe ptr copy for now.
        
        unsafe {
             let k_dst = self.k.storage.as_ptr() as *mut f32;
             let v_dst = self.v.storage.as_ptr() as *mut f32;
             
             let k_src = new_k.storage.as_ptr().add(new_k.offset) as *const f32;
             let v_src = new_v.storage.as_ptr().add(new_v.offset) as *const f32;
             
             // Stride math needed here locally.
             // Assume contiguous [Seq, Head, Dim] layout for cache.
             // pos corresponds to the first dimension index.
             let stride_0 = self.k.strides[0]; 
             let num_to_copy = new_k.numel(); // Assume dense update
             
             let offset = pos * stride_0;
             
             // Bounds check
             if pos + new_k.shape[0] > self.max_seq_len {
                 panic!("KV Cache overflow");
             }
             
             // This naive copy assumes `new_k` shape matches [Len, Head, Dim] flattening perfectly to the slice
             std::ptr::copy_nonoverlapping(k_src, k_dst.add(offset), num_to_copy);
             std::ptr::copy_nonoverlapping(v_src, v_dst.add(offset), num_to_copy);
        }
        
        self.current_pos = pos + new_k.shape[0];
    }
    
    pub fn get_view(&self, _len: usize) -> (Tensor, Tensor) {
        // Return view of k, v up to len
        // Slicing not fully implemented in Tensor generic view yet.
        // Returning full cache for now or manually constructing view.
        (self.k.clone(), self.v.clone()) 
    }
}
