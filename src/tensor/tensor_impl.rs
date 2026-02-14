use std::sync::{Arc, RwLock};
use crate::tensor::storage::{Storage, next_uid};
use crate::autograd::node::Node;
use std::fmt;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8, // Quantized
    I4, // Packed quantized
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
            DType::I4 => 0, // Special handling needed usually, but packed means 0.5 bytes? 
                            // Usually we just say block size dictates byte size. 
                            // For indexing, we might say 1 byte contains 2 I4s.
        }
    }
}

pub type Shape = Vec<usize>;
pub type Strides = Vec<usize>;

pub struct Tensor {
    pub(crate) id: usize,
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) storage: Arc<Storage>,
    pub(crate) offset: usize, // Byte offset
    pub(crate) dtype: DType,
    
    // Autograd
    pub(crate) requires_grad: bool,
    pub(crate) grad: Arc<RwLock<Option<Tensor>>>,
    pub(crate) ctx: Option<Box<dyn Node>>,
}

impl Tensor {
    pub fn new(
        storage: Arc<Storage>,
        shape: Shape,
        strides: Strides,
        offset: usize,
        dtype: DType,
        requires_grad: bool,
    ) -> Self {
        Self {
            id: next_uid(),
            shape,
            strides,
            storage,
            offset,
            dtype,
            requires_grad,
            grad: Arc::new(RwLock::new(None)),
            ctx: None,
        }
    }


    pub fn id(&self) -> usize {
        self.id
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn is_contiguous(&self) -> bool {
        // Simple check: stride[i] == stride[i+1] * shape[i+1]
        if self.shape.is_empty() { return true; }
        let mut expected_stride = 1;
        for (dim, stride) in self.shape.iter().zip(self.strides.iter()).rev() {
            if *stride != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }
    
    // Helper to calculate default strides
    pub fn default_strides(shape: &[usize]) -> Strides {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }
    
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn t(&self) -> Self {
        assert!(self.shape.len() >= 2, "Transpose requires at least 2 dimensions");
        let ndim = self.shape.len();
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        
        // Swap last two dimensions for simple 2D transpose or multi-dim last-two swap
        new_shape.swap(ndim - 1, ndim - 2);
        new_strides.swap(ndim - 1, ndim - 2);
        
        Self {
            id: next_uid(),
            shape: new_shape,
            strides: new_strides,
            storage: self.storage.clone(),
            offset: self.offset,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            ctx: None,
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            id: next_uid(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            storage: self.storage.clone(),
            offset: self.offset,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)), // New tensor, new grad buffer
            ctx: None,
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl Tensor {


    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let storage = Arc::new(Storage::new(dtype.size_of() * shape.iter().product::<usize>()));
        // Storage is zeroed by alloc or need explicit zeroing? `alloc` doesn't zero.
        // `alloc_zeroed` exists.
        // For now, let's memset in Storage or just Loop.
        // Actually Storage::new uses `alloc`.
        unsafe {
             std::ptr::write_bytes(storage.as_ptr() as *mut u8, 0, storage.len());
        }
        
        let strides = Self::default_strides(&shape);
        Self::new(storage, shape, strides, 0, dtype, false)
    }

    pub fn ones(shape: Shape, dtype: DType) -> Self {
        let t = Self::zeros(shape, dtype);
        // Fill with 1. Need ops::fill generic.
        // Prototyping: just F32 support for now
        if dtype == DType::F32 {
            let data = unsafe { std::slice::from_raw_parts_mut(t.storage.as_ptr() as *mut f32, t.numel()) };
            for x in data.iter_mut() { *x = 1.0; }
        }
        t
    }
    
    pub fn from_vec_f32(data: Vec<f32>, shape: Shape) -> Self {
        let numel = shape.iter().product();
        assert_eq!(data.len(), numel);
        
        let size_bytes = numel * 4;
        let mut storage = Storage::new(size_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), storage.as_mut_ptr() as *mut f32, numel);
        }
        
        Self::new(Arc::new(storage), shape.clone(), Self::default_strides(&shape), 0, DType::F32, false)
    }

    /// Internal helper to accumulate gradient.
    pub fn add_grad(&self, grad: Tensor) {
        // lock is RwLock<Option<Tensor>>
        let mut lock = self.grad.write().unwrap();
        
        if let Some(existing_grad) = lock.as_mut() {
             // Perform existing_grad += grad
             // Naive F32 loop
             // In reality we should use the `Add` op kernel here? 
             // But `Add` op returns a NEW tensor and records to graph.
             // Here we just want raw in-place addition.
             if self.dtype == DType::F32 {
                 // Storage is Arc<Storage>. We use as_ptr() and then cast to *mut. 
                 // This is technically interior mutability at the raw pointer level.
                 let lhs_ptr = self.storage.as_ptr() as *mut f32;
                 let rhs_ptr = grad.storage.as_ptr() as *const f32;
                 
                 let lhs = unsafe { std::slice::from_raw_parts_mut(lhs_ptr.add(existing_grad.offset / 4), existing_grad.numel()) };
                 let rhs = unsafe { std::slice::from_raw_parts(rhs_ptr.add(grad.offset / 4), grad.numel()) };
                 
                 if existing_grad.is_contiguous() && grad.is_contiguous() {
                     for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
                         *l += *r;
                     }
                 } else {
                     panic!("Gradient accumulation for non-contiguous tensors not yet implemented");
                 }
             }
        } else {
            // First gradient. Just take it.
            // Ideally we should clone the data of `grad` into a new Tensor that owns its storage,
            // because `grad` passed in might be a view or temporary.
            // For now, let's assume `grad` is a result of an op and we can just store it.
            *lock = Some(grad);
        }
    }
}

