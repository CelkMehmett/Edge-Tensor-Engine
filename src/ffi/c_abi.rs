use crate::tensor::{Tensor, Shape, DType};
use crate::ops::matmul::{matmul, matmul_int4};
use crate::autograd::backward;

use std::slice;

// Helper to convert C array to Shape
unsafe fn c_shape_to_vec(shape_ptr: *const i64, ndim: usize) -> Shape {
    let slice = slice::from_raw_parts(shape_ptr, ndim);
    slice.iter().map(|&x| x as usize).collect()
}

// --- Creation & Destruction ---

#[no_mangle]
pub extern "C" fn tensor_create_f32(data: *const f32, shape_ptr: *const i64, ndim: usize) -> *mut Tensor {
    // Catch unwinds? Rust FFI should catch panics, otherwise undefined behavior on unwind across FFI.
    // For prototype, we skip `std::panic::catch_unwind` but in prod usage it IS MANDATORY.
    
    let shape = unsafe { c_shape_to_vec(shape_ptr, ndim) };
    let numel = shape.iter().product();
    let mut data_vec = Vec::with_capacity(numel);
    unsafe {
        std::ptr::copy_nonoverlapping(data, data_vec.as_mut_ptr(), numel);
        data_vec.set_len(numel);
    }
    
    let tensor = Tensor::from_vec_f32(data_vec, shape);
    Box::into_raw(Box::new(tensor))
}

#[no_mangle]
pub extern "C" fn tensor_zeros(shape_ptr: *const i64, ndim: usize, dtype_code: i32) -> *mut Tensor {
    let shape = unsafe { c_shape_to_vec(shape_ptr, ndim) };
    let dtype = match dtype_code {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::I8, // Used for INT4 packing container?
        3 => DType::I4,
        _ => DType::F32, // Default
    };
    
    let tensor = Tensor::zeros(shape, dtype);
    Box::into_raw(Box::new(tensor))
}

#[no_mangle]
pub extern "C" fn tensor_free(ptr: *mut Tensor) {
    if ptr.is_null() { return; }
    unsafe {
        let _ = Box::from_raw(ptr); // Drop happens here
    }
}

// --- Operations ---

#[no_mangle]
pub extern "C" fn tensor_matmul(lhs: *const Tensor, rhs: *const Tensor) -> *mut Tensor {
    assert!(!lhs.is_null() && !rhs.is_null());
    let lhs = unsafe { &*lhs };
    let rhs = unsafe { &*rhs };
    
    let result = matmul(lhs, rhs);
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub extern "C" fn tensor_linear_int4(
    input: *const Tensor, 
    weight_packed: *const Tensor, 
    scales: *const Tensor,
    bias: *const Tensor // Optional, can be null
) -> *mut Tensor {
    let input = unsafe { &*input };
    let w = unsafe { &*weight_packed };
    let s = unsafe { &*scales };
    let b = if bias.is_null() { None } else { Some(unsafe { (&*bias).clone() }) }; // Clone ref/struct, storage shared
    
    // Convert bias Option<&Tensor> to Option<Tensor> for the function if needed, 
    // or change function signature. My `matmul_int4` expected `&Option<Tensor>`.
    // Wait, `matmul_int4` signature: `bias: &Option<Tensor>`.
    
    let result = matmul_int4(input, w, s, &b);
    Box::into_raw(Box::new(result))
}

// --- Autograd ---

#[no_mangle]
pub extern "C" fn tensor_backward(root: *const Tensor) {
    assert!(!root.is_null());
    let root = unsafe { &*root };
    backward(root);
}

#[no_mangle]
pub extern "C" fn tensor_grad(tensor: *const Tensor) -> *mut Tensor {
    assert!(!tensor.is_null());
    let t = unsafe { &*tensor };
    
    // Return gradient tensor if exists.
    // If we just want to peek, we can return a copy (view).
    // The `grad` field is `Arc<RwLock<Option<Tensor>>>`.
    
    let lock = t.grad.read().unwrap();
    if let Some(g) = lock.as_ref() {
        // Return a clone (new Tensor struct pointing to same storage)
        Box::into_raw(Box::new(g.clone()))
    } else {
        std::ptr::null_mut()
    }
}

// --- Accessors ---

#[no_mangle]
pub extern "C" fn tensor_data_ptr(tensor: *const Tensor) -> *const f32 {
    let t = unsafe { &*tensor };
    // Assuming F32 for now.
    // Unsafe access, only valid while Tensor is alive.
    t.storage.as_ptr() as *const f32
}

#[no_mangle]
pub extern "C" fn tensor_get_shape(tensor: *const Tensor, out_ndim: *mut usize) -> *const usize {
    let t = unsafe { &*tensor };
    unsafe { *out_ndim = t.shape.len(); }
    t.shape.as_ptr()
}
