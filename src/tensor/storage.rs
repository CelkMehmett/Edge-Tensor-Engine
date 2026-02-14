use std::alloc::{alloc, dealloc, Layout, handle_alloc_error};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Aligned memory storage for Tensor data.
/// Uses 32-byte alignment to support AVX2.
#[derive(Debug)]
pub struct Storage {
    ptr: NonNull<u8>,
    layout: Layout,
    size_bytes: usize,
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    const ALIGNMENT: usize = 32;

    pub fn new(size_bytes: usize) -> Self {
        if size_bytes == 0 {
            return Self {
                ptr: NonNull::dangling(),
                layout: Layout::from_size_align(0, Self::ALIGNMENT).unwrap(),
                size_bytes: 0,
            };
        }

        let layout = Layout::from_size_align(size_bytes, Self::ALIGNMENT)
            .expect("Invalid layout definition");
        
        let ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                handle_alloc_error(layout);
            }
            NonNull::new_unchecked(p)
        };

        Self {
            ptr,
            layout,
            size_bytes,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.size_bytes
    }
    
    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }
    
    /// Returns a slice of the bytes.
    /// Safety: Caller must ensure data is initialized if reading.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size_bytes) }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size_bytes) }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.size_bytes > 0 {
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

// Simple unique ID generator for tensors/storage tracking
static UID_COUNTER: AtomicUsize = AtomicUsize::new(1);

pub fn next_uid() -> usize {
    UID_COUNTER.fetch_add(1, Ordering::Relaxed)
}
