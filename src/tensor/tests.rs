#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, DType};


    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.numel(), 6);
        assert!(!t.requires_grad);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec_f32(data.clone(), vec![2, 2]);
        
        unsafe {
            let ptr = t.storage.as_ptr() as *const f32;
            assert_eq!(*ptr, 1.0);
            assert_eq!(*ptr.add(3), 4.0);
        }
    }

    #[test]
    fn test_view_transpose() {
        let t = Tensor::zeros(vec![2, 3], DType::F32);
        let t_t = t.t();
        assert_eq!(t_t.shape(), &[3, 2]);
        assert_eq!(t_t.strides(), &[1, 3]); // Swapped
    }
}
