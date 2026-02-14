mod tests {
    use crate::tensor::Tensor;
    use crate::ops::matmul::matmul;

    #[test]
    fn test_matmul_f32() {
        // [2, 2] x [2, 2] Identity
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        
        let c = matmul(&a, &b);
        
        unsafe {
            let ptr = c.storage.as_ptr() as *const f32;
            assert_eq!(*ptr, 1.0);
            assert_eq!(*ptr.add(1), 2.0);
            assert_eq!(*ptr.add(2), 3.0);
            assert_eq!(*ptr.add(3), 4.0);
        }
    }
}
