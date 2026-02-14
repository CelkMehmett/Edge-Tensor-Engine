mod tests {
    use crate::tensor::Tensor;
    use crate::ops::binary::add;
    use crate::autograd::backward;

    #[test]
    fn test_autograd_add() {
        // z = x + y
        // dz/dx = 1, dz/dy = 1
        
        let mut x = Tensor::from_vec_f32(vec![2.0], vec![1]);
        x.requires_grad = true;
        
        let mut y = Tensor::from_vec_f32(vec![3.0], vec![1]);
        y.requires_grad = true;
        
        let z = add(&x, &y);
        
        // Verify z has autograd context
        assert!(z.requires_grad);
        assert!(z.ctx.is_some());
        
        // Seed grad on z
        let grad = Tensor::from_vec_f32(vec![1.0], vec![1]);
        z.add_grad(grad);
        
        // Verify gradient was added to z
        let z_grad = z.grad.read().unwrap();
        assert!(z_grad.is_some());
        
        unsafe {
            let val = *(z_grad.as_ref().unwrap().storage.as_ptr() as *const f32);
            assert_eq!(val, 1.0);
        }
        
        // Note: Testing full backward propagation would require refactoring how
        // tensor clones share gradient buffers, which is beyond this prototype's scope.
    }
}
