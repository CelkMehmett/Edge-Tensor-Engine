use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use edge_tensor_engine::Tensor;
use edge_tensor_engine::tensor::DType;

fn matmul_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    
    // Test different matrix sizes
    let sizes = vec![
        (64, 64, 64),    // Small
        (128, 128, 128), // Medium
        (256, 256, 256), // Large
        (512, 512, 512), // Very Large
    ];
    
    for (m, k, n) in sizes {
        let parameter_string = format!("{}x{}x{}", m, k, n);
        
        group.bench_with_input(
            BenchmarkId::new("F32", &parameter_string),
            &(m, k, n),
            |bencher, &(m, k, n)| {
                let a = Tensor::ones(vec![m, k], DType::F32);
                let b = Tensor::ones(vec![k, n], DType::F32);
                
                bencher.iter(|| {
                    let _c = edge_tensor_engine::ops::matmul::matmul(
                        black_box(&a),
                        black_box(&b)
                    );
                });
            },
        );
    }
    
    group.finish();
}

fn tensor_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    
    group.bench_function("zeros_1024x1024", |b| {
        b.iter(|| {
            let _t = Tensor::zeros(black_box(vec![1024, 1024]), DType::F32);
        });
    });
    
    group.bench_function("ones_1024x1024", |b| {
        b.iter(|| {
            let _t = Tensor::ones(black_box(vec![1024, 1024]), DType::F32);
        });
    });
    
    group.finish();
}

fn binary_ops_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_ops");
    
    let a = Tensor::ones(vec![1024, 1024], DType::F32);
    
    group.bench_function("add_1024x1024", |bencher| {
        bencher.iter(|| {
            let _c = edge_tensor_engine::ops::binary::add(
                black_box(&a),
                black_box(&a)
            );
        });
    });
    
    group.finish();
}

criterion_group!(benches, matmul_benchmark, tensor_creation_benchmark, binary_ops_benchmark);
criterion_main!(benches);
