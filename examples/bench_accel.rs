use std::time::Instant;
use wildandev_core::accel::gemm_forward;
use wildandev_core::tensor::Tensor;
use wildandev_core::Elis;

fn main() {
    let (m, k, n) = (128, 512, 512);
    let a = vec![0.01f32; m * k];
    let b = vec![0.02f32; k * n];
    let bias = vec![0.05f32; n];
    let mut c = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..10 {
        gemm_forward(&a, &b, &bias, &mut c, m, k, n);
    }

    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        gemm_forward(&a, &b, &bias, &mut c, m, k, n);
    }
    let elapsed = t0.elapsed();
    let per_iter = elapsed.as_secs_f64() / iters as f64;
    let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (per_iter * 1e9);

    println!("wildandev_accel AVX2 MatMul: {:.3} ms/iter ({:.2} GFLOPS)", per_iter * 1000.0, gflops);
}
