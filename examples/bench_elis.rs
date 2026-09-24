use std::time::Instant;
use wildandev_core::rng::Rng;
use wildandev_core::tensor::Tensor;
use wildandev_core::Elis;

fn main() {
    println!("=== wildandev Elis XOR training (forward pakai AVX) ===");
    let x = Tensor::new(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let labels = vec![0usize, 1, 1, 0];
    let mut elis = Elis::new(&[2, 8, 2], 1234);

    let mut loss = f32::INFINITY;
    let t0 = Instant::now();
    for _ in 0..3000 {
        loss = elis.forward(&x, &labels);
        elis.backward();
        elis.sgd_step(0.5);
    }
    let dt = t0.elapsed();
    println!("3000 steps in {:?}, final loss {:.6}", dt, loss);
    println!("preds {:?}", elis.predict(&x));
}
